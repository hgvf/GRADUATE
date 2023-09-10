import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import pickle
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot_path', type=str, default='none')
    parser.add_argument('--model_opt', type=str)
    parser.add_argument('--upper_snr', type=float, default=-1)
    parser.add_argument('--lower_snr', type=float, default=-1)
    parser.add_argument('--upper_epidis', type=float, default=-1)
    parser.add_argument('--lower_epidis', type=float, default=-1)
    parser.add_argument('--data_type', type=str, default='other') # noise, other
    parser.add_argument('--dataset_opt', type=str, default='taiwan')
    parser.add_argument('--p_timestep', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--plot', type=bool, default=False)

    parser.add_argument('--wavelength', type=int, default=3000)
    parser.add_argument('--max_freq', type=int, default=12)
    parser.add_argument('--n_segmentation', type=int, default=5)
    parser.add_argument('--seg_proj_type', type=str, default='none')
    parser.add_argument('--label_type', type=str, default='all')

    parser.add_argument("--threshold_type", type=str, default='all')
    parser.add_argument('--threshold_prob', type=float, default=0.15)
    parser.add_argument('--threshold_trigger', type=int, default=5)

    opt = parser.parse_args()

    return opt

def set_generators(opt):
    cwbsn, tsmip, stead, cwbsn_noise, instance = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'stead':
       test = stead.test()
    elif opt.dataset_opt == 'instance':
        test = instance.test()
    elif opt.dataset_opt == 'taiwan':
        cwbsn_test = cwbsn.test()
        tsmip_test = tsmip.test()
        
        if opt.data_type == 'noise':
            cwbsn_noise_test = cwbsn_noise.test()

            test = cwbsn_test + tsmip_test + cwbsn_noise_test
        else:
            test = cwbsn_test + tsmip_test
    elif opt.dataset_opt == 'cwbsn_noise' and opt.data_type == 'noise':
        test = cwbsn_noise.test()

    print(f'total traces -> test: {len(test)}')

    test_generator = sbg.GenericGenerator(test)

    # set generator with or without augmentations
    if opt.dataset_opt == 'instance':
        phase_dict = ['trace_P_arrival_sample']
    else:
        phase_dict = ['trace_p_arrival_sample']

    ptime = opt.p_timestep  
    augmentations = basic_augmentations(opt, phase_dict=phase_dict, EEW=False, test=True, ptime=ptime)

    test_generator.add_augmentations(augmentations)

    return test_generator

def inference(opt, model, test_loader, device):
    pred, gt = [], [], 

    model.eval()
    with tqdm(test_loader) as epoch:
        idx = 0
        for data in epoch:
            idx += 1
            
            with torch.no_grad():
                if opt.model_opt != 'eqt':
                    _, out = model(data['X'].to(device), stft=data['stft'].float().to(device))
                    out = out[1].squeeze().detach().cpu().numpy()
                else:
                    out = model(data['X'].to(device))
                    out = out[1].detach().squeeze().cpu().numpy()

            target = data['y'][:, 0].squeeze().numpy()
            if type(out) == list:
                pass
            elif out.ndim == 2:
                pred += [out[i] for i in range(out.shape[0])]
                gt += [target[i] for i in range(target.shape[0])]
            else:
                pred += [out]
                gt += [target]

            # plt.figure(figsize=(8, 10))
            # plt.subplot(311)
            # plt.title('Z')
            # plt.axvline(x=int(opt.p_timestep), color='r')
            # plt.plot(data['X'][0, 0].T)
            # plt.subplot(312)
            # plt.title('N')
            # # plt.plot(data['y'][0, 0])
            # plt.axvline(x=int(opt.p_timestep), color='r')
            # plt.plot(data['X'][0, 1].T)
            # plt.subplot(313)
            # plt.title('E')
            # plt.axvline(x=int(opt.p_timestep), color='r')
            # # plt.plot(out[0].T)
            # plt.plot(data['X'][0, 2].T)
            # plt.xlabel('Time sample (0.01 s per sample)')
            # # plt.show()
            # plt.savefig(f"./fig/{idx}.png")
            # plt.clf()

    return pred, gt

def evaluation(pred, gt, threshold_prob, threshold_trigger, threshold_type):
    tp, fp, tn, fn = 0, 0, 0, 0 
    diff = []
    abs_diff = []
    res = []

    fn_idx, fp_idx = [], []

    sample_tolerant = 50
    for i in tqdm(range(len(pred))):
        pred_isTrigger = False
        gt_isTrigger = False
        gt_trigger = 0

        if not np.all(gt[i] == 0):
            gt_isTrigger = True            
            gt_trigger = np.argmax(gt[i])
            if gt[i][gt_trigger] < 0.3:
                gt_isTrigger = False
                gt_trigger = 0

        if threshold_type == 'single':
            a = np.where(pred[i] >= threshold_prob, 1, 0)

            if np.any(a):
                c = np.where(a==1)
                pred_isTrigger = True
                pred_trigger = c[0][0]
            else:
                pred_trigger = 0
        
        elif threshold_type == 'avg':
            a = pd.Series(pred[i])  
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                # pred_trigger = tri[0][0]-threshold_trigger+1
                pred_trigger = tri[0][0]
                pred_isTrigger = True
                
        elif threshold_type == 'continue':
            tmp = np.where(pred[i] >= threshold_prob, 1, 0)
            
            a = pd.Series(tmp)    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()
            pred_trigger = 0
            if threshold_trigger in data:
                # pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
                pred_trigger = data.index(threshold_trigger)
                pred_isTrigger = True
            else:
                pred_trigger = 0

        elif threshold_type == 'max':
            pred_trigger = np.argmax(pred[i]).item()
            
            if pred[i][pred_trigger] >= threshold_prob:
                pred_isTrigger = True
            else:
                pred_trigger = 0

        left_edge = (gt_trigger - sample_tolerant) if (gt_trigger - sample_tolerant) >= 0 else 0
        right_edge = (gt_trigger + sample_tolerant) if (gt_trigger + sample_tolerant) <= 3000 else 2999

        # case positive 
        if (pred_trigger >= left_edge) and (pred_trigger <= right_edge) and (pred_isTrigger) and (gt_isTrigger):
            tp += 1
            res.append('tp')
        elif (pred_isTrigger):
            if pred_trigger <= 50:
                tn == 1
                res.append('tn')
            else:
                fp += 1
                res.append('fp')
                fp_idx.append(i)

        # case negative
        if (not pred_isTrigger) and (gt_isTrigger):
            fn += 1
            res.append('fn')
            fn_idx.append(i)
        elif (not pred_isTrigger) and (not gt_isTrigger):
            tn += 1
            res.append('tn')

        if gt_isTrigger and pred_isTrigger:
            diff.append(pred_trigger-gt_trigger)
            abs_diff.append(abs(pred_trigger-gt_trigger))

    return tp, fp, tn, fn, diff, abs_diff, res, fn_idx, fp_idx

def plot(opt, test_loader, pred, fn_idx, fp_idx, res):
    max_plot = 200
    plot_cnt = 0
    with tqdm(test_loader) as epoch:
        idx = 0
        for data in epoch:
            if plot_cnt >= max_plot:
                break

            if idx in fn_idx or idx in fp_idx:
                pred_trigger = np.argmax(pred[idx])

                plt.figure(figsize=(12, 18))
                plt.subplot(511)
                plt.title('Z')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 0])

                plt.subplot(512)
                plt.title('N')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 1])

                plt.subplot(513)
                plt.title('E')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 2])

                plt.subplot(514)
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                    plt.title(f'Prediction: {int(pred_trigger)}')
                else:
                    plt.title('Prediction')
                plt.ylabel('Probability')
                plt.ylim([-0.05, 1.05])
                plt.legend()
                plt.plot(pred[idx])

                plt.subplot(515)
                plt.plot(data['y'][0, 0])
                plt.ylim([-0.05, 1.05])
                if res[idx] == 'fp':
                    gt_trigger = np.argmax(data['y'][0, 0])
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                    plt.title(f'Ground-truth: {int(gt_trigger)}')
                else:
                    plt.title('Ground-truth')
                plt.ylabel('Probability')
                plt.xlabel('Time sample (0.01 s per sample)')
                plt.legend()
                if opt.plot_path == 'none':
                    plt.savefig(f"./plot/{opt.model_opt}_{opt.dataset_opt}/{res[idx]}_{idx}.png", dpi=300)
                else:
                    tmp_path = f"./plot/{opt.model_opt}_{opt.dataset_opt}/{opt.plot_path}/"
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    plt.savefig(f"./plot/{opt.model_opt}_{opt.dataset_opt}/{opt.plot_path}/{res[idx]}_{idx}.png", dpi=300)
                plt.clf()
                plot_cnt += 1

            else:
                idx += 1
                continue

            idx += 1
            
def plot_zoomin(opt, test_loader, pred, fn_idx, fp_idx, res):
    max_plot = 200
    plot_cnt = 0
    with tqdm(test_loader) as epoch:
        idx = 0
        for data in epoch:
            if plot_cnt >= max_plot:
                break
            flag1, flag2 = False, False
            if idx in fn_idx or idx in fp_idx:
                pred_trigger = int(np.argmax(pred[idx]))

                if pred_trigger - 250 < 0:
                    start = 0
                else:
                    start = pred_trigger - 250
                    flag1 = True

                if pred_trigger + 250 >= 3000:
                    end = 2999
                    flag2 = True
                else:
                    end = pred_trigger + 250
                    flag1 = True
                
                if flag1:
                    pred_trigger = 250
                elif flag2:
                    pred_trigger = 500 - (3000 - pred_trigger)

                plt.figure(figsize=(12, 18))
                plt.subplot(511)
                plt.title('Z')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 0, start:end])

                plt.subplot(512)
                plt.title('N')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 1, start:end])

                plt.subplot(513)
                plt.title('E')
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.legend()
                plt.plot(data['X'][0, 2, start:end])

                plt.subplot(514)
                if res[idx] == 'fp':
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.title('Prediction')
                plt.ylabel('Probability')
                plt.ylim([-0.05, 1.05])
                plt.legend()
                plt.plot(pred[idx][start:end])

                plt.subplot(515)
                plt.plot(data['y'][0, 0, start:end])
                plt.ylim([-0.05, 1.05])
                if res[idx] == 'fp':
                    gt_trigger = np.argmax(data['y'][0, 0])
                    plt.axvline(x=int(pred_trigger), color='r', label='Prediction')
                plt.title('Ground-truth')
                plt.ylabel('Probability')
                plt.xlabel('Time sample (0.01 s per sample)')
                plt.legend()
                if opt.plot_path == 'none':
                    plt.savefig(f"./plot/{opt.model_opt}_{opt.dataset_opt}/Zoomin_{res[idx]}_{idx}.png", dpi=300)
                else:
                    tmp_path = f"./plot/{opt.model_opt}_{opt.dataset_opt}/{opt.plot_path}/"
                    if not os.path.exists(tmp_path):
                        os.makedirs(tmp_path)
                    plt.savefig(f"./plot/{opt.model_opt}_{opt.dataset_opt}/{opt.plot_path}/Zoomin_{res[idx]}_{idx}.png", dpi=300)
                plt.clf()
                plot_cnt += 1
            else:
                idx += 1
                continue

            idx += 1
  
if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    matplotlib.use('Agg')
    opt = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logdir = f"./log/{opt.model_opt}_{opt.dataset_opt}"
    print('Writing log file at -> ', logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if opt.data_type != 'noise':
        log_path = f"{logdir}/{opt.p_timestep}_{opt.lower_snr}_{opt.upper_snr}.log"
        if opt.lower_snr != -1 or opt.upper_snr != -1:
            log_path = f"{logdir}/Noise_{opt.p_timestep}_{opt.lower_snr}_{opt.upper_snr}.log"
        elif opt.lower_epidis != -1 or opt.upper_epidis != -1:
            log_path = f"{logdir}/Noise_{opt.p_timestep}_{opt.lower_epidis}_{opt.upper_epidis}.log"
    else:
        log_path = f"{logdir}/{opt.p_timestep}.log"

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)

    plot_path = f"./plot/{opt.model_opt}_{opt.dataset_opt}" 
    print('Saving png files at -> ', plot_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # load datasets
    print('loading datasets')
    test_generator = set_generators(opt)
    test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # load model
    model = load_model(opt, device)
    ckpt_path = f"./checkpoints/{opt.model_opt}_{opt.dataset_opt}/model.pt"
    if opt.dataset_opt == 'cwbsn_noise':
        ckpt_path = f"./checkpoints/{opt.model_opt}_taiwan/model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)['model']
    model.load_state_dict(checkpoint, strict=False)

    pred, gt = inference(opt, model, test_loader, device)

    tp, fp, tn, fn, diff, abs_diff, res, fn_idx, fp_idx = evaluation(pred, gt, opt.threshold_prob, opt.threshold_trigger, opt.threshold_type)
    
    if opt.plot:
        print('Start plotting...')
        print(f"FP: {len(fp_idx)} to plot, FN: {len(fn_idx)} to plot")
        test_loader = DataLoader(test_generator, batch_size=1, shuffle=False, num_workers=opt.workers)
        plot(opt, test_loader, pred, fn_idx, fp_idx, res)
        plot_zoomin(opt, test_loader, pred, fn_idx, fp_idx, res)
    
    # statisical  
    if opt.data_type == 'other':
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
        fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
        print('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f' %(recall, fpr, precision, fscore))
        print('TP: %d, FP: %d, TN: %d, FN: %d ' %(tp, fp, tn, fn))

        logging.info('======================================================')
        logging.info('SNR: %d-%d' %(opt.lower_snr, opt.upper_snr))
        logging.info('Epicentral distance: %d-%d' %(opt.lower_epidis, opt.upper_epidis))
        logging.info('threshold_type: %s' %(opt.threshold_type))
        logging.info('threshold_prob: %.2f' %(opt.threshold_prob))
        logging.info('threshold_trigger: %d' %(opt.threshold_trigger))
        logging.info('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f' %(recall, fpr, precision, fscore))
        logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
        logging.info('Total=%d' %(tp+fp+tn+fn))
        logging.info('abs_diff=%.4f, diff=%.4f' %(np.mean(abs_diff)/100, np.mean(diff)/100))
        logging.info('trigger_mean=%.4f, trigger_std=%.4f' %(np.mean(diff)/100, np.std(diff)/100))
        # logging.info('MCC=%.4f' %(mcc))
        logging.info('RMSE=%.4f, MAE=%.4f' %(np.sqrt(np.mean(np.array(diff)**2))/100, np.mean(abs_diff)/100))
        logging.info('======================================================')
    else:
        logging.info('======================================================')
        print('Accuracy: %.4f' %(tn/(tn+fp)))
        logging.info('threshold_type: %s' %(opt.threshold_type))
        logging.info('threshold_prob: %.2f' %(opt.threshold_prob))
        logging.info('threshold_trigger: %d' %(opt.threshold_trigger))
        logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
        logging.info('Accuracy: %.4f' %(tn/(tn+fp)))
        logging.info('Total=%d' %(tp+fp+tn+fn))
        logging.info('======================================================')


