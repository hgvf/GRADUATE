import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import pickle
import json
import time
import bisect
import requests
from tqdm import tqdm
from argparse import Namespace

import sys
sys.path.append('./eqt')
from load_eqt import *

from calc import calc_intensity
from snr import snr_p
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--config_path', type=str, default='none')
    parser.add_argument('--load_specific_model', type=str, default='None')
    parser.add_argument("--threshold_type", type=str, default='all')
    parser.add_argument('--threshold_prob_start', type=float, default=0.15)
    parser.add_argument('--threshold_prob_end', type=float, default=0.9)
    parser.add_argument('--threshold_trigger_start', type=int, default=5)
    parser.add_argument('--threshold_trigger_end', type=int, default=45)
    parser.add_argument('--sample_tolerant', type=int, default=50)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--p_timestep', type=int, default=750)
    parser.add_argument('--allTest', type=bool, default=False)

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--instrument', type=str, default='all')
    parser.add_argument('--location', type=int, default=-1)
    parser.add_argument("--filter_instance", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--batch_size", type=int, default=100)

    # seisbench options
    parser.add_argument('--model_opt', type=str, default='none')
    parser.add_argument('--loss_weight', type=float, default=10)
    parser.add_argument('--dataset_opt', type=str, default='cwb')
    parser.add_argument('--loading_method', type=str, default='full')
    
    # custom hyperparameters
    parser.add_argument('--conformer_class', type=int, default=8)
    parser.add_argument('--d_ffn', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_type', type=str, default='all')

    # GRADUATE model
    parser.add_argument('--rep_KV', type=bool, default=False)
    parser.add_argument('--max_freq', type=int, default=12)
    parser.add_argument('--recover_type', type=str, default='conv')
    parser.add_argument('--wavelength', type=int, default=3000)
    parser.add_argument('--stft_recovertype', type=str, default='conv')
    parser.add_argument('--dualDomain_type', type=str, default='concat')
    parser.add_argument('--ablation', type=str, default='none')

    opt = parser.parse_args()

    # load the config 
    if opt.config_path != 'none':
        f = open(opt.config_path, 'r')
        config = json.load(f)

        opt = vars(opt)
        opt.update(config)

        opt = Namespace(**opt)

    return opt

def toLine(save_path, precision, recall, fscore, mean, variance):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    message = save_path + ' -> precision: ' + str(precision) + ', recall: ' + str(recall) + ', fscore: ' + str(fscore) + ', mean: ' + str(mean) + ', variance: ' + str(variance)
    
    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
        }
        payload = {
            'message': message
        }
        response = requests.request(
            "POST",
            url,
            headers=headers,
            data=payload
        )
        if response.status_code == 200:
            print(f"Success -> {response.text}")
    except Exception as e:
        print(e)

def evaluation(pred, gt, snr_idx, snr_max_idx, intensity_idx, intensity_max_idx, threshold_prob, threshold_trigger, sample_tolerant, mode):
    tp, fp, tn, fn = 0, 0, 0, 0 
    diff = []
    abs_diff = []
    res = []

    # snr stat
    snr_stat = {}
    for i in range(snr_max_idx):
        snr_stat[str(i)] = []
        
    # intensity stat
    intensity_stat = {}
    for i in range(intensity_max_idx):
        intensity_stat[str(i)] = []

    # stat case-by-case
    case_stat = {}
    case_stat['snr'], case_stat['intensity'], case_stat['res'] = [], [], []

    for i in range(len(pred)):
        pred_isTrigger = False
        gt_isTrigger = False
        gt_trigger = 0

        snr_cur = snr_idx[i]
        intensity_cur = intensity_idx[i]
        
        if not np.all(gt[i] == 0):
            gt_isTrigger = True            
            gt_trigger = np.argmax(gt[i])
            if gt[i][gt_trigger] < 0.3:
                gt_isTrigger = False
                gt_trigger = 0

        if mode == 'single':
            a = np.where(pred[i] >= threshold_prob, 1, 0)

            if np.any(a):
                c = np.where(a==1)
                pred_isTrigger = True
                pred_trigger = c[0][0]
            else:
                pred_trigger = 0
        
        elif mode == 'avg':
            a = pd.Series(pred[i])  
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                # pred_trigger = tri[0][0]-threshold_trigger+1
                pred_trigger = tri[0][0]
                pred_isTrigger = True
                
        elif mode == 'continue':
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

        elif mode == 'max':
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
            snr_stat[str(snr_cur)].append('tp')
            intensity_stat[str(intensity_cur)].append('tp')
        elif (pred_isTrigger):
            fp += 1
            res.append('fp')
            snr_stat[str(snr_cur)].append('fp')
            intensity_stat[str(intensity_cur)].append('fp')

        # case negative
        if (not pred_isTrigger) and (gt_isTrigger):
            fn += 1
            res.append('fn')
            snr_stat[str(snr_cur)].append('fn')
            intensity_stat[str(intensity_cur)].append('fn')
        elif (not pred_isTrigger) and (not gt_isTrigger):
            tn += 1
            res.append('tn')
            snr_stat[str(snr_cur)].append('tn')
            intensity_stat[str(intensity_cur)].append('tn')

        if gt_isTrigger and pred_isTrigger:
            diff.append(pred_trigger-gt_trigger)
            abs_diff.append(abs(pred_trigger-gt_trigger))

        case_stat['snr'].append(str(snr_cur))
        case_stat['intensity'].append(str(intensity_cur))
        case_stat['res'].append(res[i])

    return tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat

def set_generators(opt, ptime=None):
    cwbsn, tsmip, stead, cwbsn_noise, instance = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        stead_dev, stead_test = stead.dev(), stead.test()
        cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()
        instance_train, instance_dev, _ = instance.train_dev_test()

        train = cwbsn_train + tsmip_train + stead_train + cwbsn_noise_train + instance_train
        dev = cwbsn_dev + tsmip_dev + stead_dev + cwbsn_noise_dev + instance_dev
    elif opt.dataset_opt == 'cwbsn':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        stead_dev, stead_test = stead.dev(), stead.test()

        dev = cwbsn_dev + stead_dev
        test = cwbsn_test + stead_test
    elif opt.dataset_opt == 'tsmip':
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()

        dev = tsmip_dev
        test = tsmip_test
    elif opt.dataset_opt == 'stead':
        _, dev, test = stead.train_dev_test()
    elif opt.dataset_opt == 'instance':
        _, dev, test = instance.train_dev_test()
    elif opt.dataset_opt == 'cwb':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()

        dev = cwbsn_dev + tsmip_dev + cwbsn_noise_dev
        test = cwbsn_test + tsmip_test + cwbsn_noise_test

    print(f'total traces -> dev: {len(dev)}, test: {len(test)}')

    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    # set generator with or without augmentations
    if not opt.allTest:
        ptime = opt.p_timestep  
    augmentations = basic_augmentations(opt, test=True, ptime=ptime)
    
    dev_generator.add_augmentations(augmentations)
    test_generator.add_augmentations(augmentations)

    return dev_generator, test_generator

def calc_snr(data):
    snr_batch = []
    gt = data['y'][:, 0]
 
    for i in range(gt.shape[0]):
        # if not noise waveform, calculate log(SNR)
        tri = torch.where(gt[i] == 1)[0]
        if len(tri) != 0:
            snr_tmp = snr_p(data['ori_X'][i, 0, :].cpu().numpy(), gt[i].cpu().numpy())

            if snr_tmp is None:
                snr_batch.append(-1.0)
            else:
                tmp = 10*np.log10(snr_tmp)
                if tmp < -1.0:
                    tmp = -1.0
                snr_batch.append(tmp)
                
        else:
            snr_batch.append(-9999)

    return snr_batch

def calc_inten(data):
    intensity_batch = []
    gt = data['y'][:, 0]

    for i in range(gt.shape[0]):
        tri = torch.where(gt[i] == 1)[0]
        if len(tri) == 0:
            intensity_batch.append(-1)
        else:
            intensity_tmp = calc_intensity(data['ori_X'][i, 0].numpy(), data['ori_X'][i, 1].numpy(), data['ori_X'][i, 2].numpy(), 'Acceleration', 100)

            intensity_batch.append(intensity_tmp)

    return intensity_batch

def convert_snr_to_level(snr_level, snr_total):
    res = []
    for i in snr_total:
        idx = bisect.bisect_right(snr_level, i)-1
        
        if idx < 0:
            idx = 0
            
        res.append(idx)

    return res

def convert_intensity_to_level(intensity_level, intensity_total):
    res = []
    for i in intensity_total:
        idx = intensity_level.index(i)

        res.append(idx)

    return res

def inference(opt, model, test_loader, device):
    # 先把整個 test set 的預測結果都跑完一遍

    pred = []
    gt = []
    snr_total = []
    intensity_total = []

    model.eval()
    with tqdm(test_loader) as epoch:
        idx = 0
        for data in epoch:  
            idx += 1        

            # calcuate SNR & intensity
            snr_total += calc_snr(data)
            intensity_total += calc_inten(data)

            with torch.no_grad():
                if opt.model_opt == 'GRADUATE':
                    out = model(data['X'].to(device), stft=data['stft'].float().to(device))
                else:
                    out = model(data['X'].to(device))

                if opt.model_opt == 'eqt':
                    out = out[1].detach().squeeze().cpu().numpy()
                elif opt.model_opt == 'phaseNet':
                    out = out[:, 0].detach().squeeze().cpu().numpy()
                else:
                    if opt.label_type == 'p':
                        out = out.detach().squeeze().cpu().numpy()
                    elif opt.label_type == 'other':
                        out = out[:, :, 0].detach().squeeze().cpu().numpy()                
                    elif opt.label_type == 'all':
                        out = out[1].squeeze().detach().cpu().numpy()

                target = data['y'][:, 0].squeeze().numpy()
                        
                if type(out) == list:
                    pass
                elif out.ndim == 2:
                    pred += [out[i] for i in range(out.shape[0])]
                    gt += [target[i] for i in range(target.shape[0])]
                else:
                    pred += [out]
                    gt += [target]

    return pred, gt, snr_total, intensity_total

def score(pred, gt, snr_total, intensity_total, mode, opt, threshold_prob, threshold_trigger, isTest=False):
    # 依照 snr 不同分別計算數據，先將原本的 snr level 轉換成對應 index
    # snr_level = list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    snr_level = [-9999] + list(np.arange(-1.0, 0.0, 0.5)) + list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    intensity_level = [-1, 0, 1, 2, 3, 4, 5, 5.5, 6, 6.5, 7]

    snr_idx = convert_snr_to_level(snr_level, snr_total)
    intensity_idx = convert_intensity_to_level(intensity_level, intensity_total)
        
    tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat = evaluation(pred, gt, snr_idx, len(snr_level), intensity_idx, len(intensity_level), threshold_prob, threshold_trigger, opt.sample_tolerant, mode)
    
    # statisical  
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
    # mcc = (tp*tn-fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    logging.info('======================================================')
    logging.info('threshold_prob: %.2f' %(threshold_prob))
    logging.info('threshold_trigger: %d' %(threshold_trigger))
    logging.info('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f' %(recall, fpr, precision, fscore))
    logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
    logging.info('abs_diff=%.4f, diff=%.4f' %(np.mean(abs_diff)/100, np.mean(diff)/100))
    logging.info('trigger_mean=%.4f, trigger_std=%.4f' %(np.mean(diff)/100, np.std(diff)/100))
    # logging.info('MCC=%.4f' %(mcc))
    logging.info('RMSE=%.4f, MAE=%.4f' %(np.sqrt(np.mean(np.array(diff)**2))/100, np.mean(abs_diff)/100))

    if isTest:
        toLine(opt.save_path, precision, recall, fscore, np.mean(diff)/100, np.std(diff)/100)

    return fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    model_dir = output_dir
    if opt.level == -1:
        level = 'all'
    else:
        level = str(opt.level)

    if not opt.allTest:
        output_dir = os.path.join(output_dir, level)
    else:
        output_dir = os.path.join(output_dir, f"allTest_{opt.dataset_opt}")

    subpath = 'threshold'
    if opt.level != -1:
        subpath = subpath + '_' + str(opt.level)
    if opt.p_timestep != 750:
        subpath = subpath + '_' + str(opt.p_timestep)
    if opt.allTest:
        subpath = subpath + '_allCase_testing_' + str(opt.level)
    if opt.load_specific_model != 'None':
        subpath = subpath + '_' + opt.load_specific_model
    if opt.instrument != 'all':
        subpath = subpath + '_' + opt.instrument
        output_dir = f"{output_dir}_{opt.instrument}"
    if opt.location != -1:
        subpath = subpath + '_' + str(opt.location)
        output_dir = f"{output_dir}_{opt.location}"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subpath = subpath + '.log'
    print('logpath: ', subpath)
    log_path = os.path.join(output_dir, subpath)

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    print(opt.save_path)

    # 設定 device (opt.device = 'cpu' or 'cuda:X')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    if not opt.allTest:
        print('loading datasets')
        dev_generator, test_generator = set_generators(opt)
        dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        
        logging.info('dev: %d, test: %d' %(len(dev_loader)*opt.batch_size, len(test_loader)*opt.batch_size))
    
    # load model
    model = load_model(opt, device)

    if opt.load_specific_model != 'None':
        print('loading ', opt.load_specific_model)
        model_path = os.path.join(model_dir, opt.load_specific_model+'.pt')
    else:
        print('loading last checkpoint')
        model_path = os.path.join(model_dir, 'checkpoint_last.pt')

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # start finding
    max_fscore = 0.0
    cnt = 0
    front = opt.sample_tolerant
    back = opt.sample_tolerant

    if opt.threshold_type == 'all':
        mode = ['max', 'single', 'continue', 'avg']  # avg, continue
        # mode = ['single', 'continue', 'avg']
    elif opt.threshold_type == 'avg':
        mode = ['avg']
    elif opt.threshold_type == 'continue':
        mode = ['continue']
    elif opt.threshold_type == 'single':
        mode = ['single']
    elif opt.threshold_type == 'max':
        mode = ['max']

    if not opt.do_test and not opt.allTest:
        # find the best criteria
        print('finding best criteria...')
        pred, gt, snr_total, intensity_total = inference(opt, model, dev_loader, device)

        best_fscore = 0.0
        best_mode = ""
        best_prob = 0.0
        best_trigger = 0
        for m in mode:
            logging.info('======================================================')
            logging.info('Mode: %s' %(m))

            for prob in tqdm(np.arange(opt.threshold_prob_start, opt.threshold_prob_end, 0.05)):  # (0.45, 0.85)
                max_fscore = 0.0
                cnt = 0

                for trigger in np.arange(opt.threshold_trigger_start, opt.threshold_trigger_end, 5): # (10, 55)
                    fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, m, opt, prob, trigger)
                    print('prob: %.2f, trigger: %d, fscore: %.4f' %(prob, trigger, fscore))

                    if fscore > max_fscore:
                        max_fscore = fscore
                        cnt = 0
                    else:
                        cnt += 1

                    if cnt == 1 or fscore == 0.0:
                        break

                    if fscore > best_fscore:
                        with open(os.path.join(output_dir, 'abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                            pickle.dump(abs_diff, f)

                        with open(os.path.join(output_dir, 'diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                            pickle.dump(diff, f)

                        with open(os.path.join(output_dir, 'snr_stat_'+str(opt.level)+'.json'), 'w') as f:
                            json.dump(snr_stat, f)

                        with open(os.path.join(output_dir, 'intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
                            json.dump(intensity_stat, f)

                        with open(os.path.join(output_dir, 'case_stat_'+str(opt.level)+'.json'), 'w') as f:
                            json.dump(case_stat, f)
                        
                        best_fscore = fscore
                        best_mode = m
                        best_prob = prob
                        best_trigger = trigger

                    if m == 'single' or m == 'max':
                        break
            
        logging.info('======================================================')
        logging.info("Best: ")
        logging.info(f"mode: {best_mode}, prob: {best_prob}, trigger: {best_trigger}, fscore: {best_fscore}")
        logging.info('======================================================')
        print(f'Best criteria -> type: {best_mode}, prob: {best_prob}, trigger: {best_trigger}, fscore: {best_fscore}')

    if opt.do_test:
        best_mode = opt.threshold_type
        best_prob = opt.threshold_prob_start
        best_trigger = opt.threshold_trigger_start

        logging.info('Inference on testing set')
        pred, gt, snr_total, intensity_total = inference(opt, model, test_loader, device)

        fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, best_mode, opt, best_prob, best_trigger, True)
        print('fscore: %.4f' %(fscore))
        
        with open(os.path.join(output_dir, 'test_abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(abs_diff, f)

        with open(os.path.join(output_dir, 'test_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(diff, f)

        with open(os.path.join(output_dir, 'test_snr_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(snr_stat, f)

        with open(os.path.join(output_dir, 'test_intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(intensity_stat, f)

        with open(os.path.join(output_dir, 'test_case_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(case_stat, f)

    # 將 p arrival 固定在多個不同時間點，分別得到實驗結果
    if opt.allTest:
        logging.info('configs: ')
        logging.info(opt)
        logging.info('dataset: %s' %(opt.dataset_opt))
        
        print('Start testing...')
        if opt.wavelength == 3000:
            ptime_list = [750, 1500, 2000, 2500, 2750]
        elif opt.wavelength == 1000:
            ptime_list = [250, 500, 750]
        elif opt.wavelength == 2000:
            ptime_list = [500, 1000, 1500, 1750]
        elif opt.wavelength == 500:
            ptime_list = [150, 300, 450]
        
        best_mode = opt.threshold_type
        best_prob = opt.threshold_prob_start
        best_trigger = opt.threshold_trigger_start

        for ptime in ptime_list:
            print('='*50)
            print(f"ptime: {ptime}")
            tmp_output_dir = os.path.join(output_dir, subpath[:-4])
            new_output_dir = os.path.join(tmp_output_dir, str(ptime))
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)

            _, test_generator = set_generators(opt, ptime=ptime)
            test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

            # start predicting on test set
            logging.info('======================================================')
            logging.info('Inference on testing set, ptime: %d' %(ptime))
            pred, gt, snr_total, intensity_total = inference(opt, model, test_loader, device)

            fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, best_mode, opt, best_prob, best_trigger, True)
            print(f"ptime: {ptime}, fscore: {fscore}")
            logging.info('======================================================')

            with open(os.path.join(new_output_dir, 'test_abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                pickle.dump(abs_diff, f)

            with open(os.path.join(new_output_dir, 'test_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                pickle.dump(diff, f)

            with open(os.path.join(new_output_dir, 'test_snr_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(snr_stat, f)

            with open(os.path.join(new_output_dir, 'test_intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(intensity_stat, f)

            with open(os.path.join(new_output_dir, 'test_case_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(case_stat, f)
