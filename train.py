import argparse
import os
import numpy as np
import logging
import requests
import pickle
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from optimizer import *
from utils import *

import seisbench.data as sbd
import seisbench.generate as sbg

def parse_args():
    parser = argparse.ArgumentParser()
    
    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='none')
    parser.add_argument('--load_specific_model', type=str, default='model')
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--patience', type=float, default=7)
    parser.add_argument('--noam', type=bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=1500)
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--config_path', type=str, default='none')

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--instrument', type=str, default='all')
    parser.add_argument('--location', type=int, default=-1)
    parser.add_argument("--filter_instance", type=bool, default=False)

    # data augmentations
    parser.add_argument('--gaussian_noise_prob', type=float, default=0.5)
    parser.add_argument('--channel_dropout_prob', type=float, default=0.3)
    parser.add_argument('--adding_gap_prob', type=float, default=0.2)

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
    parser.add_argument('--max_freq', type=int, default=12)
    parser.add_argument('--rep_KV', type=bool, default=False)
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

def toLine(save_path, train_loss, valid_loss, epoch, n_epochs, isFinish):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    if not isFinish:
        message = save_path + ' -> Epoch ['+ str(epoch) + '/' + str(n_epochs) + '] train_loss: ' + str(train_loss) +', valid_loss: ' +str(valid_loss)
    else:
        message = save_path + ' -> Finish training...'

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

def noam_optimizer(model, lr, warmup_step, device):
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    o = {}
    o["method"] = 'adam'
    o['lr'] = lr
    o['max_grad_norm'] = 0
    o['beta1'] = 0.9
    o['beta2'] = 0.999
    o['decay_method'] = 'noam'
    o['warmup_steps'] = warmup_step
    optimizer = build_optim(o, model, False, device)

    return optimizer

def eqt_init_lr(epoch, optimizer, scheduler):
    lr = 1e-3
    
    change_init_lr = [21, 41, 61, 91]
    if epoch in change_init_lr:
        if epoch > 90:
            lr *= 0.5e-3
        elif epoch > 60:
            lr *= 1e-3
        elif epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
            
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), cooldown=0, min_lr=0.5e-6, patience=3)
        
    return optimizer, scheduler

def split_dataset(opt, return_dataset=False):
    # load datasets
    print('loading datasets')
    cwbsn, tsmip, stead, cwbsn_noise, instance = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
        stead_train, stead_dev, _ = stead.train_dev_test()
        cwbsn_noise_train, cwbsn_noise_dev, _ = cwbsn_noise.train_dev_test()
        instance_train, instance_dev, _ = instance.train_dev_test()

        train = cwbsn_train + tsmip_train + stead_train + cwbsn_noise_train + instance_train
        dev = cwbsn_dev + tsmip_dev + stead_dev + cwbsn_noise_dev + instance_dev
    elif opt.dataset_opt == 'cwb':        
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
        cwbsn_noise_train, cwbsn_noise_dev, _ = cwbsn_noise.train_dev_test()

        train = cwbsn_train + tsmip_train + cwbsn_noise_train
        dev = cwbsn_dev + tsmip_dev + cwbsn_noise_dev
    elif opt.dataset_opt == 'cwbsn':
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()

        train = cwbsn_train
        dev = cwbsn_dev 
    elif opt.dataset_opt == 'tsmip':
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()

        train = tsmip_train
        dev = tsmip_dev 
    elif opt.dataset_opt == 'stead':
        train, dev, _ = stead.train_dev_test()
    elif opt.dataset_opt == 'instance':
        train, dev, _ = instance.train_dev_test()

    print(f'total traces -> train: {len(train)}, dev: {len(dev)}')

    return train, dev

def set_generators(opt, data, Taiwan_aug=False):
    generator = sbg.GenericGenerator(data)

    # set generator with or without augmentations
    augmentations = basic_augmentations(opt)

    generator.add_augmentations(augmentations)

    if opt.aug:
        # data augmentation during training
        # 1) Add gaps (0.2)
        # 2) Channel dropout (0.3)
        # 3) Gaussian noise (0.5)
        # 4) Mask AfterP (0.3)
        # 5) Shift to end (0.2)

        gap_generator = sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [opt.adding_gap_prob, 1-opt.adding_gap_prob])
        dropout_generator = sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [opt.channel_dropout_prob, 1-opt.channel_dropout_prob])
        noise_generator = sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [opt.gaussian_noise_prob, 1-opt.gaussian_noise_prob])

        generator.augmentation(gap_generator)
        generator.augmentation(dropout_generator)
        generator.augmentation(noise_generator)

    return generator

def train(model, optimizer, dataloader, valid_loader, device, cur_epoch, opt, eqt_reg=None):
    model.train()
    train_loss = 0.0
    min_loss = 1000

    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data) in train_loop:        
        if opt.model_opt == 'GRADUATE':
            out = model(data['X'].to(device), stft=data['stft'].float().to(device))
        else:
            out = model(data['X'].to(device))
            
        if opt.model_opt == 'GRADUATE':
            if opt.label_type == 'p' or opt.label_type == 'other':
                loss = loss_fn(opt, pred=out, gt=data['y'], device=device)
            elif opt.label_type == 'all':
                loss = loss_fn(opt, pred=out, gt=(data['y'], data['detections']), device=device)
        elif opt.model_opt == 'eqt':
            loss = loss_fn(opt, out, (data['y'], data['detections']), device, eqt_regularization=(model, eqt_reg))
        else:
            loss = loss_fn(opt, out, data['y'], device)

        loss = loss / opt.gradient_accumulation
        loss.backward()

        if ((idx+1) % opt.gradient_accumulation == 0) or ((idx+1) == len(dataloader)):
            if opt.clip_norm != 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.clip_norm)
            optimizer.step()
            
            model.zero_grad()

            if not opt.noam:
                optimizer.zero_grad()
        
        train_loss = train_loss + loss.detach().cpu().item()*opt.gradient_accumulation
        train_loop.set_description(f"[Train Epoch {cur_epoch+1}/{opt.epochs}]")
        train_loop.set_postfix(loss=loss.detach().cpu().item()*opt.gradient_accumulation)
        
    train_loss = train_loss / (len(dataloader))
    
    return train_loss

def valid(model, dataloader, device, cur_epoch, opt, eqt_reg=None):
    model.eval()
    dev_loss = 0.0

    valid_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in valid_loop:
        with torch.no_grad():               
            if opt.model_opt == 'GRADUATE':
                out = model(data['X'].to(device), stft=data['stft'].float().to(device))
            else:
                out = model(data['X'].to(device))

        if opt.model_opt == 'GRADUATE':
            if opt.label_type == 'p' or opt.label_type == 'other':
                loss = loss_fn(opt, pred=out, gt=data['y'], device=device)
            elif opt.label_type == 'all':
                loss = loss_fn(opt, pred=out, gt=(data['y'], data['detections']), device=device)
            elif opt.model_opt == 'eqt':
                loss = loss_fn(opt, out, (data['y'], data['detections']), device, eqt_regularization=(model, eqt_reg))
            else:
                loss = loss_fn(opt, out, data['y'], device)
        
        dev_loss = dev_loss + loss.detach().cpu().item()
        
        valid_loop.set_description(f"[Valid Epoch {cur_epoch+1}/{opt.epochs}]")
        valid_loop.set_postfix(loss=loss.detach().cpu().item())
        
    valid_loss = dev_loss / (len(dataloader))

    return valid_loss

def save_after_train(output_dir, epoch, model, optimizer, min_loss):
    # save every epoch
    targetPath = os.path.join(output_dir, f"model_epoch{epoch}.pt")

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer,
        'min_loss': min_loss,
        'epoch': epoch
    }, targetPath)

if __name__ == '__main__':
    opt = parse_args()
    
    output_dir = os.path.join('./results', opt.save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    logging.info('start training')
    logging.info('configs: ')
    logging.info(opt)
    logging.info('======================================================')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    os.environ["OMP_NUM_THREADS"] = str(opt.workers)

    logging.info('device: %s'%(device))

    print('loading model...')
    model = load_model(opt, device)
    
    # collect the module's name to regularization, only for Eqt
    if opt.model_opt == 'eqt':
        with open('./eqt/regularization_layer.txt', 'r') as f:
            eqt_reg = f.readlines()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('trainable parameters: %d' %(trainable))
    print('trainable parameters: %d' %(trainable))

    print('loading optimizer & scheduler...')
    if opt.noam:
        optimizer = noam_optimizer(model, opt.lr, opt.warmup_step, device)
    elif opt.model_opt == 'eqt':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), cooldown=0, min_lr=0.5e-6, patience=opt.patience-7)
    else:
        optimizer = optim.Adam(model.parameters(), opt.lr)

    train_set, dev_set = split_dataset(opt)

    train_generator = set_generators(opt, train_set)
    dev_generator = set_generators(opt, dev_set)

    # create dataloaders
    print('creating dataloaders')
    train_loader = DataLoader(train_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    logging.info('train: %d, dev: %d' %(len(train_loader)*opt.batch_size, len(dev_loader)*opt.batch_size))

    # load checkpoint
    if opt.resume_training:
        logging.info('Resume training...')

        checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_last.pt'), map_location=device)
        print('resume training, load checkpoint_last.pt...')

        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        min_loss = checkpoint['min_loss']
        print(f'start from epoch {init_epoch}, min_loss: {min_loss}')
        logging.info('resume training at epoch: %d' %(init_epoch))

    elif opt.pretrained_path != 'none':
        logging.info('Loading pretrained checkpoint... %s' %(opt.pretrained_path))

        print('Loading pretrained checkpoint...', opt.pretrained_path)
        checkpoint = torch.load(f"./results/{opt.pretrained_path}/{opt.load_specific_model}.pt", map_location=device)

        model.load_state_dict(checkpoint['model'])
        init_epoch = 0
        min_loss = 100000
    else:
        init_epoch = 0
        min_loss = 100000

    early_stop_cnt = 0
    prev_loss, valid_loss = 1000, 1000
    reload_dataset = False
    for epoch in range(init_epoch, opt.epochs):
        # Load the dataset again
        if early_stop_cnt >= 4 and opt.model_opt == 'GRADUATE' and not reload_dataset: 
            reload_dataset = True
            early_stop_cnt -= 1
            train_set, dev_set = split_dataset(opt)
    
            train_generator = set_generators(opt, train_set)
            dev_generator = set_generators(opt, dev_set)

            # create dataloaders
            print('creating dataloaders again')
            train_loader = DataLoader(train_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
            dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

        if opt.model_opt == 'eqt':
            optimizer, scheduler = eqt_init_lr(epoch, optimizer, scheduler)

            train_loss = train(model, optimizer, train_loader, dev_loader, device, epoch, opt, output_dir, eqt_reg=eqt_reg)
            save_after_train(output_dir, epoch, model, optimizer, min_loss)
            valid_loss = valid(model, dev_loader, device, epoch, opt, eqt_reg=eqt_reg)
        
            scheduler.step(valid_loss)
        else:
            train_loss = train(model, optimizer, train_loader, dev_loader, device, epoch, opt, output_dir)
            save_after_train(output_dir, epoch, model, optimizer, min_loss)
            valid_loss = valid(model, dev_loader, device, epoch, opt)

        print('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        print('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        logging.info('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))

        if opt.noam:
            print('Learning rate: %.10f' %(optimizer.learning_rate))
            logging.info('Learning rate: %.10f' %(optimizer.learning_rate))
        else:
            print('Learning rate: %.10f' %(optimizer.param_groups[0]['lr']))
            logging.info('Learning rate: %.10f' %(optimizer.param_groups[0]['lr']))
        logging.info('======================================================')

        # Line notify
        toLine(opt.save_path, train_loss, valid_loss, epoch, opt.epochs, False)

        # Early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss

            # Saving model
            targetPath = os.path.join(output_dir, 'model.pt')

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer,
                'min_loss': min_loss,
                'epoch': epoch
            }, targetPath)

            early_stop_cnt = 0
            print(f"Validation loss improved from {min_loss} to {valid_loss}...")
        else:
            early_stop_cnt += 1
            print('Validation loss did not improve, early stopping cnt: ', early_stop_cnt)

        # Saving model
        targetPath = os.path.join(output_dir, 'checkpoint_last.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer,
            'min_loss': min_loss,
            'epoch': epoch
        }, targetPath)

        if early_stop_cnt == opt.patience:
            logging.info('early stopping...')

            break

    print('Finish training...')
    toLine(opt.save_path, train_loss, valid_loss, epoch, opt.epochs, True)

    # save the config into json 
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(opt, f, indent=2)













