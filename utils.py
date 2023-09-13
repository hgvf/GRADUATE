import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

import numpy as np
import sys
import torch
import torch.nn.functional as F


from model import *

def load_dataset(opt):
    cwbsn, tsmip, stead, cwbsn_noise, instance = 0, 0, 0, 0, 0
    
    if opt.dataset_opt == 'instance' or opt.dataset_opt == 'all':
        print('loading INSTANCE')
        kwargs={'download_kwargs': {'basepath': '/home/weiwei/disk4/seismic_datasets/'}}
        instance = sbd.InstanceCountsCombined(**kwargs)

        instance = apply_filter(instance, isINSTANCE=True, filter_instance=opt.filter_instance)

    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)

        stead = apply_filter(stead, snr_threshold=opt.snr_threshold, isStead=True)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method=opt.loading_method, **kwargs)
        cwbsn = apply_filter(cwbsn, isCWBSN=True, level=opt.level, instrument=opt.instrument, location=opt.location)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all':
        # TSMIP
        print('loading TSMIP') 
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method=opt.loading_method, sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        tsmip = apply_filter(tsmip, instrument=opt.instrument, location=opt.location)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'cwb' or opt.dataset_opt == 'all':
        # CWBSN noise
        print('loading CWBSN noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/disk4/weiwei/seismic_datasets/CWB_noise/'}}
        cwbsn_noise = sbd.CWBSN_noise(**kwargs)
        
        cwbsn_noise = apply_filter(cwbsn_noise, instrument=opt.instrument, isNoise=True, location=opt.location)

        print('traces: ', len(cwbsn_noise))

    return cwbsn, tsmip, stead, cwbsn_noise, instance

def apply_filter(data, isCWBSN=False, level=-1, isStead=False, isNoise=False, instrument='all', location=-1, isINSTANCE=False, filter_instance=False):
    # Apply filter on seisbench.data class

    print('original traces: ', len(data))
    
    # 只選波型完整的 trace
    if not isStead and not isNoise and not isINSTANCE:
        if isCWBSN:
            if level != -1:
                complete_mask = data.metadata['trace_completeness'] == level
            else:
                complete_mask = np.logical_or(data.metadata['trace_completeness'] == 3, data.metadata['trace_completeness'] == 4)
        else:
            complete_mask = data.metadata['trace_completeness'] == 1

        # 只選包含一個事件的 trace
        single_mask = data.metadata['trace_event_number'] == 1

        # making final mask
        mask = np.logical_and(single_mask, complete_mask)
        data.filter(mask)

    if location != -1:
        location_mask = np.logical_or(data.metadata['station_location_code'] == location, data.metadata['station_location_code'] == str(location))
        data.filter(location_mask)

    # 篩選儀器
    if instrument != 'all':
        instrument_mask = data.metadata["trace_channel"] == instrument
        data.filter(instrument_mask)

    if isINSTANCE and filter_instance:
        p_weight_mask = data.metadata['path_weight_phase_location_P'] >= 50
        eqt_mask = np.logical_and(data.metadata['trace_EQT_number_detections'] == 1, data.metadata['trace_EQT_P_number'] == 1)
        instance_mask = np.logical_and(p_weight_mask, eqt_mask)
        data.filter(instance_mask)

    print('filtered traces: ', len(data))

    return data

def basic_augmentations(opt, ptime=None, test=False):
    # basic augmentations:
    #   1) Windowed around p-phase pick
    #   2) Random cut window, wavelen=3000
    #   3) Filter 
    #   4) Normalize: demean, zscore,
    #   5) Change dtype to float32
    #   6) Probabilistic: gaussian function
    
    if opt.dataset_opt == 'instance':
        p_phases = 'trace_P_arrival_sample'
        s_phases = 'trace_S_arrival_sample'
    else:
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'

    if opt.model_opt == 'phaseNet':
        phase_dict = [p_phases, s_phases]
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3001-ptime, windowlen=3001, strategy='pad'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std", keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3001, strategy="pad"),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
    elif opt.model_opt == 'eqt':
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    # sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    # sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    # sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    elif opt.model_opt == 'GRADUATE':
        if opt.label_type == 'all':
            phase_dict = [p_phases, s_phases]
        else:
            phase_dict = [p_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                        sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                        sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                        sbg.STFT(max_freq=opt.max_freq),
                        sbg.CharStaLta(),
                        sbg.ChangeDtype(np.float32),
                        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
            else:
                augmentations = [
                        sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                        sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                        # sbg.VtoA(),
                        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                        sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                        sbg.STFT(max_freq=opt.max_freq),
                        sbg.CharStaLta(),
                        sbg.ChangeDtype(np.float32),
                        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
        
        if opt.label_type == 'all':
            augmentations.append(sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4))
    
    return augmentations

def load_model(opt, device):
    assert opt.model_opt != None, "Choose one of the model in seisbench."

    if opt.model_opt == 'eqt':
        model = sbm.EQTransformer(in_samples=opt.wavelength)
    elif opt.model_opt == 'phaseNet':
        model = sbm.PhaseNet(in_channels=3, classes=3, phases='NPS')
    elif opt.model_opt == 'GRADUATE':
        model = GRADUATE(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, nhead=opt.nhead, d_model=opt.d_model, enc_layers=opt.enc_layers, 
                    dec_layers=opt.dec_layers, rep_KV=opt.rep_KV, label_type=opt.label_type, recover_type=opt.recover_type, 
                    max_freq=opt.max_freq, wavelength=opt.wavelength, stft_recovertype=opt.stft_recovertype,
                    dualDomain_type=opt.dualDomain_type, ablation=opt.ablation)
    
    return model.to(device)

def loss_fn(opt, pred, gt, device, cur_epoch=None, eqt_regularization=None):
    if opt.model_opt == 'eqt':
        picking_gt, detection_gt = gt
        loss_weight = [0.4, 0.55, 0.05]       # P, S, detection
        reduction = 'sum'

        # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
        # detection
        # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
        loss = 0.0
        for i in range(3):
            if i == 0 or i == 1:
                nonzero_idx = (picking_gt[:, i] != 0)
                weights = torch.ones(picking_gt[:, i].shape)*0.11
                weights[nonzero_idx] = 0.89

                loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[i+1].to(device), target=picking_gt[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
            else:
                nonzero_idx = (detection_gt[:, 0] != 0)
                weights = torch.ones(detection_gt[:, 0].shape)*0.11
                weights[nonzero_idx] = 0.89

                loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[0].to(device), target=detection_gt[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)

        reg = 0.0
        l1 = 1e-4
        model, eqt_reg = eqt_regularization
        for name, param in model.state_dict().items():
            if name+'\n' in eqt_reg:
                reg += torch.norm(param.data, 1)
        
        loss = loss + reg * l1
    elif opt.model_opt == 'phaseNet':
        # vector cross entropy loss
        h = gt.to(device) * torch.log(pred.to(device) + 1e-5)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis

        loss = -h
    elif opt.model_opt == 'GRADUATE':
        pred_picking = pred
        reduction = 'mean'

        # ======================== Picking ======================= #
        if opt.label_type == 'p':
            gt_picking = gt

            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            picking_loss = F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, 0].to(device), target=gt_picking[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'other':
            gt_picking = gt
 
            picking_loss = 0.0
            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            for i in range(2):
                picking_loss += F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'all':
            pred_detection, pred_p, pred_s = pred_picking
            
            pred = [pred_p, pred_s]
            gt_picking, gt_detection = gt
            
            loss_weight = [0.6, 0.35, 0.05]       # P, S, detection
        
            # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
            # detection
            # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
            picking_loss = 0.0
            for i in range(3):
                if i == 0 or i == 1:
                    weights = torch.add(torch.mul(gt_picking[:, i], opt.loss_weight), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
                else:
                    weights = torch.add(torch.mul(gt_detection[:, 0], opt.loss_weight), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_detection.squeeze().to(device), target=gt_detection[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)
        loss = picking_loss
    
    return loss

