import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

import numpy as np
import torch
import torch.nn.functional as F

import obspy

from model import *

def load_dataset(opt):
    cwbsn, tsmip, stead, cwbsn_noise, instance = 0, 0, 0, 0, 0
    
    if opt.dataset_opt == 'instance' or opt.dataset_opt == 'all':
        print('loading INSTANCE')
        kwargs={'download_kwargs': {'basepath': '/home/weiwei/disk4/seismic_datasets/'}}
        instance = sbd.InstanceCountsCombined(**kwargs)

        instance = apply_filter(instance, isINSTANCE=True, filter_instance=True)

    snr = [opt.upper_snr, opt.lower_snr]
    epidis = [opt.upper_epidis, opt.lower_epidis]
    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)

        # stead = apply_filter(stead, snr_threshold=opt.snr_threshold, s_wave=opt.s_wave, isStead=True, magnitude=True)
        if opt.data_type == 'noise':
            stead = apply_filter(stead, snr_threshold=snr, isStead=True, isNoise=True)
        else:
            stead = apply_filter(stead, snr_threshold=snr, isStead=True)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan' or opt.dataset_opt == 'EEW':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method='full', **kwargs)
        cwbsn = apply_filter(cwbsn, snr_threshold=snr, isCWBSN=True, level=4, instrument='all', location=-1, epidis=epidis)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan' or opt.dataset_opt == 'EEW':
        # TSMIP
        print('loading TSMIP') 
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method='full', sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        tsmip = apply_filter(tsmip, snr_threshold=snr, instrument='all', location=-1, epidis=epidis)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'EEW' or opt.dataset_opt == 'cwbsn_noise':
        # CWBSN noise
        print('loading CWBSN noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/disk4/weiwei/seismic_datasets/CWB_noise/'}}
        cwbsn_noise = sbd.CWBSN_noise(**kwargs)
        
        cwbsn_noise = apply_filter(cwbsn_noise, instrument='all', isNoise=True, location=-1, epidis=epidis)

        print('traces: ', len(cwbsn_noise))

    return cwbsn, tsmip, stead, cwbsn_noise, instance

def apply_filter(data, snr_threshold=-1, isCWBSN=False, level=-1, s_wave=False, isStead=False, isNoise=False, instrument='all', noise_sample=-1, magnitude=False, location=-1, isINSTANCE=False, filter_instance=False, epidis=-1):
    # Apply filter on seisbench.data class

    print('original traces: ', len(data))
    
    # 只選波型完整的 trace
    if not isStead and not isNoise and not isINSTANCE:
        if isCWBSN:
            complete_mask = data.metadata['trace_completeness'] == 4
        else:
            complete_mask = data.metadata['trace_completeness'] == 1

        # 只選包含一個事件的 trace
        single_mask = data.metadata['trace_event_number'] == 1

        # making final mask
        mask = np.logical_and(single_mask, complete_mask)
        data.filter(mask)

    # 也需要 s_arrival time, 且 p, s 波距離不能太遠
    if not isNoise and not isStead and not isINSTANCE:
        upper, lower = snr_threshold
        if upper == -1 and lower == -1:
            pass
        elif upper == -1:
            mask = data.metadata['trace_Z_snr_db'] >= lower
            data.filter(mask)
        elif lower == -1:
            mask = data.metadata['trace_Z_snr_db'] <= upper
            data.filter(mask)
        else:
            mask = np.logical_and(data.metadata['trace_Z_snr_db'] >= lower, data.metadata['trace_Z_snr_db'] <= upper)
            data.filter(mask)

        upper, lower = epidis
        if upper == -1 and lower == -1:
            pass
        elif upper == -1:
            mask = data.metadata['path_ep_distance_km'] >= lower
            data.filter(mask)
        elif lower == -1:
            mask = data.metadata['path_ep_distance_km'] <= upper
            data.filter(mask)
        else:
            mask = np.logical_and(data.metadata['path_ep_distance_km'] >= lower, data.metadata['path_ep_distance_km'] <= upper)
            data.filter(mask)

    if isStead and isNoise:
        noise_mask = data.metadata['trace_category'] == 'noise'
        data.filter(noise_mask)

    if isINSTANCE and filter_instance:
        p_weight_mask = data.metadata['path_weight_phase_location_P'] >= 50
        eqt_mask = np.logical_and(data.metadata['trace_EQT_number_detections'] == 1, data.metadata['trace_EQT_P_number'] == 1)
        instance_mask = np.logical_and(p_weight_mask, eqt_mask)
        data.filter(instance_mask)

    print('filtered traces: ', len(data))

    return data

def basic_augmentations(opt, phase_dict, ptime=None, test=False, EEW=False):
    # basic augmentations:
    #   1) Windowed around p-phase pick
    #   2) Random cut window, wavelen=3000
    #   3) Filter 
    #   4) Normalize: demean, zscore,
    #   5) Change dtype to float32
    #   6) Probabilistic: gaussian function
    
    if opt.seg_proj_type == 'none':
        seg_null = True
    else:
        seg_null = False

    if opt.dataset_opt == 'instance':
        p_phases = 'trace_P_arrival_sample'
        s_phases = 'trace_S_arrival_sample'
    else:
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'

    if opt.model_opt == 'eqt':
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
        elif EEW:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=330),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
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
    elif opt.model_opt == 'GRADUATE' or opt.model_opt == 'GRADUATE_noTF':
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
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
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
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
        elif EEW:
            if opt.wavelength == 3000:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=100, high=opt.wavelength+300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                        sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                        sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                        sbg.STFT(max_freq=opt.max_freq),
                        sbg.CharStaLta(),
                        sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
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
                        sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
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
    elif opt.model_opt == 'GRADUATE':
        model = GRADUATE(conformer_class=8, d_ffn=128, nhead=4, d_model=12, enc_layers=2, 
                    encoder_type='conformer', dec_layers=1, norm_type='123', l=1, cross_attn_type=4, 
                    decoder_type='crossattn', rep_KV=True, seg_proj_type='none',
                    label_type='all', recover_type='conv', rep_query=False, input_type='normal', 
                    stft_loss=False, patch_crossattn=False, max_freq=opt.max_freq, wavelength=opt.wavelength, stft_recovertype='conv',
                    stft_residual=False, dualDomain_type='concat', ablation='none')
    elif opt.model_opt == 'GRADUATE_noTF':
        model = GRADUATE(conformer_class=8, d_ffn=128, nhead=4, d_model=12, enc_layers=2, 
                    encoder_type='conformer', dec_layers=1, norm_type='123', l=1, cross_attn_type=4, 
                    decoder_type='crossattn', rep_KV=True, seg_proj_type='none',
                    label_type='all', recover_type='conv', rep_query=False, input_type='normal', 
                    stft_loss=False, patch_crossattn=False, max_freq=opt.max_freq, wavelength=opt.wavelength, stft_recovertype='conv',
                    stft_residual=False, dualDomain_type='concat', ablation='time-frequency')

    return model.to(device)

