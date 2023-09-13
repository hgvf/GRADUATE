# Arguments 解釋

## 模型相關
| Argument | Meaning | Value (paper setting) | train.py | find_threshold.py |
|:------:|:------:|:------:|:------:|:------:|
| ```max_freq``` | STFT spectrogram 所採用的最大頻率 (Hz) | 任何 > 1 的數字 (12) | $\checkmark$ | $\checkmark$ |
| ```rep_KV``` | Decoder 是否以 time representation 當作 Key & Value | True/False (True) | $\checkmark$ | $\checkmark$ |
| ```recover_type``` | Merge module 中，用來復原 representation 長度的方式 | conv or crossattn (conv) | $\checkmark$ | $\checkmark$ |
| ```wavelength``` | 模型輸入的波形資料長度 | 任何 > 1 的數字 (3000) | $\checkmark$ | $\checkmark$ |
| ```stft_recovertype``` | Time-frequency domain branch 中，用來抽取特徵的方式 | conv or crossattn (conv) | $\checkmark$ | $\checkmark$ |
| ```dualDomain_type``` | 結合兩個 domain representations 的方式 | concat or crossattn (concat) | $\checkmark$ | $\checkmark$ |
| ```ablation``` | 消融研究中要拔掉的模組 | 'time' or 'time-frequency' or None (None) | $\checkmark$ | $\checkmark$ |
| ```conformer_class``` | Time domain branch 中，Conformer 輸出的維度 | 任何 > 1 的數字 (8) | $\checkmark$ | $\checkmark$ |
| ```d_ffn``` | 整體模型有用到 feed-forward layer 時的維度 | 任何 > 1 的數字 (128) | $\checkmark$ | $\checkmark$ |
| ```d_model``` | Time domain branch 中，Conformer 輸入的資料維度 | 任何 > 1 的數字 (12) | $\checkmark$ | $\checkmark$ |
| ```nhead``` | 整體模型有用到 multihead attention 時的維度 | 任何 > 1 的數字 (4) | $\checkmark$ | $\checkmark$ |
| ```enc_layers``` | Time domain branch 中，Conformer 的層數 | 任何 > 1 的數字 (2) | $\checkmark$ | $\checkmark$ |
| ```dec_layers``` | Decoder 的層數 | 任何 > 1 的數字 (1) | $\checkmark$ | $\checkmark$ |
| ```dropout``` | 整體模型有用到 dropout 時的 dropout rate | 任何介於 0~1 的小數 (0.1) | $\checkmark$ | $\checkmark$ |
| ```label_type``` | 模型輸出的 label 種類 | p, other, or all (all) | $\checkmark$ | $\checkmark$ |

* ```label_type``` = 'p': 只輸出 P-phase 機率
* ```label_type``` = 'other': 只輸出 P-phase 機率與非 P-phase 機率
* ```label_type``` = 'all': 輸出 P-phase, S-phase, 與 earthquake detection 機率
---
## 資料相關
| Argument | Meaning | Value (paper setting) | train.py | find_threshold.py |
|:------:|:------:|:------:|:------:|:------:|
| ```loading_method``` | Load CWB 資料時的模式 | full, single (full) | $\checkmark$ | $\checkmark$ |
| ```dataset_opt``` | 指定使用的資料集 | 略 ('cwb') | $\checkmark$ | $\checkmark$ |
| ```aug``` | 訓練時是否使用 data augmentation | True/False (True) | $\checkmark$ | $\times$ |
| ```level``` | Load CWBSN 資料時所選的 trace_completeness 值 | 任何介於 0~4 的數字 (4) | $\checkmark$ | $\checkmark$ |
| ```instrument``` | Load CWB 資料時，選用的儀器種類 | HL, EH, HH, -1 (-1) | $\checkmark$ | $\checkmark$ |
| ```location``` | Load CWB 資料時，選用的儀器 location 種類 | 10, 20, 0, -1 (-1) | $\checkmark$ | $\checkmark$ |
| ```filter_instance``` | Load INSTANCE 時，是否套用預設的資料篩選過程 | True/False (True) | $\checkmark$ | $\checkmark$ |
| ```gaussian_noise_prob``` | Data augmentation, adding Gaussian noise 機率 | 任何介於 0~1 的小數 (0.1) | $\checkmark$ | $\times$ |
| ```channel_dropout_prob``` | Data augmentation, channel dropping out 機率 | 任何介於 0~1 的小數 (0.1) | $\checkmark$ | $\times$ |
| ```adding_gap_prob``` | Data augmentation, adding gaps 機率 | 任何介於 0~1 的小數 (0.1) | $\checkmark$ | $\times$ |

* ```dataset_opt``` 種類: cwb, stead, instance, cwbsn, tsmip, cwbsn_noise
* ```instrument``` or ```location``` = -1: 表示所有儀器全拿
---
## 訓練相關
| Argument | Meaning | Value (paper setting) | train.py | find_threshold.py |
|:------:|:------:|:------:|:------:|:------:|
| ```model_opt``` | Picker 種類 | 略 (GRADUATE) | $\checkmark$ | $\checkmark$ |
| ```loss_weight``` | 計算 loss 時，加上的 weight | 任何數字 (10) | $\checkmark$ | $\checkmark$ |
| ```workers``` | Load 資料時使用的 workers 數量 | 任何 > 1 的數字 (12) | $\checkmark$ | $\checkmark$ |
| ```batch_size``` | 略 | 略 | $\checkmark$ | $\checkmark$ |
| ```epochs``` | 略 | 略 | $\checkmark$ | $\times$ |
| ```gradient_accumulation``` | 略 | 略 | $\checkmark$ | $\times$ |
| ```clip_norm``` | 略 | 略 | $\checkmark$ | $\times$ |
| ```lr``` | learning rate | 略 | $\checkmark$ | $\times$ |
| ```patience``` | Early stopping 參數 | 任何 > 1 的數字 (7) | $\checkmark$ | $\times$ |
| ```noam``` | 是否使用 Noam optimizer | True/False (True) | $\checkmark$ | $\times$ |
| ```warmup_step```| 略 | 略 | $\checkmark$ | $\times$ |
| ```save_path```| 存放這個 checkpoint 的資料夾名稱 | 略 | $\checkmark$ | $\checkmark$ |
| ```config_path``` | 沿用已寫好的參數檔路徑 | 略 | $\checkmark$ | $\checkmark$ |
| ```device``` | 好像沒用 | 略 | $\checkmark$ | $\checkmark$ |
| ```resume_training``` | 是否沿用訓練到一半的模型繼續訓練 | True/False (False)| $\checkmark$ | $\times$ |
| ```load_specific_model``` | Load pretrained model 時，model 的檔案名稱 | 略 | $\checkmark$ | $\checkmark$ |
| ```pretrained_path``` | 沿用哪個 checkpoint 繼續訓練 | checkpoint name or none (none) | $\checkmark$ | $\times$ |

---
## 測試相關 
| Argument | Meaning | Value (paper setting) | train.py | find_threshold.py |
|:------:|:------:|:------:|:------:|:------:|
| ```threshold_type``` | 選擇的 threshold 方法 | 略 ('all') | $\times$ | $\checkmark$ |
| ```threshold_prob_start``` | 選擇的 threshold 機率 | 任何介於 0~1 的小數 | $\times$ | $\checkmark$ |
| ```threshold_prob_end``` | 略 | 任何介於 0~1 的小數 | $\times$ | $\checkmark$ |
| ```threshold_trigger_start``` | 選擇觸發 threshold 所需的 sample 數 | 任何> 1 的數字 | $\times$ | $\checkmark$ |
| ```threshold_trigger_end``` | 略 | 任何> 1 的數字 | $\times$ | $\checkmark$ |
| ```sample_tolerant``` | 定義 True positive 的容忍值 | 任何> 1 的數字 | $\times$ | $\checkmark$ |
| ```p_timestep``` | 固定所使用的 validation or testing set 的 P-phase arrival 時間點 | 任何> 1 的數字 | $\times$ | $\checkmark$ |
| ```allTest```| 測試模型在要比較的多個時間點上 | True/False | $\times$ | $\checkmark$ |
| ```do_test``` | 單獨測試模型在 testing set 上 | True/False | $\times$ | $\checkmark$ |

* ```sample_tolerant``` = 50 表示誤差值在 50 個 samples 以內都算 True positive (0.5 s)
* **找 best criteria**
  - ```threshold_type``` = all, 代表所有方法都嘗試
  - 機率值也窮舉從 ```threshold_prob_start```~```threshold_prob_end```
  - 觸發所需的 sample 數量也從 ```threshold_trigger_start```~```threshold_trigger_end``` 窮舉 (只是用當 threshold 方法是 continue or avg)
  - 固定 validation set 中，每筆資料的 P-phase arrival 在第 ```p_timestep``` 個 sample 去找 criteria
    
* **測試模型**
  - 設定 ```allTest``` = True
  - 根據找出來的 criteria，指定 ```threshold_type```, ```threshold_prob_start```, 與 ```threshold_trigger_start```

* **單獨測試模型在某個情境下**
  - 設定 ```do_test``` = True
  - 設定 criteria: ```threshold_type```, ```threshold_prob_start```, 與 ```threshold_trigger_start```
  - 設定要固定 testing set 的 P-phase arrival: ```p_timestep```
