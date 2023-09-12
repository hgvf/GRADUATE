# GRADUATE

## Installation
### Step0. Clone the repository
```shell
$ git clone https://github.com/hgvf/GRADUATE.git
```

### Step1. Create a new Anaconda virtual environment
```shell
$ conda create --name GRADUATE python=3.8
$ conda activate GRADUATE
```

### Step2. Install the dependencies
```shell
$ pip install -r requirements.txt
```

### Step3. Install the custom seisbench
```shell
$ cd seisbench
$ pip install .
```

---
## Loading the dataset to the Seisbench format
* 更改資料路徑
* utils.py -> load_dataset() -> 裡面所有 kwargs 的 ```download_kwargs```，改為對應的資料集路徑

## (Optional) Replace the line notify token
* 如果要用 Line notify 的話，要去改 line notify token，不然都發到我的群組= =
* train.py & find_threshold.py -> toLine() -> 裡面的 ```token``` 改成自己的
---

## Training
* Model arguments:
  - **Time domain branch**: ```conformer_class```, ```d_ffn```, ```d_model```, ```nhead```, ```enc_layers```, ```dropout```
  - **Time-frequency domain branch**: ```stft_recovertype```, ```max_freq```
  - **Decoder**: ```dec_layers```, ```rep_KV```, ```dualDomain_type```
  - **Others**: ```label_type```, ```wavelength```, ```ablation```
* Training arguments:
  - **Checkpoint name**: ```save_path```
  - **Training with data augmentations**: ```aug```
  - **Filtering the dataset**: ```level```, ```instrument```, and ```location``` for CWB, ```filter_instance``` for INSTANCE,
  - **Dataset**: ```dataset_opt```
  - **Model type**: ```model_opt```

```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore train.py <arguments>...
```

## Evaluate the picker
* Scenario 1: Finding the best criteria on validation set

```shell
$ python average_checkpoints.py --save_path <checkpoint-name> -n <number-of-checkpoints-to-average>

$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py --load_specific_model averaged_checkpoint/
  <arguments>...
```

* (Optional) Scenario 2: Testing the model on testing set only 
```shell
$ python average_checkpoints.py --save_path <checkpoint-name> -n <number-of-checkpoints-to-average>

$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py --load_specific_model averaged_checkpoint/
  --threshold_type <type-of-threshold> --threshold_prob_start <probability-of-threshold> --threshold_trigger_start <trigger-sample-of-threshold>/
  --p_timestep <fixed-parrival-at-timestep> --do_test True <arguments>...
```

## Testing the model on different P-arrival time
* Arguments:
  - **Testing on multiple P-phase arrival**: ```allTest```

```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py --allTest True --load_specific_model averaged_checkpoint/
  --threshold_type <type-of-criteria> --threshold_prob_start <prob-of-criteria> --threshold_trigger_start <trigger-sample-of-criteria>\
  <arguments>...
```

---
## (Appendix) Using different characteristic of dataset for analyzing the model
* 自己研究

```shell
$ cd analyze
```

