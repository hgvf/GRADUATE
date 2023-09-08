# GRADUATE

## Installation

---
## Step1. Training
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore train.py <arguments>...
```
---
## Step2. Finding best picking criteria on validation set
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py <arguments>...
```
---

## Step3. Testing the model on different P-arrival time
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py --allTest True\
  --threshold_type <type-of-criteria> --threshold_prob_start <prob-of-criteria> --threshold_trigger_start <trigger-sample-of-criteria>\
  <arguments>...
```
---
