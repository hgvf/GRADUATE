# GRADUATE

## Installation
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
### Training
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore train.py <arguments>...
```

### Finding best picking criteria on validation set
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py <arguments>...
```


### Testing the model on different P-arrival time
```shell
$ CUDA_VISIBLE_DEVICES=<gpu-id> taskset -c <cpu-number-start>-<cpu-number_end> python -W ignore find_threshold.py --allTest True\
  --threshold_type <type-of-criteria> --threshold_prob_start <prob-of-criteria> --threshold_trigger_start <trigger-sample-of-criteria>\
  <arguments>...
```

