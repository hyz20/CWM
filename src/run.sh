#!/bin/bash
set -e
set -x

randseed=61
dataname="KuaiRand"
windows_size=3
eps=0.5

python prepare_data.py --group_num 60 --windows_size ${windows_size} --eps ${eps} --dat_name ${dataname} --is_load 0 

randseed=61
c_inv=40
sigma=2
modelname="FM"
labelname="CWM"

CUDA_VISIBLE_DEVICES=0 python main.py --fout ../rec_datasets/WM_KuaiRand/${modelname}_${labelname}_test_${c_inv}_${sigma}_${randseed} --dat_name ${dataname} --model_name ${modelname} --label_name ${labelname} --sigma ${sigma} --c_inv ${c_inv}  --randseed ${randseed} --load_to_eval 0