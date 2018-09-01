#!/bin/bash
python3 bin/train_elmo_n_gpus.py --train_prefix '/ssd4/public/stoeckel/data/Leipzig40MT2010_lowered/Leipzig40MT2010_lowered_training/*' \
--save_dir '/ssd4/public/stoeckel/elmo/Leipzig40MT2010_lowered' --vocab_file '/ssd4/public/stoeckel/data/Leipzig40MT2010_lowered/Leipzig40MT2010_lowered.5.vocab' \
--n_tokens 1098163591 --use_gpus 2 --batchsize 256