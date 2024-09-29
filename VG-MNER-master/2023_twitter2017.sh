#!/usr/bin/env bash
# Required environment variables:
# batch_size (recommendation: 8 / 16)
# lr: learning rate (recommendation: 3e-5 / 5e-5)
# seed: random seed, default is 1234
# BERT_NAME: pre-trained text model name ( bert-*)
# max_seq: max sequence length
# sample_ratio: few-shot learning, default is 1.0
# save_path: model saved path

DATASET_CHOOSE="umgf" # umt umgf hvpnet
DATASET_NAME="twitter17"
BERT_NAME="bert_base_uncased"

CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --dataset_choose=${DATASET_CHOOSE} \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=12 \
        --batch_size=16 \
        --bert_lr=3e-5 \
        --crf_lr=3e-1 \
        --other_lr=1e-3 \
        --warmup_ratio=0.1 \
        --eval_begin_epoch=3 \
	--warmup_epoch=20 \
        --warmup_power=2.0 \
        --seed=42 \
        --do_train \
        --ignore_idx=0 \
        --save_path=your_ckpt_path \
        --dropout_prob=0.2 \
        --negative_slope1=0.01 \
	--negative_slope2=0.01 \
        --dyn_k=10 \
        --embed_dim=768 \
        --queue_size=4096 \
        --momentum=0.995 \
        --temp=0.1 \
        --alpha=0.4 \
        --itc_loss_weight=0.1 \
        --ce_loss_weight=0.9
