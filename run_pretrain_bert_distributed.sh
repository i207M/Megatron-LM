#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.3
# export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

EXP_NAME=bert_pretrain_origin
DATA_PATH=/data2/yaojiachen20/pretrain_data/wiki_bookcorpus_text_sentence
CHECKPOINT_PATH=runs/$EXP_NAME

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 8 \
       --global-batch-size 1024 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 2000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file bert-base-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 1990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.01 \
       --init-method-std 0.02 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --tensorboard-dir logs/$EXP_NAME \
       --log-validation-ppl-to-tensorboard \
       --seed 5678 \
       # --checkpoint-activations \
