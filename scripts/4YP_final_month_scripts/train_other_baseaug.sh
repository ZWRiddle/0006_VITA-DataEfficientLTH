#!/bin/bash
set -e

OUT_DIR_PREFIX="${1:-"SNIP-IMP"}"         # FD/ERK(+)/Uni(+)/SNIP/GraSP | - | IMP/LM/LWR # This is combination of MODE and STRATEGY
DATASET=cifar10
MODE="${2:-"imp"}"                   # imp / Mask / LWR
STRATEGY="${3:-"SNIP"}"
DATA_SIZE=1
DATA_SIZE_2="${4:-"0.02"}"
timestamp=$(date '+%m-%d_%H-%M')
AUGMENTATION="baseaug"
LoadTimeStamp=""
SEED="${5:-"42"}"


echo "mytrain.py ${AUGMENTATION} to ${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}"
python3 "mytrain.py" --dset $DATASET --data_root data/ --data_size $DATA_SIZE --data_size_2 $DATA_SIZE_2\
    --mode $MODE --prune_strategy $STRATEGY \
    --seed $SEED\
    --out_dir "checkpoints/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}"\
    --epoch 160 --pruning_iters 20 --lr 0.1 --prune_rate 0.2\
    > "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}.out"\
    
grep 'epoch:\|pruning state:\|best_val = \|remaining weight\|layerwise sparsity =\|Checkpoint'\
    "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}.out" >\
    "logs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}_logs.txt"

grep 'pruning state:\|best_val = \|remaining weight'\
    "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}.out" >\
    "logs/accuracy/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}_accuracy.txt"

grep 'pruning state:\|layerwise sparsity =\|remaining weight'\
    "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}.out" >\
    "logs/layerwise_sparsity/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}_lws.txt"

echo "completed expts ${OUT_DIR_PREFIX} ${MODE} ${STRATEGY} ${DATA_SIZE} ${DATA_SIZE_2} ${AUGMENTATION}"
