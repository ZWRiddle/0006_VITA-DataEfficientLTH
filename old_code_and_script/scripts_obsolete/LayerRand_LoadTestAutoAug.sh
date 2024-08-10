#!/bin/bash
set -e

OUT_DIR_PREFIX="LWR"
DATASET=cifar10
DATA_SIZE=1
MODE="imp"
DATA_SIZE_2=0.02
timestamp=$(date '+%Y-%m-%d_%H-%M')
LoadTimeStamp="2024-01-26_03-45"

echo "train_LoadLWR.py baseaug to ${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug"
python3 "train_LoadLWR.py" --dset ${DATASET} --data_root data/ --data_size $DATA_SIZE --data_size_2 $DATA_SIZE_2 --mode $MODE\
    --out_dir "checkpoints/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug"\
    > "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug.out"\
    --epoch 160 --pruning_iters 16 --lr 0.1 --prune_rate 0.2\
    --load_dir "checkpoints/${LoadTimeStamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug"\
    --dist --auto_aug
grep 'epoch:\|pruning state:\|remaining weight ='\
    "outputs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug.out" >\
    "logs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug_logs.txt"

echo "completed expts ${OUT_DIR_PREFIX} ${DATA_SIZE} ${MODE} ${DATA_SIZE_2}"

