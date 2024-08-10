#!/bin/bash
set -e

STRATEGY="${1:-"SNIP"}"                     # New/Load/ERK(+)/Uni(+)/SNIP(+)/GraSP(+)
MODE="${2:-"IMP"}"                          # IMP/LWR
DATASET="${3:-"cifar10"}" 
# Assigning D_SIZE value to the correct arg.
D_SIZE="${4:-"0.02"}"
DATA_SIZE=1
DATA_SIZE_2=1
if [ "$STRATEGY" = "New" ]; then
    DATA_SIZE="$D_SIZE"
else
    DATA_SIZE_2="$D_SIZE"
fi
timestamp=$(date '+%m-%d_%H-%M')
AUGMENTATION="${5:-"autoaug"}"
LoadTimeStamp=""
# Using a seed picker
SeedIndex="${6:-"0"}"
case $SeedIndex in
    0) SEED=42 ;;
    1) SEED=6 ;;
    2) SEED=66 ;;
    3) SEED=666 ;;
    4) SEED=6666 ;;
    5) SEED=66666 ;;
    6) SEED=666666 ;;
    7) SEED=6666666 ;;
    8) SEED=66666666 ;;
    9) SEED=666666666 ;;
    *) echo "Invalid input. SEED must be an integer between 0-9." ;;
esac


if [ "$STRATEGY" = "full_dset" ]; then
    OUT_DIR_PREFIX="Load-$MODE"
else
    OUT_DIR_PREFIX="$STRATEGY-$MODE"
fi

if [ "$STRATEGY" = "New" ]; then
    PROJECT_NAME="${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${AUGMENTATION}_S${SeedIndex}"
elif [ "$STRATEGY" = "full_dset" ]; then
    PROJECT_NAME="${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${AUGMENTATION}_S${SeedIndex}"
else
    PROJECT_NAME="${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE_2}_${AUGMENTATION}_S${SeedIndex}"
fi


echo "mytrain.py ${AUGMENTATION} to ${PROJECT_NAME}"
python3 "mytrain.py" --dset $DATASET --data_root data/ --data_size $DATA_SIZE --data_size_2 $DATA_SIZE_2\
    --mode $MODE --prune_strategy $STRATEGY --auto_aug \
    --seed $SEED\
    --out_dir "checkpoints/${PROJECT_NAME}"\
    --load_dir "checkpoints/${LoadTimeStamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${AUGMENTATION}"\
    --epoch 2 --pruning_iters 2 --lr 0.1 --prune_rate 0.2\
    > "outputs/${PROJECT_NAME}.out"\
    
# grep 'epoch:\|pruning state:\|best_val = \|remaining weight\|layerwise sparsity =\|Checkpoint'\
#     "outputs/${PROJECT_NAME}.out" >\
#     "logs/${PROJECT_NAME}_logs.txt"

# Step 1: Extract lines before and including the "Checkpoint 3" line, then write to the output file
awk '/^------------------------ Checkpoint 3  ------------------------/ {print; exit} {print}' \
    "outputs/${PROJECT_NAME}.out" > "logs/${PROJECT_NAME}_logs.txt"

# Step 2: Extract lines after the "Checkpoint 3" line, filter with grep, and append to the same file
awk '/^------------------------ Checkpoint 3  ------------------------/ {found=1; next} found' \
    "outputs/${PROJECT_NAME}.out" | \
    grep 'epoch:\|pruning state:\|best_val = \|remaining weight\|layerwise sparsity =\|Checkpoint' >> \
    "logs/${PROJECT_NAME}_logs.txt"


grep 'pruning state:\|best_val = \|remaining weight'\
    "outputs/${PROJECT_NAME}.out" >\
    "logs/accuracy/${PROJECT_NAME}_accuracy.txt"

grep 'pruning state:\|layerwise sparsity =\|remaining weight'\
    "outputs/${PROJECT_NAME}.out" >\
    "logs/layerwise_sparsity/${PROJECT_NAME}_lws.txt"

echo "completed expts ${PROJECT_NAME}"
