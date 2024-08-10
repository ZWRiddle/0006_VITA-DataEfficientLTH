# OUT_DIR_PREFIX="${1:-"Uni+-IMP"}"         # FD/ERK(+)/Uni(+)/SNIP/GraSP | - | IMP/LM/LWR # This is combination of MODE and STRATEGY
# DATASET=cifar10
# MODE="${2:-"imp"}"                   # imp / Mask / LWR
# STRATEGY="${3:-"Uni+"}"
# DATA_SIZE=1
# DATA_SIZE_2="${4:-"0·02"}"

# Checking resource consumption...
# nvidia-smi
# ps -eo pid,%cpu,%mem,rss,command | grep 'python3'
# nohup bash scripts_new/all.sh &


# bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 1
bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 0.006
bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 0.02
bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 0.1
bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 0.2
bash scripts_new/train_new-imp_baseaug.sh New-IMP imp full_dset 0.5