# OUT_DIR_PREFIX="${1:-"SNIP-IMP"}"         # FD/ERK(+)/Uni(+)/SNIP/GraSP | - | IMP/LM/LWR # This is combination of MODE and STRATEGY
# DATASET=cifar10
# MODE="${2:-"imp"}"                   # imp / Mask / LWR
# STRATEGY="${3:-"SNIP"}"
# DATA_SIZE=1
# DATA_SIZE_2="${4:-"0·02"}"

# Checking resource consumption...
# nvidia-smi
# ps -eo pid,%cpu,%mem,rss,command | grep 'python3'
# nohup bash scripts_new/all.sh &

bash scripts_new/train_other_baseaug.sh GraSP-IMP imp GraSP 0.1
bash scripts_new/train_other_autoaug.sh GraSP-IMP imp GraSP 0.1

bash scripts_new/train_other_baseaug.sh SNIP-IMP imp SNIP 0.1
bash scripts_new/train_other_autoaug.sh SNIP-IMP imp SNIP 0.1

bash scripts_new/train_other_baseaug.sh Uni+-IMP imp Uni+ 0.1
bash scripts_new/train_other_autoaug.sh Uni+-IMP imp Uni+ 0.1

bash scripts_new/train_load-imp_baseaug.sh FD-IMP imp full_dset 0.1
bash scripts_new/train_load-imp_autoaug.sh FD-IMP imp full_dset 0.1


bash scripts_new/train_other_baseaug.sh GraSP-LWR LWR GraSP 0.1
bash scripts_new/train_other_autoaug.sh GraSP-LWR LWR GraSP 0.1

bash scripts_new/train_other_baseaug.sh SNIP-LWR LWR SNIP 0.1
bash scripts_new/train_other_autoaug.sh SNIP-LWR LWR SNIP 0.1

bash scripts_new/train_other_baseaug.sh Uni+-LWR LWR Uni+ 0.1
bash scripts_new/train_other_autoaug.sh Uni+-LWR LWR Uni+ 0.1

bash scripts_new/train_load-imp_baseaug.sh FD-LWR LWR full_dset 0.1
bash scripts_new/train_load-imp_autoaug.sh FD-LWR LWR full_dset 0.1