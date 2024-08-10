#!/bin/bash
set -e

# input="2024-01-26_03-45_LayerRand_1_0.02_cifar10_baseaug"
# input="2024-01-26_03-45_LayerRand_0.02_0.02_cifar10_baseaug"
# input="2024-01-27_08-53_LayerRand_1_0.02_cifar10_baseaug"
input="01-27_12-23_LWR_1_0.02_cifar10_autoaug"
# input="01-27_12-26_LWR_0.02_0.02_cifar10_autoaug"

# output="01-26_03-45_1_0.02_baseaug.txt"
# output="01-26_03-45_0.02_0.02_baseaug.txt"
# output="01-27_08-53_1.02_0.02_baseaug.txt"
output="01-27_12-26_1.02_0.02_autoaug.txt"
# output="01-27_12-26_0.02_0.02_autoaug.txt"


grep 'epoch: \|pruning state:\|remaining weight =\|extracting layer sparsity at end of iteration\|layerwise sparsity ='\
    "outputs/${input}.out" > "plot/${output}"
    # "logs/${timestamp}_${OUT_DIR_PREFIX}_${DATA_SIZE}_${DATA_SIZE_2}_${DATASET}_baseaug_logs.txt"
    

