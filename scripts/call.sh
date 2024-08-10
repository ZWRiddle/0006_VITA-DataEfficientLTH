#!/bin/bash
set -e

# STRATEGY="${1:-"SNIP"}"                     # New/Load/ERK(+)/Uni(+)/SNIP(+)/GraSP(+)
# MODE="${2:-"IMP"}"                          # IMP/LWR
# DATASET="${3:-"cifar10"}" 
# D_SIZE="${4:-"0.02"}"
# AUGMENTATION="${5:-"autoaug"}"
# SeedIndex="${6:-"0"}"


bash scripts/mytrain.sh SNIP+ IMP cifar10 0.02 autoaug 1