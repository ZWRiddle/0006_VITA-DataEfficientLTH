#!/bin/bash
set -e

FILENAME="05-06_12-03_FD-IMP_1_0.02_baseaug"     # Without the .out extension

grep 'epoch: 159\|pruning state:\|remaining weight'\
    "outputs/${FILENAME}.out" >\
    "logs/${FILENAME}_logs.txt"