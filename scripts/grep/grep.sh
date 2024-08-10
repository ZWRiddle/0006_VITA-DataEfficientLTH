

FILENAME="05-19_14-58_Load-IMP_1_0.5_baseaug"


grep 'epoch:\|pruning state:\|best_val = \|remaining weight\|layerwise sparsity =\|Checkpoint'\
    "outputs/${FILENAME}.out" >\
    "logs/${FILENAME}_logs.txt"

# grep 'pruning state:\|best_val = \|remaining weight'\
#     "outputs/${FILENAME}.out" >\
#     "logs/accuracy/${FILENAME}_accuracy.txt"

# grep 'pruning state:\|layerwise sparsity =\|remaining weight'\
#     "outputs/${FILENAME}.out" >\
#     "logs/layerwise_sparsity/${FILENAME}_lws.txt"

echo "completed expts ${FILENAME}"
