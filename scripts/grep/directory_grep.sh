
# Define the directory to search for files
OUTPUT_DIR="outputs"

# Loop through all files in the output directory
for FILE in "$OUTPUT_DIR"/*; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$FILE" ]; then
        # Extract the filename without the extension
        FILENAME="${FILE##*/}"
        FILENAME="${FILENAME%.*}"
        
        # echo "${FILENAME}"
        # Run the script for the current file
        grep 'epoch:\|pruning state:\|best_val = \|remaining weight\|layerwise sparsity =\|Checkpoint'\
        "outputs/${FILENAME}.out" >\
        "logs/${FILENAME}_logs.txt"
        grep 'pruning state:\|best_val = \|remaining weight'\
        "outputs/${FILENAME}.out" >\\ 
        "logs/accuracy/${FILENAME}_accuracy.txt"
        grep 'pruning state:\|layerwise sparsity =\|remaining weight'\
        "outputs/${FILENAME}.out" >\
        "logs/layerwise_sparsity/${FILENAME}_lws.txt"
        echo "completed expts ${FILENAME}"
    fi
done