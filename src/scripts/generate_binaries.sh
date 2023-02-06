#!/bin/bash
# Use with
# ```bash
# $ python -i src/scripts/benchmark_binary.py
# >>> compare_binaries(["binaries/"+i for i in os.listdir("binaries")], tries=10, dataset='train')
# ```
BIN_OUT=binaries
FILE_TO_MODIFY="src/cnn/include/train.h"
VARIABLE_TO_MODIFY="LEARNING_RATE"

mkdir -p "$BIN_OUT"
rm -rf "$BIN_OUT"/*

values="0 5 25 50" # Example values

for val in $values; do
	# For a variable
	# sed -i 's/'"$VARIABLE_TO_MODIFY"'=.*/'"$VARIABLE_TO_MODIFY'='$val"';/g' "$FILE_TO_MODIFY"
	# For a define
	sed -i 's/#define '"$VARIABLE_TO_MODIFY"' .*$/#define '"$VARIABLE_TO_MODIFY"' '"$val"'/g' "$FILE_TO_MODIFY"
	make all
	cp build/cnn-main "$BIN_OUT"/"$VARIABLE_TO_MODIFY=$val"
done
