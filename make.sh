#!/bin/bash

FLAGS="-std=c99 -lm"

if [[ $1 == "preview" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p out
		gcc src/mnist/preview.c -o out/preview_mnist $FLAGS
		exit
	elif [[ $2 == "train" ]]; then
		[[ -f out/preview_mnist ]] || $0 preview build
		out/preview_mnist data/train-images-idx3-ubyte data/train-labels-idx1-ubyte
		exit
	elif [[ $2 == "t10k" ]]; then
		[[ -f out/preview_mnist ]] || $0 preview build
		out/preview_mnist data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte
		exit
	fi
fi

if [[ $1 == "test" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p out
		for i in $(ls test); do
			gcc "test/$i" -o "out/test_$(echo $i | awk -F. '{print $1}')" $FLAGS
		done
		exit
	elif [[ $2 == "run" ]]; then
		$0 test build
		for i in $(ls out/test_*); do
			echo "--- $i ---"
			$i
		done
		exit
	fi
fi
