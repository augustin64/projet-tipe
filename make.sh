#!/bin/bash

if [[ $1 == "preview" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p out
		gcc src/mnist/preview.c -o out/preview_mnist
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
