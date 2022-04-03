#!/bin/bash

FLAGS="-std=c99 -lm"

if [[ $1 == "preview" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p out
		echo "Compilation de src/mnist/preview.c"
		gcc src/mnist/preview.c -o out/preview_mnist $FLAGS
		echo "Fait."
		exit
	elif [[ $2 == "train" ]]; then
		[[ -f out/preview_mnist ]] || $0 preview build
		out/preview_mnist data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte
		exit
	elif [[ $2 == "t10k" ]]; then
		[[ -f out/preview_mnist ]] || $0 preview build
		out/preview_mnist data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte
		exit
	fi
fi

if [[ $1 == "test" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p out
		for i in $(ls test); do
			echo "Compilation de test/$i"
			gcc "test/$i" -o "out/test_$(echo $i | awk -F. '{print $1}')" $FLAGS
			echo "Fait."
		done
		exit
	elif [[ $2 == "run" ]]; then
		$0 test build
		mkdir -p .test-cache
		for i in $(ls out/test_*); do
			echo "--- $i ---"
			$i
		done
		exit
	fi
fi

echo "Usage:"
echo -e "\t$0 preview ( build | train | t10k )"
echo -e "\t$0 test    ( build | run )\n"
echo -e "Les fichiers de test sont recompilés à chaque exécution,\nles autres programmes sont compilés automatiquement si manquants"
