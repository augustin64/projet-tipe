#!/bin/bash

FLAGS="-std=c99 -lm"
OUT="out"

if [[ $1 == "preview" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p "$OUT"
		echo "Compilation de src/mnist/preview.c"
		gcc src/mnist/preview.c -o "$OUT/preview_mnist" $FLAGS
		echo "Fait."
		exit 0
	elif [[ $2 == "train" ]]; then
		[[ -f "$OUT/preview_mnist" ]] || $0 preview build
		"$OUT/preview_mnist" data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte
		exit 0
	elif [[ $2 == "t10k" ]]; then
		[[ -f "$OUT/preview_mnist" ]] || $0 preview build
		"$OUT/preview_mnist" data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte
		exit 0
	fi
fi

if [[ $1 == "test" ]]; then
	[[ $2 ]] || set -- "$1" "build"
	if [[ $2 == "build" ]]; then
		mkdir -p "$OUT"
		for i in $(ls test); do
			echo "Compilation de test/$i"
			gcc "test/$i" -o "$OUT/test_$(echo $i | awk -F. '{print $1}')" $FLAGS
			echo "Fait."
		done
		exit 0
	elif [[ $2 == "run" ]]; then
		$0 test build
		mkdir -p .test-cache
		for i in $(ls "$OUT/test_"*); do
			echo "--- $i ---"
			$i
		done
		exit 0
	fi
fi

if [[ $1 == "build" ]]; then
	mkdir -p "$OUT"
	echo "Compilation de src/mnist/main.c"
	gcc src/mnist/main.c -o "$OUT/main" $FLAGS
	echo "Fait."
	exit 0
fi

if [[ $1 == "train" ]]; then
	[[ -f "$OUT/main" ]] || $0 build
	[[ $2 ]] || set -- "$1" "train"
	[[ $3 == "-r" || $3 == "--recover" ]] && RECOVER="-r .cache/network.bin"
	mkdir -p .cache
	"$OUT/main" train \
		--images "data/mnist/$2-images-idx3-ubyte" \
		--labels "data/mnist/$2-labels-idx1-ubyte" \
		--out ".cache/network.bin" \
		$RECOVER
	exit 0
fi

if [[ $1 == "recognize" ]]; then
	if [[ $2 ]]; then
		[[ $3 ]] || set -- "$1" "$2" "text"
		[[ -f "$OUT/main" ]] || $0 build
		[[ -f ".cache/network.bin" ]] || $0 train train
		"$OUT/main" recognize \
			--modele ".cache/network.bin" \
			--in "$2" \
			--out "$3"
		exit 0
	else
		echo "Pas de fichier d'entrée spécifié. Abandon"
		exit 1
	fi
fi

if [[ $1 == "webserver" ]]; then
	[[ -f "$OUT/main" ]] || $0 build
	[[ -f ".cache/network.bin" ]] || $0 train train
	FLASK_APP="src/webserver/app.py" flask run
	exit 0
fi

echo "Usage:"
echo -e "\t$0 preview ( build | train | t10k )"
echo -e "\t$0 test    ( build | run )"
echo -e "\t$0 build"
echo -e "\t$0 train   ( train | t10k ) ( -r | --recover )"
echo -e "\t$0 recognize [FILENAME] ( text | json )"
echo -e "\t$0 webserver\n"
echo -e "Les fichiers de test sont recompilés à chaque exécution,\nles autres programmes sont compilés automatiquement si manquants"
