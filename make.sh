#!/bin/bash
# C compiler can be defined with the $CC environment variable
OUT="out"

set -e

compile_cuda () {
	# nvcc will compile .c files as if they did not have
	# CUDA program parts so we need to copy them to .cu
	mv $1 "$1"u
	echo "" > "$1" # If we compile file.cu, file.c needs to exist to, even if it is empty
	nvcc "$1"u ${*:1}
	mv "$1"u "$1"
}

build () {
	mkdir -p "$OUT"
	[[ $1 ]] || set "mnist-main"
	case $1 in
		"mnist-main")
			echo "Compilation de src/mnist/main.c";
			$CC src/mnist/main.c -o "$OUT/mnist_main" $FLAGS;
			echo "Fait.";
			return 0;;
		"mnist-preview")
			echo "Compilation de src/mnist/preview.c";
			$CC src/mnist/preview.c -o "$OUT/mnist_preview" $FLAGS;
			echo "Fait.";
			return 0;;
		"mnist-utils")
			echo "Compilation de src/mnist/utils.c";
			$CC "src/mnist/utils.c" -o "$OUT/mnist_utils" $FLAGS;
			echo "Fait.";
			return 0;;
		"mnist")
			build mnist-main;
			build mnist-preview;
			build mnist-utils;;

		"cnn-main")
			echo "Compilation de src/cnn/main.c";
			$CC "src/cnn/main.c" -o "$OUT/cnn_main" $FLAGS;
			echo "Fait.";;
		"cnn")
			build cnn-main;;

		"test")
			rm "$OUT/test_"* || true;
			for i in "test/"*".c"; do
				echo "Compilation de $i";
				$CC "$i" -o "$OUT/test_$(echo $i | awk -F. '{print $1}' | awk -F/ '{print $NF}')" $FLAGS;
				echo "Fait.";
			done;
			if ! command -v nvcc &> /dev/null; then
				echo "Tests CUDA évités";
			elif [[ $SKIP_CUDA == 1 ]]; then
				echo "Tests CUDA évités";
			else
				for i in "test/"*".cu"; do
					echo "Compilation de $i";
					nvcc "$i" -o "$OUT/test_$(echo $i | awk -F. '{print $1}' | awk -F/ '{print $NF}')";
					echo "Fait.";
				done;
			fi;
			return 0;;
		*)
			echo -e "\033[1m\033[34m###  Building mnist  ###\033[0m";
			build mnist-main;
			build mnist-preview;
			build mnist-utils;
			echo -e "\033[1m\033[34m###   Building cnn   ###\033[0m";
			build cnn;
			echo -e "\033[1m\033[34m###  Building tests  ###\033[0m";
			build test;
			return 0;;
	esac
}

preview () {
	case $1 in
		"train")
			[[ -f "$OUT/mnist_preview" ]] || $0 build mnist-preview;
			"$OUT/mnist_preview" data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte;
			return 0;;
		"t10k")
			[[ -f "$OUT/mnist_preview" ]] || $0 build mnist-preview;
			"$OUT/mnist_preview" data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte;
			return 0;;
		*)
		build mnist-preview;
		return 0;;
	esac
}

test () {
	case $1 in
		"run")
			build test;
			mkdir -p .test-cache;
			for i in "$OUT/test_"*; do
				echo "--- $i ---";
				$i;
			done
			for i in "test/"*".sh"; do
				echo "--- $i ---";
				chmod +x "$i";
				"$i" "$OUT" "$0";
			done;
			return 0;;
		*)
			build test;
			return 0;;
	esac
}

train () {
	[[ -f "$OUT/mnist_main" ]] || build mnist-main
	[[ $1 ]] || set -- "train"
	[[ $2 == "-r" || $2 == "--recover" ]] && RECOVER="-r .cache/reseau.bin"
	mkdir -p .cache
	"$OUT/mnist_main" train \
		--images "data/mnist/$1-images-idx3-ubyte" \
		--labels "data/mnist/$1-labels-idx1-ubyte" \
		--out ".cache/reseau.bin" \
		$RECOVER
	return 0
}

test_reseau () {
	[[ -f "$OUT/mnist_main" ]] || build mnist-main
	[[ $1 ]] || set -- "train"
	[[ -f ".cache/reseau.bin" ]] || train train
	"$OUT/mnist_main" test \
		--images "data/mnist/$1-images-idx3-ubyte" \
		--labels "data/mnist/$1-labels-idx1-ubyte" \
		--modele ".cache/reseau.bin"
	return 0
}

recognize () {
	if [[ $1 ]]; then
		[[ $2 ]] || set -- "$2" "text"
		[[ -f "$OUT/mnist_main" ]] || build mnist-main
		[[ -f ".cache/reseau.bin" ]] || train train
		"$OUT/mnist_main" recognize \
			--modele ".cache/reseau.bin" \
			--in "$1" \
			--out "$2"
		return 0
	else
		echo "Pas de fichier d'entrée spécifié. Abandon"
		return 1
	fi
}

utils () {
	[[ -f "$OUT/mnist_utils" ]] || build mnist-utils
	"$OUT/mnist_utils" ${*:1}
	return 0
}

webserver () {
	[[ -f "$OUT/mnist_main" ]] || build mnist-main
	[[ -f ".cache/reseau.bin" ]] || train train
	FLASK_APP="src/webserver/app.py" flask run
	return 0
}

usage () {
	echo "Usage:"
	echo -e "\t$0 build       ( test | all | ... )"
	echo -e "\t\t\tmnist:  mnist"
	echo -e "\t\t\t\tmnist-main"
	echo -e "\t\t\t\tmnist-preview"
	echo -e "\t\t\t\tmnist-utils"
	echo -e "\t\t\tcnn:    cnn"
	echo -e "\t\t\t\tcnn-main\n"
	echo -e "\t$0 train       ( train | t10k ) ( -r | --recover )"
	echo -e "\t$0 preview     ( train | t10k )"
	echo -e "\t$0 test_reseau ( train | t10k )\n"
	echo -e "\t$0 recognize   [FILENAME] ( text | json )"
	echo -e "\t$0 utils       ( help )\n"
	echo -e "\t$0 test        ( run )"
	echo -e "\t$0 webserver\n"
	echo -e "Les fichiers de test sont recompilés à chaque exécution,\nles autres programmes sont compilés automatiquement si manquants\n"
	echo -e "La plupart des options listées ici sont juste faites pour une utilisation plus rapide des commandes fréquentes,"
	echo -e "d'autres options sont uniquement disponibles via les fichiers binaires dans '$OUT'"
}


[[ $CC ]] || CC=gcc
if [[ "$CC" == "gcc" ]]; then
	FLAGS="-std=c99 -lm -lpthread -Wall -Wextra" # GCC flags
elif [[ "$CC" == "nvcc" ]]; then
	CC=compile_cuda
	FLAGS="" # NVCC flags
else
	FLAGS=""
fi

if [[ $1 && $(type "$1") = *"is a"*"function"* || $(type "$1") == *"est une fonction"* ]]; then
	$1 ${*:2} # Call the function
else
	usage
	exit 1
fi;