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
	[[ $1 ]] || set "main"
	if [[ $1 == "main" ]]; then
		echo "Compilation de src/mnist/main.c"
		$CC src/mnist/main.c -o "$OUT/main" $FLAGS
		echo "Fait."
		return 0
	elif [[ $1 == "preview" ]]; then
		echo "Compilation de src/mnist/preview.c"
		$CC src/mnist/preview.c -o "$OUT/preview_mnist" $FLAGS
		echo "Fait."
		return 0
	elif [[ $1 == "test" ]]; then
		for i in "test/"*".c"; do
			echo "Compilation de $i"
			$CC "$i" -o "$OUT/test_$(echo $i | awk -F. '{print $1}' | awk -F/ '{print $NF}')" $FLAGS
			echo "Fait."
		done
		return 0
	elif [[ $1 == "utils" ]]; then
		echo "Compilation de src/mnist/utils.c"
		$CC "src/mnist/utils.c" -o "$OUT/utils" $FLAGS
		echo "Fait."
		return 0
	else
		build main
		build preview
		build test
		build utils
		return 0
	fi
}

preview () {
	if [[ ! $1 ]]; then
		build preview
		return 0
	elif [[ $1 == "train" ]]; then
		[[ -f "$OUT/preview_mnist" ]] || $0 build preview
		"$OUT/preview_mnist" data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte
		return 0
	elif [[ $1 == "t10k" ]]; then
		[[ -f "$OUT/preview_mnist" ]] || $0 build preview
		"$OUT/preview_mnist" data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte
		return 0
	fi
}

test () {
	if [[ ! $1 ]]; then
		build test
		return 0
	elif [[ $1 == "run" ]]; then
		build test
		mkdir -p .test-cache
		for i in "$OUT/test_"*; do
			echo "--- $i ---"
			$i
		done
		for i in "test/"*".sh"; do
			echo "--- $i ---"
			chmod +x "$i"
			"$i" "$OUT" "$0"
		done
		return 0
	fi
}

train () {
	[[ -f "$OUT/main" ]] || build main
	[[ $1 ]] || set -- "train"
	[[ $2 == "-r" || $2 == "--recover" ]] && RECOVER="-r .cache/reseau.bin"
	mkdir -p .cache
	"$OUT/main" train \
		--images "data/mnist/$1-images-idx3-ubyte" \
		--labels "data/mnist/$1-labels-idx1-ubyte" \
		--out ".cache/reseau.bin" \
		$RECOVER
	return 0
}

test_reseau () {
	[[ -f "$OUT/main" ]] || build main
	[[ $1 ]] || set -- "train"
	[[ -f ".cache/reseau.bin" ]] || train train
	"$OUT/main" test \
		--images "data/mnist/$1-images-idx3-ubyte" \
		--labels "data/mnist/$1-labels-idx1-ubyte" \
		--modele ".cache/reseau.bin"
	return 0
}

recognize () {
	if [[ $1 ]]; then
		[[ $2 ]] || set -- "$2" "text"
		[[ -f "$OUT/main" ]] || build main
		[[ -f ".cache/reseau.bin" ]] || train train
		"$OUT/main" recognize \
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
	[[ -f "$OUT/utils" ]] || build utils
	"$OUT/utils" ${*:1}
	return 0
}

webserver () {
	[[ -f "$OUT/main" ]] || build main
	[[ -f ".cache/reseau.bin" ]] || train train
	FLASK_APP="src/webserver/app.py" flask run
	return 0
}

usage () {
	echo "Usage:"
	echo -e "\t$0 build       ( main | preview | train | utils | all )\n"
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
	echo $(type "$1")
	exit 1
fi;