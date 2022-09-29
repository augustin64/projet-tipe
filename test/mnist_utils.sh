#!/bin/bash

set -e

OUT="$1"
[[ -f "$OUT/mnist_utils" ]] || "$2" build mnist-utils

echo "Compte des labels"
"$OUT/mnist_utils" count-labels -l data/mnist/t10k-labels-idx1-ubyte > /dev/null
echo "OK"

echo "Création du réseau"
"$OUT/mnist_utils" creer-reseau -n 3 -o .test-cache/reseau.bin > /dev/null
echo "OK"

echo "Affichage poids"
"$OUT/mnist_utils" print-poids -r .test-cache/reseau.bin > /dev/null
echo "OK"

echo "Affichage biais"
"$OUT/mnist_utils" print-biais -r .test-cache/reseau.bin > /dev/null
echo "OK"