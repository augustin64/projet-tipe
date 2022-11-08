#!/bin/bash

set -e

OUT="build"
make $OUT/mnist-utils

echo "Compte des labels"
"$OUT/mnist-utils" count-labels -l data/mnist/t10k-labels-idx1-ubyte
echo "OK"

echo "Création du réseau"
mkdir -p .test-cache
"$OUT/mnist-utils" creer-reseau -n 3 -o .test-cache/reseau.bin
echo "OK"

echo "Affichage poids"
"$OUT/mnist-utils" print-poids -r .test-cache/reseau.bin > /dev/null
echo "OK"

echo "Affichage biais"
"$OUT/mnist-utils" print-biais -r .test-cache/reseau.bin > /dev/null
echo "OK"
