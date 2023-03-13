#!/bin/bash

set -e

OUT="build"
make $OUT/dense-utils

echo "Compte des labels"
"$OUT/dense-utils" count-labels -l data/mnist/t10k-labels-idx1-ubyte
echo -e "\033[32mOK\033[0m"

echo "Création du réseau"
mkdir -p .test-cache
"$OUT/dense-utils" creer-reseau -n 3 -o .test-cache/reseau.bin
echo -e "\033[32mOK\033[0m"

echo "Affichage poids"
"$OUT/dense-utils" print-poids -r .test-cache/reseau.bin > /dev/null
echo -e "\033[32mOK\033[0m"

echo "Affichage biais"
"$OUT/dense-utils" print-biais -r .test-cache/reseau.bin > /dev/null
echo -e "\033[32mOK\033[0m"
