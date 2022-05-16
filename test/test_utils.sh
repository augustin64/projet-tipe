#!/bin/bash

set -e

OUT="$1"
[[ -f "$OUT/utils" ]] || "$2" build utils

echo "Compte des labels"
"$OUT/utils" count-labels -l data/mnist/t10k-labels-idx1-ubyte > /dev/null
echo "OK"

echo "Création du réseau"
"$OUT/utils" creer-reseau -n 3 -o .test-cache/reseau.bin > /dev/null
echo "OK"

echo "Affichage poids"
"$OUT/utils" print-poids -r .test-cache/reseau.bin > /dev/null
echo "OK"

echo "Affichage biais"
"$OUT/utils" print-biais -r .test-cache/reseau.bin > /dev/null
echo "OK"