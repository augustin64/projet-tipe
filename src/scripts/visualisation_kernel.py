#!/usr/bin/python3
#
# Use it like that:
# ```bash
# build/cnn-export print-poids-kernel-cnn -m out.bin > fichier.json
# python -c "from src.scripts import visualisation_kernel"
# ```
import os
import json
from .visualisation import *

fichier = "fichier.json" # Fichier d'entrée
output_dir = "test-img-vis-kernel" # Dossier de sortie

with open(fichier, 'r') as f:
	data = json.load(f)

IMAGE_WIDTH = 5  # Manière très grossière de modifier le script visualisation.py
IMAGE_HEIGHT = 5 # pour utiliser une de ses fonctions sans la réécrire

os.makedirs(output_dir, exist_ok=True)
for key in data.keys():
    for i in range(len(data[key])):
        for j in range(len(data[key][i])):
            IMAGE_WIDTH = len(data[key][i][j])
            IMAGE_HEIGHT = len(data[key][i][j][0])
            png.from_array([
                [max(0,min(int((j2+0.35)*255),255)) for j2 in i2] # Valeurs arbitraires à changer selon les valeurs du kernel
                for i2 in data[key][i][j]], 'L').save(f"{output_dir}/{key}-{i}-{j}.png"
            )

