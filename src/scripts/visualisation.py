#!/usr/bin/python3
"""
Ensemble de fonctions permettant de visualiser
les différentes données disponibles dans le réseau de neurones
"""
import json
import math
import os

import png
from matplotlib import pyplot as plt

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

def image_from_file(filepath, dest="./images/"):
    """
    Enregistre un ensemble d'images au format PNG
    à partir d'un fichier texte comprenant une liste d'images
    chaque image étant un tableau de poids entre 0 et 255
    """
    os.makedirs(dest, exist_ok=True)
    with open(filepath, "r", encoding="utf8") as fp:
        data = json.load(fp)

    nb_elem = len(data)
    for i in range(nb_elem):
        png.from_array(data[i], 'L').save(os.path.join(dest, f"{i}.png"))


def image_from_list(filepath, exp=False):
    """
    Enregistre une liste de poids sous forme d'une image
    exp sert à spécifier si il faut passer à une forme exponentielle
    afin de mieux distinguer les points prédominants.
    """
    with open(filepath, "r", encoding="utf8") as fp:
        data = json.load(fp)

    mini = min(data)
    data = [i-mini for i in data] # Set min to 0

    maxi = max(data)
    if exp:
        ratio = 255/math.exp(maxi)
        data = [int(math.exp(i)*ratio) for i in data]
    else:
        ratio = 255/maxi
        data = [int(i*ratio) for i in data]

    new_data = [[0 for i in range(IMAGE_WIDTH)] for j in range(IMAGE_HEIGHT)]

    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            new_data[i][j] = data[i*IMAGE_HEIGHT+j]

    return new_data


def graph_from_test_reseau(erreurs, reussites):
    """
    Affiche un graphique à partir d'un fichier contenant les
    réussites et d'un autre contenant les erreurs (sortie brutes de out/mnist_main)
    """
    with open(erreurs, "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("--- Image")[1:]
    data = [i.split("\n")[:IMAGE_HEIGHT] for i in data]
    labels = []
    for i in data:
        labels.append((int(i[0].split(",")[1][1]), int(i[0][-5])))

    data = [[float(j[IMAGE_HEIGHT+5:]) for j in i if j[IMAGE_HEIGHT+5:] != ''] for i in data]

    x = []
    y = []

    for i, label in enumerate(labels):
        x.append(data[i][label[0]])
        y.append(data[i][label[1]])

    plt.plot(x, y, "+r")

    with open(reussites, "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("--- Image")[1:]
    data = [i.split("\n")[:IMAGE_HEIGHT] for i in data]
    labels = []
    for i in data:
        labels.append((int(i[0].split(",")[1][1]), int(i[0][-5])))

    data = [[float(j[IMAGE_HEIGHT+5:]) for j in i if j[IMAGE_HEIGHT+5:] != ''] for i in data]

    x = []
    y = []

    for i, label in enumerate(labels):
        x.append(data[i][label[0]])
        y.append(data[i][label[1]])

    plt.plot(x, y, "+b")
    plt.xlabel("Réel")
    plt.ylabel("Prévision")
    plt.legend()
    plt.show()


def images_neurons(neurons, dest="neurons", exp=False):
    """
    Appelle le programme C ainsi que la fonction image_from_list
    Afin de créer un ensemble d'image visualisant les poids
    """
    os.makedirs(dest, exist_ok=True)
    data = []
    for i in neurons:
        os.system(f"./make.sh utils print-poids-neurone --reseau \
            .cache/reseau.bin --neurone {i} > .cache/poids.txt")
        data.append(image_from_list(".cache/poids.txt", exp=exp))

    new_data = data.copy()

    for i, _ in enumerate(new_data):
        for j, _ in enumerate(new_data[i]):
            for k, _ in enumerate(new_data[i][j]):
                for l, _ in enumerate(new_data):
                    if i != l:
                        new_data[i][j][k] -= data[l][j][k] * 0.11
                new_data[i][j][k] = max(int(new_data[i][j][k]), 0)
    
    for i in neurons:
        png.from_array(data[i], 'L').save(os.path.join(dest, f"{i}.png"))
