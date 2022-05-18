#!/usr/bin/python3
"""
Client se connectant au serveur Flask afin de fournir de la puissance de calcul.
"""
import json
import os
import sys

import psutil
import requests

CACHE = ".cache"
DELTA = os.path.join(CACHE, "delta_shared.bin")
RESEAU = os.path.join(CACHE, "reseau_shared.bin")
SECRET = input("SECRET : ")
HOST = input("HOST : ")
os.makedirs(CACHE, exist_ok=True)


def get_performance():
    """
    Renvoie un indice de performance du client afin de savoir quelle quantité de données lui fournir
    """
    cores = os.cpu_count()
    max_freq = psutil.cpu_freq()[2]
    return int(cores * max_freq * 0.01)


def authenticate():
    """
    S'inscrit en tant que client auprès du serveur
    """
    performance = get_performance()
    data = {
        "performance":performance,
        "secret": SECRET
    }
    req = requests.post(
        f"http://{HOST}/authenticate",
        json=data
    )
    data = json.loads(req.text)
    if data["status"] != "ok":
        print("authentication error:", data["status"])
        sys.exit(1)
    else:
        return data


def download_network(token):
    """
    Récupère le réseau depuis le serveur
    """
    data = {"token": token}
    with requests.get(f"http://{HOST}/get_network", stream=True, json=data) as req:
        req.raise_for_status()
        with open(os.path.join(CACHE, RESEAU), "wb") as file:
            for chunk in req.iter_content(chunk_size=8192):
                file.write(chunk)



def train_shared(dataset, start, nb_elem, epochs=1, out=DELTA):
    """
    Entraînement du réseau
    """
    raise NotImplementedError


def __main__():
    data = authenticate()

    token = data["token"]
    dataset = data["dataset"]

    start = data["start"]
    nb_elem = data["nb_elem"]

    download_network(token)

    while True:
        train_shared(dataset, start, nb_elem, epochs=1, out=DELTA)


if __name__ == "__main__":
    __main__()
