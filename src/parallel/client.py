#!/usr/bin/python3
"""
Client se connectant au serveur Flask afin de fournir de la puissance de calcul.
"""
import json
import os
import sys
import time
import subprocess

import psutil
import requests

# Définition de constantes
CACHE = "/tmp/parallel/client_cache"  # Replace with an absolute path
DELTA = os.path.join(CACHE, "delta_shared.bin")
RESEAU = os.path.join(CACHE, "reseau_shared.bin")

if len(sys.argv) > 1:
    HOST = sys.argv[1]
else:
    HOST = input("HOST : ")

if len(sys.argv) > 2:
    SECRET = sys.argv[2]
else:
    SECRET = input("SECRET : ")

session = requests.Session()
os.makedirs(CACHE, exist_ok=True)


def get_performance():
    """
    Renvoie un indice de performance du client afin de savoir quelle quantité de données lui fournir
    """
    cores = os.cpu_count()
    max_freq = psutil.cpu_freq()[2]
    return int(cores * max_freq * 0.5)


def authenticate():
    """
    S'inscrit en tant que client auprès du serveur
    """
    performance = get_performance()
    data = {"performance": performance, "secret": SECRET}
    # Les données d'identification seront ensuite stockées dans un cookie de l'objet session
    req = session.post(f"http://{HOST}/authenticate", json=data)

    data = json.loads(req.text)
    if data["status"] != "ok":
        print("error in authenticate():", data["status"])
        sys.exit(1)
    else:
        return data


def download_network():
    """
    Récupère le réseau depuis le serveur
    """
    with session.get(f"http://{HOST}/get_network", stream=True) as req:
        req.raise_for_status()
        with open(RESEAU, "wb") as file:
            for chunk in req.iter_content(chunk_size=8192):
                file.write(chunk)


def send_delta_network(continue_=False):
    """
    Envoie le réseau différentiel et obéit aux instructions suivantes
    """
    with open(DELTA, "rb") as file:
        files = {"file": file}
        req = session.post(f"http://{HOST}/post_network", files=files)
    req_data = json.loads(req.text)

    # Actions à effectuer en fonction de la réponse
    if "instruction" not in req_data.keys():
        print(req_data["status"])
        raise NotImplementedError

    if req_data["instruction"] == "sleep":
        print(f"Sleeping {req_data['sleep_time']}s.")
        time.sleep(req_data["sleep_time"])
        send_delta_network(continue_=continue_)

    elif req_data["instruction"] == "stop":
        print(req_data["status"])
        print("Shutting down.")

    elif req_data["instruction"] == "train":
        download_network()
        train_shared(req_data["dataset"], req_data["start"], req_data["nb_elem"])

    else:
        json.dumps(req_data)
        raise NotImplementedError


def train_shared(dataset, start, nb_elem, epochs=1, out=DELTA):
    """
    Entraînement du réseau
    """
    # Utiliser un dictionnaire serait plus efficace et plus propre
    if dataset == "mnist-train":
        images = "data/mnist/train-images-idx3-ubyte"
        labels = "data/mnist/train-labels-idx1-ubyte"
    elif dataset == "mnist-t10k":
        images = "data/mnist/t10k-images-idx3-ubyte"
        labels = "data/mnist/t10k-labels-idx1-ubyte"
    else:
        print(f"Dataset {dataset} not implemented yet")
        raise NotImplementedError

    # On compile out/main si il n'existe pas encore
    if not os.path.isfile("out/main"):
        subprocess.call(["./make.sh", "build", "main"])

    # Entraînement du réseau
    subprocess.call(
        [
            "out/main", "train",
            "--epochs", str(epochs),
            "--images", images,
            "--labels", labels,
            "--recover", RESEAU,
            "--delta", out,
            "--nb-images", str(nb_elem),
            "--start", str(start),
        ],
        stdout=subprocess.DEVNULL,
    )
    return send_delta_network(continue_=True)


def __main__():
    data = authenticate()

    dataset = data["dataset"]
    start = data["start"]
    nb_elem = data["nb_elem"]

    download_network()
    # train_shared s'appelle récursivement sur lui même jusqu'à la fin du programme
    try:
        train_shared(dataset, start, nb_elem, epochs=1, out=DELTA)
    except requests.exceptions.ConnectionError:
        print("Host disconnected")


if __name__ == "__main__":
    __main__()
