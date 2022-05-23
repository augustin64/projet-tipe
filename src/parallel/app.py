#!/usr/bin/python3
"""
Serveur Flask pour entraîner le réseau sur plusieurs machines en parallèle.
"""
import os
import time
import random
import subprocess
from threading import Thread
from secrets import token_urlsafe

from flask import Flask, request, send_from_directory, session

from structures import (Client, NoMoreJobAvailableError, Training,
                        TryLaterError, clients)

# Définitions de variables
DATASET = "mnist-train"
TEST_SET = "mnist-t10k"
SECRET = str(random.randint(1000, 10000))
CACHE = "/tmp/parallel/app_cache"  # À remplacer avec un chemin absolu
BATCHS = 10
RESEAU = os.path.join(CACHE, "reseau.bin")

training = Training(BATCHS, DATASET, TEST_SET, CACHE)

os.makedirs(CACHE, exist_ok=True)
# On crée un réseau aléatoire si il n'existe pas encore
if not os.path.isfile(RESEAU):
    if not os.path.isfile("out/main"):
        subprocess.call(["./make.sh", "build", "main"])
    subprocess.call(
    [
        "out/main", "train",
        "--epochs", "0",
        "--images", "data/mnist/train-images-idx3-ubyte",
        "--labels", "data/mnist/train-labels-idx1-ubyte",
        "--out", RESEAU
    ])
    print(f" * Created {RESEAU}")
else:
    print(f" * {RESEAU} already exists")


app = Flask(__name__)
# On définit une clé secrète pour pouvoir utiliser des cookies de session
app.config["SECRET_KEY"] = token_urlsafe(40)
print(f" * Secret: {SECRET}")


@app.route("/authenticate", methods=["POST"])
def authenticate():
    """
    Authentification d'un nouvel utilisateur
    """
    if not request.is_json:
        return {"status": "request format is not json"}
    content = request.get_json()
    if content["secret"] != SECRET:
        return {"status": "invalid secret"}

    token = token_urlsafe(30)
    while token in clients.keys():
        token = token_urlsafe(30)

    clients[token] = Client(content["performance"], token)

    # On prépare la réponse du serveur
    data = {}
    data["status"] = "ok"
    data["dataset"] = training.dataset
    session["token"] = token

    try:
        clients[token].get_job(training)
        data["nb_elem"] = clients[token].performance
        data["start"] = clients[token].start
        data["instruction"] = "train"

    except NoMoreJobAvailableError:
        data["status"] = "Training already ended"
        data["nb_elem"] = 0
        data["start"] = 0
        data["instruction"] = "stop"

    except TryLaterError:
        data["status"] = "Wait for next batch"
        data["nb_elem"] = 0
        data["start"] = 0
        data["instruction"] = "sleep"
        data["sleep_time"] = 0.2

    return data


@app.route("/post_network", methods=["POST"])
def post_network():
    """
    Applique le patch renvoyé dans le nouveau réseau
    """
    token = session.get("token")
    if not token in clients.keys():
        return {"status": "token invalide"}

    while training.is_patch_locked():
        time.sleep(0.1)

    request.files["file"].save(training.delta)
    training.patch()
    training.computed_images += clients[token].performance
    # Préparation de la réponse
    data = {}
    data["status"] = "ok"
    data["dataset"] = training.dataset

    try:
        clients[token].get_job(training)
        data["dataset"] = training.dataset
        data["nb_elem"] = clients[token].performance
        data["start"] = clients[token].start
        data["instruction"] = "train"

    except NoMoreJobAvailableError:
        data["status"] = "Training already ended"
        data["nb_elem"] = 0
        data["start"] = 0
        data["instruction"] = "stop"

    except TryLaterError:
        Thread(target=training.test_network()).start()
        data["status"] = "Wait for next batch"
        data["nb_elem"] = 0
        data["start"] = 0
        data["instruction"] = "sleep"
        data["sleep_time"] = 0.02

    return data


@app.route("/get_network", methods=["GET", "POST"])
def get_network():
    """
    Renvoie le réseau neuronal
    """
    token = session.get("token")
    if not token in clients.keys():
        return {"status": "token invalide"}

    if token not in clients.keys():
        return {"status": "token invalide"}

    return send_from_directory(directory=CACHE, path="reseau.bin")
