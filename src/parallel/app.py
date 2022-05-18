#!/usr/bin/python3
"""
Serveur Flask pour entraîner le réseau sur plusieurs machines en parallèle.
"""
import os
import random
from secrets import token_urlsafe

from flask import Flask, request, send_file

from clients import Client, clients

DATASET = "mnist-train"
SECRET = str(random.randint(1000, 10000))
CACHE = ".cache"

os.makedirs(CACHE, exist_ok=True)

app = Flask(__name__)
print(f" * Secret: {SECRET}")

@app.route("/authenticate", methods = ['POST'])
def authenticate():
    """
    Authentification d'un nouvel utilisateur
    """
    if not request.is_json:
        return {
            "status":
            "request format is not json"
        }
    content = request.get_json()
    if content["secret"] != SECRET:
        return {"status": "invalid secret"}

    performance = content["performance"]
    token = token_urlsafe(30)

    while token in [client.token for client in clients]:
        token = token_urlsafe(30)

    clients.append(Client(performance, token))

    data = {}
    data["token"] = token
    data["nb_elem"] = performance
    data["start"] = 0
    data["dataset"] = DATASET
    data["status"] = "ok"

    return data


@app.route("/get_network")
def get_network():
    """
    Renvoie le réseau neuronal
    """
    if not request.is_json:
        return {
            "status":
            "request format is not json"
        }
    token = request.get_json()["token"]
    if token not in [client.token for client in clients]:
        return {
            "status":
            "token invalide"
        }
    return send_file(
        os.path.join(CACHE, "reseau.bin")
    )
