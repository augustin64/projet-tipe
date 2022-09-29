#!/usr/bin/python3
from flask import Flask, render_template, request
import subprocess
import json

MAGIC_NUMBER = 2051

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mnist")
def mnist():
    return render_template("mnist.html")

@app.route("/post", methods=["POST"])
def post_json_handler():
    """
    Gère les requêtes POST
    """
    if request.is_json:
        content = request.get_json()
        req_type = content["type"]
        if req_type == "prediction":
            dataset = content["dataset"]
            image = content["data"]
            if dataset == "mnist":
                return recognize_mnist(image)
    return {"status": "404"}


def recognize_mnist(image):
    """Appelle le programme C reconnaissant les images"""
    # Créer le fichier binaire
    write_image_to_binary(image, ".cache/image-idx3-ubyte")

    try:
        output = subprocess.check_output([
            'out/mnist_main',
            'recognize',
            '--modele', '.cache/reseau.bin',
            '--in', '.cache/image-idx3-ubyte',
            '--out', 'json'
        ]).decode("utf-8")
        json_data = json.loads(output.replace("nan", "0.0"))["0"]
        return {"status": 200, "data": json_data}
    except subprocess.CalledProcessError:
        return {
            "status": 500,
            "data": "Internal Server Error"
        }

def write_image_to_binary(image, filepath):
    byteorder = "big"

    bytes_ = MAGIC_NUMBER.to_bytes(4, byteorder=byteorder)
    bytes_ += (1).to_bytes(4, byteorder=byteorder)
    bytes_ += len(image).to_bytes(4, byteorder=byteorder)
    bytes_ += len(image[0]).to_bytes(4, byteorder=byteorder)
    for row in image:
        for nb in row:
            bytes_ +=  int(nb).to_bytes(1, byteorder=byteorder)

    with open(filepath, "wb") as f:
        f.write(bytes_)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')