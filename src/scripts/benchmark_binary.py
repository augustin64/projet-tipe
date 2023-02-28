#!/usr/bin/python3
#
# Steps to use:
# - modify src/scripts/generate_binaries.sh to suit your needs
# - execute the following commands:
# ```bash
# src/scripts/generate_binaries.sh
# python -i src/scripts/benchmark_binary.py
# >>> compare_binaries(["binaries/"+i for i in os.listdir("binaries")], tries=4, dataset='train')
# ```
# tries is the number of epochs to train on and dataset is the dataset to use ('train' or 't10k')
import subprocess
import json
import os

from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
    progress = tqdm
except ModuleNotFoundError:
    progress = lambda x : x

def train(binary, base_network, tries=3, dataset="train"):
    temp_out = f".cache/tmp-{abs(hash(binary))}.bin"
    results = []

    # 1e époque training
    try:
        train_output = subprocess.check_output([
            binary,
            "train",
            "--dataset", "mnist",
            "--images", f"data/mnist/{dataset}-images-idx3-ubyte",
            "--labels", f"data/mnist/{dataset}-labels-idx1-ubyte",
            "--epochs", "1",
            "-o", temp_out,
            "--recover", base_network
        ]).decode("utf-8")
        fail = 0
    except:
        fail = 1

    # 1e époque testing
    test_output = subprocess.check_output([
        binary,
        'test',
        '--modele', temp_out,
        '-d', 'mnist',
        '-i', 'data/mnist/t10k-images-idx3-ubyte',
        '-l', "data/mnist/t10k-labels-idx1-ubyte"
    ]).decode("utf-8")

    results.append({
        "train": {
            "accuracy": float(train_output.split("Accuracy: \x1b[32m")[-1].split("%")[0]), # \x1b[32m est la couleur verte pour le terminal
            "loss": float(train_output.split("Loss: ")[-1].split("\t")[0])
        },
        "test": {
            "accuracy": float(test_output.split("Taux de réussite: ")[-1].split("%")[0]),
            "loss": float(test_output.split("Loss: ")[-1])
        }
    })

    for i in progress(range(tries-1)):
        # On ne continue pas si on a déjà eu une saturation du réseau, on ajoute juste des valeurs
        if fail == 0:
            # i-ème époque training
            try:
                train_output = subprocess.check_output([
                    binary,
                    "train",
                    "--dataset", "mnist",
                    "--images", f"data/mnist/{dataset}-images-idx3-ubyte",
                    "--labels", f"data/mnist/{dataset}-labels-idx1-ubyte",
                    "--epochs", "1",
                    "-o", temp_out,
                    "--recover", temp_out
                ]).decode("utf-8")
                fail = 0
            except:
                fail = 1

            # i-ème époque testing
            test_output = subprocess.check_output([
                binary,
                'test',
                '--modele', temp_out,
                '-d', 'mnist',
                '-i', 'data/mnist/t10k-images-idx3-ubyte',
                '-l', "data/mnist/t10k-labels-idx1-ubyte"
            ]).decode("utf-8")

            # Ajout des résultats
            results.append({
                "train": {
                    "accuracy": float(train_output.split("Accuracy: \x1b[32m")[-1].split("%")[0]), # \x1b[32m est la couleur verte pour le terminal
                    "loss": float(train_output.split("Loss: ")[-1].split("\t")[0])
                },
                "test": {
                    "accuracy": float(test_output.split("Taux de réussite: ")[-1].split("%")[0]),
                    "loss": float(test_output.split("Loss: ")[-1])
                }
            })
        else:
            # Le réseau a saturé
            results.append({
                "train": {
                    "accuracy": 0,
                    "loss": 0
                },
                "test": {
                    "accuracy": 0,
                    "loss": 0
                }
            })

    return results

def create_base_network(binary, file):
    os.system(f"{binary} train --dataset mnist --images data/mnist/train-images-idx3-ubyte --labels data/mnist/train-labels-idx1-ubyte --epochs 0 --out {file}")


"""
binaries: list of files
tries: number of epochs to train on
dataset: must be "train" or "t10k"
metric: "accuracy" or "loss"
"""
def compare_binaries(binaries, tries=3, dataset="train", metric="accuracy", values=None):
    if values is None:
        print(f"========== {len(binaries)} Fichiers chargés ==========")
        base_net = f".cache/basenet-{abs(hash(''.join(binaries)))}.bin"
        create_base_network(binaries[0], base_net)
        results = {}
        for i in range(len(binaries)):
            binary = binaries[i]
            print(f"========== Benchmark de {binary} ({1+i}/{len(binaries)}) ==========")
            try:
                results[binaries[i]] = (train(binary, base_net, tries, dataset=dataset))
            except Exception as e:
                print(f"========== Erreur sur {binary} ==========")
                print(e)
                # Delete value if nothing happened
                # results.append([{'train': {'accuracy': 0., 'loss': 0.}, 'test': {'accuracy': 0., 'loss': 0.}}]*tries)
    else:
        results = values
        binaries = [key for key in values.keys()]

    x = [i+1 for i in range(tries)]

    fig, ax = plt.subplots()
    
    for binary in results.keys():
        for key in results[binary][0].keys():
            key_values = [j[key][metric] for j in results[binary]]

            courbe = ax.plot(x, key_values)[0]
            courbe.set_label(f"{key}/{binary}")

    
    ax.set_ylabel(f"{metric}")
    ax.set_xlabel("Nombre d'époques")

    ax.legend()

    # plt.ylim(0, 100)
    plt.show()
    return results
    