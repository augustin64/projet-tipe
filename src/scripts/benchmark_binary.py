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

def train(binary, base_network, tries=3, dataset="train"):
    temp_out = f".cache/tmp-{abs(hash(binary))}.bin"
    results = []
    fail = os.system(f"{binary} train --dataset mnist --images data/mnist/{dataset}-images-idx3-ubyte --labels data/mnist/{dataset}-labels-idx1-ubyte --epochs 1 -o {temp_out} --recover {base_network}")
    output = subprocess.check_output([
        binary,
        'test',
        '--modele', temp_out,
        '-d', 'mnist',
        '-i', 'data/mnist/t10k-images-idx3-ubyte',
        '-l', "data/mnist/t10k-labels-idx1-ubyte"
    ]).decode("utf-8")
    results.append(float(output.split('\r')[-1].split(" ")[-1].split("%")[0]))
    for i in range(tries-1):
        if fail == 0:
            fail = os.system(f"{binary} train --dataset mnist --images data/mnist/{dataset}-images-idx3-ubyte --labels data/mnist/{dataset}-labels-idx1-ubyte --epochs 1 --out {temp_out} --recover {temp_out}")
            output = subprocess.check_output([
                binary,
                'test',
                '--modele', temp_out,
                '-d', 'mnist',
                '-i', 'data/mnist/t10k-images-idx3-ubyte',
                '-l', "data/mnist/t10k-labels-idx1-ubyte"
            ]).decode("utf-8")
            results.append(float(output.split('\r')[-1].split(" ")[-1].split("%")[0]))
        else:
            results.append(results[-1])

    return results

def create_base_network(binary, file):
    os.system(f"{binary} train --dataset mnist --images data/mnist/train-images-idx3-ubyte --labels data/mnist/train-labels-idx1-ubyte --epochs 0 --out {file}")


def compare_binaries(binaries, tries=3, dataset="train"):
    print(f"========== {len(binaries)} Fichiers chargés ==========")
    base_net = f".cache/basenet-{abs(hash(''.join(binaries)))}.bin"
    create_base_network(binaries[0], base_net)
    results = []
    for i in range(len(binaries)):
        binary = binaries[i]
        print(f"========== Benchmmark de {binary} ({1+i}/{len(binaries)}) ==========")
        try:
            results.append(train(binary, base_net, tries, dataset=dataset))
        except:
            print(f"========== Erreur sur {binary} ==========")
            results.append(0)

    x = [i for i in range(tries)]

    fig, ax = plt.subplots()
    
    res = []
    for i in range(len(binaries)):
        if results[i] != 0:
            res.append(ax.plot(x, results[i])[0])
            res[i].set_label(binaries[i])
    
    ax.set_ylabel("Taux de réussite (%)")
    ax.set_xlabel("Nombre de batchs")

    ax.legend()

    plt.ylim(0, 100)
    plt.show()
    return results
    