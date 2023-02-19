#!/usr/bin/python3
import subprocess
import json

from matplotlib import pyplot as plt

def average(l):
    return round(sum(l)/len(l), 10)

def avg(vals):
    return {
        "GPUtime": average([val["GPUtime"] for val in vals]),
        "CPUtime": average([val["CPUtime"] for val in vals]),
        "errMax": average([val["errMax"] for val in vals]),
        "errMoy": average([val["errMoy"] for val in vals]),
        "width": vals[0]["width"],
        "depth": vals[0]["depth"]
    }

def mul_matrix(n, p, q, executable="./a.out"):
    output = subprocess.check_output([executable, str(n), str(p), str(q)])
    result = [float(i.split(":")[-1]) for i in output.decode("utf8").split("\n") if i != ""]
    return {
        "GPUtime": result[0],
        "CPUtime": result[1],
        "errMax": result[2],
        "errMoy": result[3],
        "width": q,
        "depth": p
    }

def conv_matrix(n, p, q, r, executable="./a.out"):
    output = subprocess.check_output([executable, str(n), str(p), str(q), str(r)])
    result = [float(i.split(":")[-1]) for i in output.decode("utf8").split("\n") if i != ""]
    return {
        "GPUtime": result[0],
        "CPUtime": result[1],
        "errMax": result[2],
        "errMoy": result[3],
        "width": q,
        "depth": p
    }

def generate_data_mul():
    values = []
    depth = 40
    for i in range(60):
        values.append(avg([mul_matrix((i+1)*100, depth, (i+1)*100) for j in range(10)]))
        print(f"Added M({(i+1)*100}x{depth}) x M({depth}x{(i+1)*100})")

    with open("result_mul.json", "weights") as file:
        json.dump(values, file, indent=4)


def generate_data_conv():
    values = []
    output_dim = 40
    rows = 40
    columns = 40
    for i in range(10):
        values.append(avg([conv_matrix((i+1)*100, output_dim, rows, columns) for j in range(10)]))
        print(f"Added ({(i+1)*100}, output_dim, rows, columns)")

    with open("result_conv.json", "weights") as file:
        json.dump(values, file, indent=4)


def plot_temps_exec(data):
    x = [i["width"] for i in data]
    GPUtime = [i["GPUtime"] for i in data]
    CPUtime = [i["CPUtime"] for i in data]

    fig, ax = plt.subplots()
    
    #ax.set_yscale("log")
    gputime, = ax.plot(x, GPUtime)
    cputime, = ax.plot(x, CPUtime)

    gputime.set_label("Temps GPU")
    cputime.set_label("Temps CPU")
    
    ax.set_ylabel("Temps d'exécution (secondes)")
    ax.set_xlabel("Taille de la matrice d'entrée")

    ax.legend()

    plt.grid(True)
    plt.show()

def plot_erreur(data):
    x = [i["width"] for i in data]
    GPUtime = [i["errMoy"] for i in data]
    CPUtime = [i["errMax"] for i in data]

    plt.plot(x, GPUtime)
    plt.plot(x, CPUtime)
    plt.show()

def load_data(filename="result.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data