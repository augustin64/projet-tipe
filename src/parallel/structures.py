#!/usr/bin/python3
"""
Description des structures.
"""
import os
import time
import subprocess


class NoMoreJobAvailableError(Exception):
    """Entraînement du réseau fini"""
    pass


class TryLaterError(Exception):
    """Batch fini, réessayer plus tard"""
    pass


class Client:
    """
    Description d'un client se connectant au serveur
    """
    def __init__(self, performance, token):
        self.performance = performance
        self.token = token
        self.start = 0
        self.nb_images = 0


    def get_job(self, training):
        """
        Donne un travail au client
        """
        if training.nb_images == training.cur_image:
            if training.batchs == training.cur_batch:
                raise NoMoreJobAvailableError
            raise TryLaterError

        self.start = training.cur_image
        self.nb_images = min(training.nb_images - training.cur_image, self.performance)
        training.cur_image += self.nb_images


clients = {}


class Training:
    """
    Classe training
    """
    def __init__(self, batchs, dataset, test_set, cache):
        # Définition de variables
        self.batchs = batchs
        self.cur_batch = 1
        self.cur_image = 0
        self.dataset = dataset
        self.test_set = test_set
        self.cache = cache
        self.reseau = os.path.join(self.cache, "reseau.bin")
        self.delta = os.path.join(self.cache, "delta.bin")

        # Définition des chemins et données relatives à chaque set de données
        # TODO: implémenter plus proprement avec un dictionnaire ou même un fichier datasets.json
        if self.dataset == "mnist-train":
            self.nb_images = 60000
        elif self.dataset == "mnist-t10k":
            self.nb_images = 10000
        else:
            raise NotImplementedError

        if self.test_set == "mnist-train":
            self.test_images = "data/mnist/train-images-idx3-ubyte"
            self.test_labels = "data/mnist/train-labels-idx1-ubyte"
        elif self.test_set == "mnist-t10k":
            self.test_images = "data/mnist/t10k-images-idx3-ubyte"
            self.test_labels = "data/mnist/t10k-labels-idx1-ubyte"
        else:
            print(f"{self.test_set} test dataset unknown.")
            raise NotImplementedError

        # On supprime le fichier de lock qui permet de
        # ne pas écrire en même temps plusieurs fois sur le fichier reseau.bin
        if os.path.isfile(self.reseau + ".lock"):
            os.remove(self.reseau + ".lock")


    def test_network(self):
        """
        Teste les performances du réseau avant le batch suivant
        """
        if not os.path.isfile("out/main"):
            subprocess.call(["make.sh", "main"])

        subprocess.call(
        [
            "out/main", "test",
            "--images", self.test_images,
            "--labels", self.test_labels,
            "--modele", self.reseau
        ])
        self.cur_batch += 1
        self.cur_image = 0


    def patch(self):
        """
        Applique un patch au réseau
        """
        # On attend que le lock se libère puis on patch le réseau
        while self.is_patch_locked():
            time.sleep(0.1)

        with open(self.reseau + ".lock", "w", encoding="utf8") as file:
            file.write("")

        if not os.path.isfile("out/main"):
            subprocess.call(["make.sh", "utils"])
        subprocess.call
        ([
            "out/utils", "patch-network",
            "--network", self.reseau,
            "--delta", self.delta,
        ])

        os.remove(self.reseau + ".lock")


    def is_patch_locked(self):
        """
        Petit raccourci pour vérifier si le lock est présent
        """
        return os.path.isfile(self.reseau + ".lock")
