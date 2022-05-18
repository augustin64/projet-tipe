#!/usr/bin/python3
"""
Description des clients se connectant au serveur.
"""

class Client():
    """
    Classe client
    """
    def __init__(self, performance, token):
        self.performance = performance
        self.token = token


clients = []
