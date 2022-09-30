#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/mnist/include/neural_network.h"
#include "../src/mnist/include/neuron_io.h"

int main() {
    printf("Création du réseau\n");
    Network* network = (Network*)malloc(sizeof(Network));
    int tab[5] = {30, 25, 20, 15, 10};
    network_creation(network, tab, 5);
    printf("OK\n");

    printf("Initialisation du réseau\n");
    network_initialisation(network);
    printf("OK\n");

    deletion_of_network(network);
    return 0;
}