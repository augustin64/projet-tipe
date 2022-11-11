#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/mnist/include/neural_network.h"
#include "../src/mnist/include/neuron_io.h"
#include "../src/include/colors.h"

int main() {
    printf("Création du réseau\n");
    Network* network = (Network*)malloc(sizeof(Network));
    int tab[5] = {30, 25, 20, 15, 10};
    network_creation(network, tab, 5);
    printf(GREEN "OK\n" RESET);

    printf("Initialisation du réseau\n");
    network_initialisation(network);
    printf(GREEN "OK\n" RESET);

    deletion_of_network(network);
    return 0;
}