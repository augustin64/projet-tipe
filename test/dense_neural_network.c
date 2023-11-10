#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/test.h"

#include "../src/dense/include/neural_network.h"
#include "../src/dense/include/neuron_io.h"
#include "../src/common/include/colors.h"

int main() {
    _TEST_PRESENTATION("Dense: Création")

    Network* network = (Network*)malloc(sizeof(Network));
    int tab[5] = {30, 25, 20, 15, 10};
    network_creation(network, tab, 5);
    _TEST_ASSERT(true, "Création");

    network_initialisation(network);
    _TEST_ASSERT(true, "Initialisation");

    deletion_of_network(network);
    _TEST_ASSERT(true, "Suppression");
    return 0;
}