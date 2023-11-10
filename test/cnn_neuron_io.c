#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/test.h"

#include "../src/common/include/colors.h"
#include "../src/cnn/include/neuron_io.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/models.h"
#include "../src/cnn/include/utils.h"
#include "../src/cnn/include/free.h"


int main() {
    _TEST_PRESENTATION("CNN Lecture/Écriture")

    Network* network = create_network_lenet5(0, 0, 3, GLOROT, 32, 1, 2); // Pas besoin d'initialiser toute la backprop
    _TEST_ASSERT(true, "Création du réseau");

    write_network((char*)".test-cache/cnn_neuron_io.bin", network);
    _TEST_ASSERT(true, "Écriture du réseau");

    Network* network2 = read_network((char*)".test-cache/cnn_neuron_io.bin");
    Network* network3 = read_network((char*)".test-cache/cnn_neuron_io.bin");
    _TEST_ASSERT(true, "Vérification de l'accès en lecture");

    _TEST_ASSERT(equals_networks(network, network2), "Égalité des réseaux");
    _TEST_ASSERT(equals_networks(network2, network3), "Égalité de deux lectures");

    free_network(network);
    free_network(network2);
    free_network(network3);
    _TEST_ASSERT(true, "Libération de la mémoire");

    return 0;
}