#include <stdlib.h>
#include <stdio.h>

#include "include/test.h"

#include "../src/common/include/colors.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/models.h"
#include "../src/cnn/include/utils.h"
#include "../src/cnn/include/free.h"

int main() {
    _TEST_PRESENTATION("Utilitaires du CNN");

    Network* network = create_network_lenet5(0, 0, 3, 2, 32, 1, 0);
    Network* network2 = create_network_lenet5(0, 0, 3, 2, 32, 1, 0);
    _TEST_ASSERT(true, "Création de réseaux");

    Network* network_cp = copy_network(network);
    _TEST_ASSERT(true, "Copie de réseau (copy_network)");

    _TEST_ASSERT(equals_networks(network, network_cp), "Égalité du réseau copié (copy_network)");

    copy_network_parameters(network, network2);
    _TEST_ASSERT(true, "Copie de réseau (copy_network_parameters)");

    _TEST_ASSERT(equals_networks(network, network2), "Égalité du réseau copié (copy_network_parameters)");

    free_network(network_cp);
    free_network(network2);
    free_network(network);
    return 0;
}