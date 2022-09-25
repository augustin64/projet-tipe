#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/cnn/neuron_io.c"
#include "../src/cnn/creation.c"



int main() {
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 3, 2);
    printf("OK\n");

    printf("Écriture du réseau\n");
    write_network(".test-cache/cnn_neuron_io.bin", network);
    printf("OK\n");

    printf("Vérification de l'accès en lecture\n");
    Network* network2 = read_network(".test-cache/cnn_neuron_io.bin");
    printf("OK\n");

    /*
    printf("Réécriture du nouveau réseau\n");
    write_network(".test-cache/cnn_neuron_io_2.bin", network2);
    printf("OK\n");
    */
    return 0;
}