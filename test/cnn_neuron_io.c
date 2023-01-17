#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/include/colors.h"
#include "../src/cnn/include/neuron_io.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/utils.h"
#include "../src/cnn/include/free.h"
#include "../src/include/colors.h"


int main() {
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 0, 3, GLOROT, 32, 1);
    printf(GREEN "OK\n" RESET);

    printf("Écriture du réseau\n");
    write_network((char*)".test-cache/cnn_neuron_io.bin", network);
    printf(GREEN "OK\n" RESET);

    printf("Vérification de l'accès en lecture\n");
    Network* network2 = read_network((char*)".test-cache/cnn_neuron_io.bin");
    Network* network3 = read_network((char*)".test-cache/cnn_neuron_io.bin");
    printf(GREEN "OK\n" RESET);

    printf("Vérification de l'égalité des réseaux\n");
    if (! equals_networks(network, network2)) {
        printf_error(RED "Le réseau lu ne contient pas les mêmes données.\n" RESET);
        exit(1);
    }
    if (! equals_networks(network2, network3)) {
        printf_error(RED "La lecture du réseau donne des résultats différents.\n" RESET);
        exit(1);
    }
    printf(GREEN "OK\n" RESET);

    printf("Libération de la mémoire\n");
    free_network(network);
    free_network(network2);
    free_network(network3);
    printf(GREEN "OK\n" RESET);

    return 0;
}