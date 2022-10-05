#include <stdlib.h>
#include <stdio.h>

#include "../src/colors.h"
#include "../src/cnn/creation.c"
#include "../src/cnn/utils.c"

int main() {
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 0, 3, 2, 32, 1);
    printf("OK\n");

    printf("Copie du réseau\n");
    Network* network_cp = copy_network(network);
    printf("OK\n");

    printf("Vérification de l'égalité des réseaux\n");
    if (! equals_networks(network, network_cp)) {
        printf_error("Les deux réseaux obtenus ne sont pas égaux.\n");
        exit(1);
    }
    printf("OK\n");

    return 0;
}