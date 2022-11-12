#include <stdlib.h>
#include <stdio.h>

#include "../src/include/colors.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/utils.h"
#include "../src/include/colors.h"

int main() {
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 0, 3, 2, 32, 1);
    printf(GREEN "OK\n" RESET);

    printf("Copie du réseau\n");
    Network* network_cp = copy_network(network);
    printf(GREEN "OK\n" RESET);

    printf("Vérification de l'égalité des réseaux\n");
    if (! equals_networks(network, network_cp)) {
        printf_error(RED "Les deux réseaux obtenus ne sont pas égaux.\n" RESET);
        exit(1);
    }
    printf(GREEN "OK\n" RESET);

    return 0;
}