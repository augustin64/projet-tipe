#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/mnist/neural_network.c"
#include "../src/mnist/neuron_io.c"

int main() {
    printf("Création du réseau\n");

    Reseau* reseau_neuronal = malloc(sizeof(Reseau));
    int tab[5] = {30, 25, 20, 15, 10};
    creation_du_reseau_neuronal(reseau_neuronal, tab, 5);

    printf("OK\n");
    printf("Initialisation du réseau\n");

    initialisation_du_reseau_neuronal(reseau_neuronal);

    printf("OK\n");
    printf("Enregistrement du réseau\n");

    ecrire_reseau("/tmp/reseau_test.bin", reseau_neuronal);

    printf("OK\n");
    return 1;
}