#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "neural_network.c"
#include "neuron_io.c"

void print_biais(char* filename) {
    Reseau* reseau = lire_reseau(".cache/reseau.bin");

    for (int i=1; i < reseau->nb_couches -1; i++) {
        printf("Couche %d\n", i);
        for (int j=0; j < reseau->couches[i]->nb_neurones; j++) {
            printf("Couche %d\tNeurone %d\tBiais: %0.1f\n", i, j, reseau->couches[i]->neurones[j]->biais);
        }
    }
}

int main(int argc, char* argv[]) {
    print_biais(".cache/reseau.bin");
    return 1;
}