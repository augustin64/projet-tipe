#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/include/colors.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/utils.h"
#include "../src/cnn/include/free.h"
#include "../src/include/colors.h"


int main() {
    Kernel* kernel;
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 0, 3, 2, 32, 1);
    printf(GREEN "OK\n" RESET);

    printf("Architecture LeNet5:\n");
    for (int i=0; i < network->size; i++) {
        kernel = network->kernel[i];
        if ((!kernel->cnn)&&(!kernel->nn)) {
            printf("\n==== Couche %d de type "YELLOW"Pooling"RESET" ====\n", i);
            printf("Linéarisation: %d\n", kernel->linearisation);
        } else if (!kernel->cnn) {
            printf("\n==== Couche %d de type "GREEN"NN"RESET" ====\n", i);
            printf("input: %d\n", kernel->nn->input_units);
            printf("output: %d\n", kernel->nn->output_units);
        } else {
            printf("\n==== Couche %d de type "BLUE"CNN"RESET" ====\n", i);
            printf("k_size: %d\n", kernel->cnn->k_size);
            printf("rows: %d\n", kernel->cnn->rows);
            printf("columns: %d\n", kernel->cnn->columns);
        }
        printf("width: %d\n", network->width[i]);
        printf("depth: %d\n", network->depth[i]);
    }
    printf(GREEN "\nOK\n" RESET);

    printf("Libération de la mémoire\n");
    free_network(network);
    printf(GREEN "OK\n" RESET);

    return 0;
}