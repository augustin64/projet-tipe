#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/common/include/colors.h"
#include "../src/cnn/include/creation.h"
#include "../src/cnn/include/utils.h"
#include "../src/cnn/include/free.h"


int main() {
    Kernel* kernel;
    printf("Création du réseau\n");
    Network* network = create_network_lenet5(0, 0, 3, 2, 32, 1);
    printf(GREEN "OK\n" RESET);

    printf("Architecture LeNet5:\n");
    for (int i=0; i < network->size-1; i++) {
        kernel = network->kernel[i];
        if ((!kernel->cnn)&&(!kernel->nn)) {
            if (kernel->pooling == AVG_POOLING) {
                printf("\n==== Couche %d de type "YELLOW"Average Pooling"RESET" ====\n", i);
            } else {
                printf("\n==== Couche %d de type "YELLOW"Max Pooling"RESET" ====\n", i);
            }
            int kernel_size = 2*kernel->padding + network->width[i] + kernel->stride - network->width[i+1]*kernel->stride;
            printf("kernel: %dx%d, pad=%d, stride=%d\n", kernel_size, kernel_size, kernel->padding, kernel->stride);
        } else if (!kernel->cnn) {
            printf("\n==== Couche %d de type "GREEN"NN"RESET" ====\n", i);
            if (kernel->linearisation) {
                printf(YELLOW"Linéarisation: %d\n"RESET, kernel->linearisation);
            }
            printf("input: %d\n", kernel->nn->size_input);
            printf("output: %d\n", kernel->nn->size_output);
        } else {
            printf("\n==== Couche %d de type "BLUE"CNN"RESET" ====\n", i);
            printf("kernel: %dx%d, pad=%d, stride=%d\n", kernel->cnn->k_size, kernel->cnn->k_size, kernel->padding, kernel->stride);
            printf("%d kernels\n", kernel->cnn->columns);
        }
        if (!kernel->nn) {
            printf("depth: %d\n", network->depth[i]);
            printf("width: %d\n", network->width[i]);
        }
        if (kernel->nn || kernel->cnn) {
            printf("activation: %d\n", kernel->activation);
        }
    }
    printf("\n" GREEN "OK\n" RESET);

    printf("Libération de la mémoire\n");
    free_network(network);
    printf(GREEN "OK\n" RESET);

    return 0;
}