#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "../include/memory_management.h"
#include "../include/colors.h"
#include "include/struct.h"

#define copyVar(var) network_cp->var = network->var
#define copyVarParams(var) network_dest->var = network_src->var

#define checkEquals(var, name, indice)                                                     \
if (network1->var != network2->var) {                                                      \
    printf_error((char*)"network1->" name " et network2->" name " ne sont pas égaux\n");   \
    if (indice != -1) {                                                                    \
        printf(BOLDBLUE "[ INFO_ ]" RESET " indice: %d\n", indice);                        \
        }                                                                                  \
    return false;                                                                          \
}

void swap(int* tab, int i, int j) {
    int tmp = tab[i];
    tab[i] = tab[j];
    tab[j] = tmp;
}

void knuth_shuffle(int* tab, int n) {
    for(int i=1; i < n; i++) {
        swap(tab, i, rand() %i);
    }
}

bool equals_networks(Network* network1, Network* network2) {
    checkEquals(size, "size", -1);
    checkEquals(initialisation, "initialisation", -1);
    checkEquals(dropout, "dropout", -1);

    for (int i=0; i < network1->size; i++) {
        checkEquals(width[i], "input_width", i);
        checkEquals(depth[i], "input_depth", i);
    }

    for (int i=0; i < network1->size-1; i++) {
        checkEquals(kernel[i]->activation, "kernel[i]->activation", i);
        if ((!network1->kernel[i]->cnn ^ !network2->kernel[i]->cnn) || (!network1->kernel[i]->nn ^ !network2->kernel[i]->nn)) {
            printf(BOLDRED "[ ERROR ]" RESET "network1->kernel[%d] et network1->kernel[%d] diffèrent de type\n", i, i);
            return false;
        }
        checkEquals(kernel[i]->linearisation, "kernel[i]->linearisation", i);
        if (!network1->kernel[i]->cnn && !network1->kernel[i]->nn) {
            // Type Pooling
            checkEquals(kernel[i]->activation, "kernel[i]->activation pour un pooling", i);
            checkEquals(kernel[i]->pooling, "kernel[i]->pooling pour un pooling", i);
        } else if (!network1->kernel[i]->cnn) {
            // Type NN
            checkEquals(kernel[i]->nn->size_input, "kernel[i]->nn->size_input", i);
            checkEquals(kernel[i]->nn->size_output, "kernel[i]->nn->size_output", i);
            for (int j=0; j < network1->kernel[i]->nn->size_output; j++) {
                checkEquals(kernel[i]->nn->bias[j], "kernel[i]->nn->bias[j]", j);
            }
            for (int j=0; j < network1->kernel[i]->nn->size_input; j++) {
                for (int k=0; k < network1->kernel[i]->nn->size_output; k++) {
                    checkEquals(kernel[i]->nn->weights[j][k], "kernel[i]->nn->weights[j][k]", k);
                }
            }
        } else {
            // Type CNN
            checkEquals(kernel[i]->cnn->k_size, "kernel[i]->k_size", i);
            checkEquals(kernel[i]->cnn->rows, "kernel[i]->rows", i);
            checkEquals(kernel[i]->cnn->columns, "kernel[i]->columns", i);
            for (int j=0; j < network1->kernel[i]->cnn->columns; j++) {
                checkEquals(kernel[i]->cnn->bias[j], "kernel[i]->cnn->bias[j]", j);
            }
            for (int j=0; j < network1->kernel[i]->cnn->rows; j++) {
                for (int k=0; k < network1->kernel[i]->cnn->columns; k++) {
                    for (int l=0; l < network1->kernel[i]->cnn->k_size; l++) {
                        for (int m=0; m < network1->kernel[i]->cnn->k_size; m++) {
                            checkEquals(kernel[i]->cnn->weights[j][k][l][m], "kernel[i]->cnn->weights[j][k][l][m]", m);
                        }
                    }
                }
            }
        }
    }

    return true;
}


Network* copy_network(Network* network) {
    Network* network_cp = (Network*)nalloc(1, sizeof(Network));
    // Paramètre du réseau
    int size = network->size;
    // Paramètres des couches NN
    int size_input;
    int size_output;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;

    copyVar(dropout);
    copyVar(learning_rate);
    copyVar(initialisation);
    copyVar(max_size);
    copyVar(size);

    network_cp->width = (int*)nalloc(size, sizeof(int));
    network_cp->depth = (int*)nalloc(size, sizeof(int));

    for (int i=0; i < size; i++) {
        copyVar(width[i]);
        copyVar(depth[i]);
    }

    network_cp->kernel = (Kernel**)nalloc(size-1, sizeof(Kernel*));
    for (int i=0; i < size-1; i++) {
        network_cp->kernel[i] = (Kernel*)nalloc(1, sizeof(Kernel));
        if (!network->kernel[i]->nn && !network->kernel[i]->cnn) { // Cas de la couche de linéarisation
            copyVar(kernel[i]->pooling);
            copyVar(kernel[i]->activation);
            copyVar(kernel[i]->linearisation); // 1
            network_cp->kernel[i]->cnn = NULL;
            network_cp->kernel[i]->nn = NULL;
        }
        else if (!network->kernel[i]->cnn) { // Cas du NN
            copyVar(kernel[i]->pooling);
            copyVar(kernel[i]->activation);
            copyVar(kernel[i]->linearisation); // 0

            size_input = network->kernel[i]->nn->size_input;
            size_output = network->kernel[i]->nn->size_output;

            network_cp->kernel[i]->cnn = NULL;
            network_cp->kernel[i]->nn = (Kernel_nn*)nalloc(1, sizeof(Kernel_nn));

            copyVar(kernel[i]->nn->size_input);
            copyVar(kernel[i]->nn->size_output);

            network_cp->kernel[i]->nn->bias = (float*)nalloc(size_output, sizeof(float));
            network_cp->kernel[i]->nn->d_bias = (float*)nalloc(size_output, sizeof(float));
            for (int j=0; j < size_output; j++) {
                copyVar(kernel[i]->nn->bias[j]);
                network_cp->kernel[i]->nn->d_bias[j] = 0.;
            }

            network_cp->kernel[i]->nn->weights = (float**)nalloc(size_input, sizeof(float*));
            network_cp->kernel[i]->nn->d_weights = (float**)nalloc(size_input, sizeof(float*));
            for (int j=0; j < size_input; j++) {
                network_cp->kernel[i]->nn->weights[j] = (float*)nalloc(size_output, sizeof(float));
                network_cp->kernel[i]->nn->d_weights[j] = (float*)nalloc(size_output, sizeof(float));
                for (int k=0; k < size_output; k++) {
                    copyVar(kernel[i]->nn->weights[j][k]);
                    network_cp->kernel[i]->nn->d_weights[j][k] = 0.;
                }
            }
        }
        else { // Cas du CNN
            copyVar(kernel[i]->pooling);
            copyVar(kernel[i]->activation);
            copyVar(kernel[i]->linearisation); // 0

            rows = network->kernel[i]->cnn->rows;
            k_size = network->kernel[i]->cnn->k_size;
            columns = network->kernel[i]->cnn->columns;

            network_cp->kernel[i]->nn = NULL;
            network_cp->kernel[i]->cnn = (Kernel_cnn*)nalloc(1, sizeof(Kernel_cnn));

            copyVar(kernel[i]->cnn->rows);
            copyVar(kernel[i]->cnn->k_size);
            copyVar(kernel[i]->cnn->columns);

            network_cp->kernel[i]->cnn->bias = (float*)nalloc(columns, sizeof(float));
            network_cp->kernel[i]->cnn->d_bias = (float*)nalloc(columns, sizeof(float));
            for (int j=0; j < columns; j++) {
                copyVar(kernel[i]->cnn->bias[j]);
                network_cp->kernel[i]->cnn->d_bias[j] = 0.;
            }

            network_cp->kernel[i]->cnn->weights = (float****)nalloc(rows, sizeof(float***));
            network_cp->kernel[i]->cnn->d_weights = (float****)nalloc(rows, sizeof(float***));
            for (int j=0; j < rows; j++) {
                network_cp->kernel[i]->cnn->weights[j] = (float***)nalloc(columns, sizeof(float**));
                network_cp->kernel[i]->cnn->d_weights[j] = (float***)nalloc(columns, sizeof(float**));
                for (int k=0; k < columns; k++) {
                    network_cp->kernel[i]->cnn->weights[j][k] = (float**)nalloc(k_size, sizeof(float*));
                    network_cp->kernel[i]->cnn->d_weights[j][k] = (float**)nalloc(k_size, sizeof(float*));
                    for (int l=0; l < k_size; l++) {
                        network_cp->kernel[i]->cnn->weights[j][k][l] = (float*)nalloc(k_size, sizeof(float));
                        network_cp->kernel[i]->cnn->d_weights[j][k][l] = (float*)nalloc(k_size, sizeof(float));
                        for (int m=0; m < k_size; m++) {
                            copyVar(kernel[i]->cnn->weights[j][k][l][m]);
                            network_cp->kernel[i]->cnn->d_weights[j][k][l][m] = 0.;
                        }
                    }
                }
            }
        }
    }

    network_cp->input = (float****)nalloc(size, sizeof(float***));
    for (int i=0; i < size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network_cp->input[i] = (float***)nalloc(network->depth[i], sizeof(float**));
        for (int j=0; j < network->depth[i]; j++) {
            network_cp->input[i][j] = (float**)nalloc(network->width[i], sizeof(float*));
            for (int k=0; k < network->width[i]; k++) {
                network_cp->input[i][j][k] = (float*)nalloc(network->width[i], sizeof(float));
                for (int l=0; l < network->width[i]; l++) {
                    network_cp->input[i][j][k][l] = 0.;
                }
            }
        }
    }

    network_cp->input_z = (float****)nalloc(size, sizeof(float***));
    for (int i=0; i < size; i++) { // input_z[size][couche->depth][couche->dim][couche->dim]
        network_cp->input_z[i] = (float***)nalloc(network->depth[i], sizeof(float**));
        for (int j=0; j < network->depth[i]; j++) {
            network_cp->input_z[i][j] = (float**)nalloc(network->width[i], sizeof(float*));
            for (int k=0; k < network->width[i]; k++) {
                network_cp->input_z[i][j][k] = (float*)nalloc(network->width[i], sizeof(float));
                for (int l=0; l < network->width[i]; l++) {
                    network_cp->input_z[i][j][k][l] = 0.;
                }
            }
        }
    }

    return network_cp;
}


void copy_network_parameters(Network* network_src, Network* network_dest) {
    // Paramètre du réseau
    int size = network_src->size;
    // Paramètres des couches NN
    int size_input;
    int size_output;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;

    copyVarParams(learning_rate);

    for (int i=0; i < size-1; i++) {
        if (!network_src->kernel[i]->cnn && network_src->kernel[i]->nn) { // Cas du NN

            size_input = network_src->kernel[i]->nn->size_input;
            size_output = network_src->kernel[i]->nn->size_output;

            for (int j=0; j < size_output; j++) {
                copyVarParams(kernel[i]->nn->bias[j]);
            }
            for (int j=0; j < size_input; j++) {
                for (int k=0; k < size_output; k++) {
                    copyVarParams(kernel[i]->nn->weights[j][k]);
                }
            }
        }
        else if (network_src->kernel[i]->cnn && !network_src->kernel[i]->nn) { // Cas du CNN

            rows = network_src->kernel[i]->cnn->rows;
            k_size = network_src->kernel[i]->cnn->k_size;
            columns = network_src->kernel[i]->cnn->columns;

            for (int j=0; j < columns; j++) {
                copyVarParams(kernel[i]->cnn->bias[j]);
            }
            for (int j=0; j < rows; j++) {
                for (int k=0; k < columns; k++) {
                    for (int l=0; l < k_size; l++) {
                        for (int m=0; m < k_size; m++) {
                            copyVarParams(kernel[i]->cnn->weights[j][k][l][m]);
                        }
                    }
                }
            }
        }
    }
}


int count_null_weights(Network* network) {
    float epsilon = 0.000001;

    int null_weights = 0;
    int null_bias = 0;

    int size = network->size;
    // Paramètres des couches NN
    int size_input;
    int size_output;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;

    for (int i=0; i < size-1; i++) {
        if (!network->kernel[i]->cnn && network->kernel[i]->nn) { // Cas du NN

            size_input = network->kernel[i]->nn->size_input;
            size_output = network->kernel[i]->nn->size_output;

            for (int j=0; j < size_output; j++) {
                null_bias += fabs(network->kernel[i]->nn->bias[j]) <= epsilon;
            }
            for (int j=0; j < size_input; j++) {
                for (int k=0; k < size_output; k++) {
                    null_weights += fabs(network->kernel[i]->nn->weights[j][k]) <= epsilon;
                }
            }
        }
        else if (network->kernel[i]->cnn && !network->kernel[i]->nn) { // Cas du CNN

            rows = network->kernel[i]->cnn->rows;
            k_size = network->kernel[i]->cnn->k_size;
            columns = network->kernel[i]->cnn->columns;

            for (int j=0; j < columns; j++) {
                null_bias += fabs(network->kernel[i]->cnn->bias[j]) <= epsilon;
            }
            for (int j=0; j < rows; j++) {
                for (int k=0; k < columns; k++) {
                    for (int l=0; l < k_size; l++) {
                        for (int m=0; m < k_size; m++) {
                            null_weights = fabs(network->kernel[i]->cnn->weights[j][k][l][m]) <= epsilon;
                        }
                    }
                }
            }
        }
    }

    (void)null_bias;
    return null_weights;
}