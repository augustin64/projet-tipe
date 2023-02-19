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

#define checkEquals(var, name, indice)                                              \
if (network1->var != network2->var) {                                               \
    printf_error("network1->" name " et network2->" name " ne sont pas égaux\n");   \
    if (indice != -1) {                                                             \
        printf(BOLDBLUE"[ INFO_ ]" RESET " indice: %d\n", indice);                  \
        }                                                                           \
    return false;                                                                   \
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
    int output_dim;
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
            checkEquals(kernel[i]->nn->input_units, "kernel[i]->nn->input_units", i);
            checkEquals(kernel[i]->nn->output_units, "kernel[i]->nn->output_units", i);
            for (int j=0; j < network1->kernel[i]->nn->output_units; j++) {
                checkEquals(kernel[i]->nn->bias[j], "kernel[i]->nn->bias[j]", j);
            }
            for (int j=0; j < network1->kernel[i]->nn->input_units; j++) {
                for (int k=0; k < network1->kernel[i]->nn->output_units; k++) {
                    checkEquals(kernel[i]->nn->weights[j][k], "kernel[i]->nn->weights[j][k]", k);
                }
            }
        } else {
            // Type CNN
            output_dim = network1->width[i+1];
            checkEquals(kernel[i]->cnn->k_size, "kernel[i]->k_size", i);
            checkEquals(kernel[i]->cnn->rows, "kernel[i]->rows", i);
            checkEquals(kernel[i]->cnn->columns, "kernel[i]->columns", i);
            for (int j=0; j < network1->kernel[i]->cnn->columns; j++) {
                for (int k=0; k < output_dim; k++) {
                    for (int l=0; l < output_dim; l++) {
                        checkEquals(kernel[i]->cnn->bias[j][k][l], "kernel[i]->cnn->bias[j][k][l]", l);
                    }
                }
            }
            for (int j=0; j < network1->kernel[i]->cnn->rows; j++) {
                for (int k=0; k < network1->kernel[i]->cnn->columns; k++) {
                    for (int l=0; l < network1->kernel[i]->cnn->k_size; l++) {
                        for (int m=0; m < network1->kernel[i]->cnn->k_size; m++) {
                            checkEquals(kernel[i]->cnn->w[j][k][l][m], "kernel[i]->cnn->bias[j][k][l][m]", m);
                        }
                    }
                }
            }
        }
    }

    return true;
}


Network* copy_network(Network* network) {
    Network* network_cp = (Network*)nalloc(sizeof(Network));
    // Paramètre du réseau
    int size = network->size;
    // Paramètres des couches NN
    int input_units;
    int output_units;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;
    int output_dim;

    copyVar(dropout);
    copyVar(learning_rate);
    copyVar(initialisation);
    copyVar(max_size);
    copyVar(size);

    network_cp->width = (int*)nalloc(sizeof(int)*size);
    network_cp->depth = (int*)nalloc(sizeof(int)*size);

    for (int i=0; i < size; i++) {
        copyVar(width[i]);
        copyVar(depth[i]);
    }

    network_cp->kernel = (Kernel**)nalloc(sizeof(Kernel*)*(size-1));
    for (int i=0; i < size-1; i++) {
        network_cp->kernel[i] = (Kernel*)nalloc(sizeof(Kernel));
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

            input_units = network->kernel[i]->nn->input_units;
            output_units = network->kernel[i]->nn->output_units;

            network_cp->kernel[i]->cnn = NULL;
            network_cp->kernel[i]->nn = (Kernel_nn*)nalloc(sizeof(Kernel_nn));

            copyVar(kernel[i]->nn->input_units);
            copyVar(kernel[i]->nn->output_units);

            network_cp->kernel[i]->nn->bias = (float*)nalloc(sizeof(float)*output_units);
            network_cp->kernel[i]->nn->d_bias = (float*)nalloc(sizeof(float)*output_units);
            for (int j=0; j < output_units; j++) {
                copyVar(kernel[i]->nn->bias[j]);
                network_cp->kernel[i]->nn->d_bias[j] = 0.;
            }

            network_cp->kernel[i]->nn->weights = (float**)nalloc(sizeof(float*)*input_units);
            network_cp->kernel[i]->nn->d_weights = (float**)nalloc(sizeof(float*)*input_units);
            for (int j=0; j < input_units; j++) {
                network_cp->kernel[i]->nn->weights[j] = (float*)nalloc(sizeof(float)*output_units);
                network_cp->kernel[i]->nn->d_weights[j] = (float*)nalloc(sizeof(float)*output_units);
                for (int k=0; k < output_units; k++) {
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
            output_dim = network->width[i+1];


            network_cp->kernel[i]->nn = NULL;
            network_cp->kernel[i]->cnn = (Kernel_cnn*)nalloc(sizeof(Kernel_cnn));

            copyVar(kernel[i]->cnn->rows);
            copyVar(kernel[i]->cnn->k_size);
            copyVar(kernel[i]->cnn->columns);

            network_cp->kernel[i]->cnn->bias = (float***)nalloc(sizeof(float**)*columns);
            network_cp->kernel[i]->cnn->d_bias = (float***)nalloc(sizeof(float**)*columns);
            for (int j=0; j < columns; j++) {
                network_cp->kernel[i]->cnn->bias[j] = (float**)nalloc(sizeof(float*)*output_dim);
                network_cp->kernel[i]->cnn->d_bias[j] = (float**)nalloc(sizeof(float*)*output_dim);
                for (int k=0; k < output_dim; k++) {
                    network_cp->kernel[i]->cnn->bias[j][k] = (float*)nalloc(sizeof(float)*output_dim);
                    network_cp->kernel[i]->cnn->d_bias[j][k] = (float*)nalloc(sizeof(float)*output_dim);
                    for (int l=0; l < output_dim; l++) {
                        copyVar(kernel[i]->cnn->bias[j][k][l]);
                        network_cp->kernel[i]->cnn->d_bias[j][k][l] = 0.;
                    }
                }
            }

            network_cp->kernel[i]->cnn->w = (float****)nalloc(sizeof(float***)*rows);
            network_cp->kernel[i]->cnn->d_w = (float****)nalloc(sizeof(float***)*rows);
            for (int j=0; j < rows; j++) {
                network_cp->kernel[i]->cnn->w[j] = (float***)nalloc(sizeof(float**)*columns);
                network_cp->kernel[i]->cnn->d_w[j] = (float***)nalloc(sizeof(float**)*columns);
                for (int k=0; k < columns; k++) {
                    network_cp->kernel[i]->cnn->w[j][k] = (float**)nalloc(sizeof(float*)*k_size);
                    network_cp->kernel[i]->cnn->d_w[j][k] = (float**)nalloc(sizeof(float*)*k_size);
                    for (int l=0; l < k_size; l++) {
                        network_cp->kernel[i]->cnn->w[j][k][l] = (float*)nalloc(sizeof(float)*k_size);
                        network_cp->kernel[i]->cnn->d_w[j][k][l] = (float*)nalloc(sizeof(float)*k_size);
                        for (int m=0; m < k_size; m++) {
                            copyVar(kernel[i]->cnn->w[j][k][l][m]);
                            network_cp->kernel[i]->cnn->d_w[j][k][l][m] = 0.;
                        }
                    }
                }
            }
        }
    }

    network_cp->input = (float****)nalloc(sizeof(float***)*size);
    for (int i=0; i < size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network_cp->input[i] = (float***)nalloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network_cp->input[i][j] = (float**)nalloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network_cp->input[i][j][k] = (float*)nalloc(sizeof(float)*network->width[i]);
                for (int l=0; l < network->width[i]; l++) {
                    network_cp->input[i][j][k][l] = 0.;
                }
            }
        }
    }

    network_cp->input_z = (float****)nalloc(sizeof(float***)*size);
    for (int i=0; i < size; i++) { // input_z[size][couche->depth][couche->dim][couche->dim]
        network_cp->input_z[i] = (float***)nalloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network_cp->input_z[i][j] = (float**)nalloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network_cp->input_z[i][j][k] = (float*)nalloc(sizeof(float)*network->width[i]);
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
    int input_units;
    int output_units;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;
    int output_dim;

    copyVarParams(learning_rate);

    for (int i=0; i < size-1; i++) {
        if (!network_src->kernel[i]->cnn && network_src->kernel[i]->nn) { // Cas du NN

            input_units = network_src->kernel[i]->nn->input_units;
            output_units = network_src->kernel[i]->nn->output_units;

            for (int j=0; j < output_units; j++) {
                copyVarParams(kernel[i]->nn->bias[j]);
            }
            for (int j=0; j < input_units; j++) {
                for (int k=0; k < output_units; k++) {
                    copyVarParams(kernel[i]->nn->weights[j][k]);
                }
            }
        }
        else if (network_src->kernel[i]->cnn && !network_src->kernel[i]->nn) { // Cas du CNN

            rows = network_src->kernel[i]->cnn->rows;
            k_size = network_src->kernel[i]->cnn->k_size;
            columns = network_src->kernel[i]->cnn->columns;
            output_dim = network_src->width[i+1];

            for (int j=0; j < columns; j++) {
                for (int k=0; k < output_dim; k++) {
                    for (int l=0; l < output_dim; l++) {
                        copyVarParams(kernel[i]->cnn->bias[j][k][l]);
                    }
                }
            }
            for (int j=0; j < rows; j++) {
                for (int k=0; k < columns; k++) {
                    for (int l=0; l < k_size; l++) {
                        for (int m=0; m < k_size; m++) {
                            copyVarParams(kernel[i]->cnn->w[j][k][l][m]);
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
    int input_units;
    int output_units;
    // Paramètres des couches CNN
    int rows;
    int k_size;
    int columns;
    int output_dim;

    for (int i=0; i < size-1; i++) {
        if (!network->kernel[i]->cnn && network->kernel[i]->nn) { // Cas du NN

            input_units = network->kernel[i]->nn->input_units;
            output_units = network->kernel[i]->nn->output_units;

            for (int j=0; j < output_units; j++) {
                null_bias += fabs(network->kernel[i]->nn->bias[j]) <= epsilon;
            }
            for (int j=0; j < input_units; j++) {
                for (int k=0; k < output_units; k++) {
                    null_weights += fabs(network->kernel[i]->nn->weights[j][k]) <= epsilon;
                }
            }
        }
        else if (network->kernel[i]->cnn && !network->kernel[i]->nn) { // Cas du CNN

            rows = network->kernel[i]->cnn->rows;
            k_size = network->kernel[i]->cnn->k_size;
            columns = network->kernel[i]->cnn->columns;
            output_dim = network->width[i+1];

            for (int j=0; j < columns; j++) {
                for (int k=0; k < output_dim; k++) {
                    for (int l=0; l < output_dim; l++) {
                        null_bias += fabs(network->kernel[i]->cnn->bias[j][k][l]) <= epsilon;
                    }
                }
            }
            for (int j=0; j < rows; j++) {
                for (int k=0; k < columns; k++) {
                    for (int l=0; l < k_size; l++) {
                        for (int m=0; m < k_size; m++) {
                            null_weights = fabs(network->kernel[i]->cnn->w[j][k][l][m]) <= epsilon;
                        }
                    }
                }
            }
        }
    }

    return null_weights;
}