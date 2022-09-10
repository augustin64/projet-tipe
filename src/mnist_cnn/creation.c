#include <stdio.h>
#include <stdlib.h>
#include "creation.h"
#include "function.h"
#include "initialisation.h"

Network* create_network(int max_size, int dropout, int initialisation, int input_dim, int input_depth) {
    if (dropout < 0 || dropout > 100) {
        printf("Erreur, la probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = (Network*)malloc(sizeof(Network));
    network->max_size = max_size;
    network->dropout = dropout;
    network->initialisation = initialisation;
    network->size = 1;
    network->input = (float****)malloc(sizeof(float***)*max_size);
    network->kernel = (Kernel**)malloc(sizeof(Kernel*)*(max_size-1));
    network->dim = (int**)malloc(sizeof(int*)*max_size);
    for (int i=0; i < max_size; i++) {
        network->dim[i] = (int*)malloc(sizeof(int)*2);
        network->kernel[i] = (Kernel*)malloc(sizeof(Kernel));
    }
    network->dim[0][0] = input_dim;
    network->dim[0][1] = input_depth;
    create_a_cube_input_layer(network, 0, input_depth, input_dim);
    return network;
}

Network* create_network_lenet5(int dropout, int activation, int initialisation) {
    Network* network = create_network(8, dropout, initialisation, 32, 1);
    add_convolution(network, 6, 5, activation);
    add_average_pooling(network, 2, activation);
    add_convolution(network, 16, 5, activation);
    add_average_pooling_flatten(network, 2, activation);
    add_dense(network, 120, 84, activation);
    add_dense(network, 84, 10, activation);
    add_dense(network, 10, 10, SOFTMAX);
    return network;
}

void create_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    network->input[pos] = (float***)malloc(sizeof(float**)*depth);
    for (int i=0; i < depth; i++) {
        network->input[pos][i] = (float**)malloc(sizeof(float*)*dim);
        for (int j=0; j < dim; j++) {
            network->input[pos][i][j] = (float*)malloc(sizeof(float)*dim);
        }
    }
    network->dim[pos][0] = dim;
    network->dim[pos][1] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    network->input[pos] = (float***)malloc(sizeof(float**));
    network->input[pos][0] = (float**)malloc(sizeof(float*));
    network->input[pos][0][0] = (float*)malloc(sizeof(float)*dim);
}

void add_average_pooling(Network* network, int kernel_size, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n]->cnn = NULL;
    network->kernel[n]->nn = NULL;
    network->kernel[n]->activation = activation + 100*kernel_size;
    create_a_cube_input_layer(network, n, network->dim[n-1][1], network->dim[n-1][0]/2);
    network->size++;
}

void add_average_pooling_flatten(Network* network, int kernel_size, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n]->cnn = NULL;
    network->kernel[n]->nn = NULL;
    network->kernel[n]->activation = activation + 100*kernel_size;
    int dim = (network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1])/(kernel_size*kernel_size);
    create_a_line_input_layer(network, n, dim);
    network->size++;
}

void add_convolution(Network* network, int nb_filter, int kernel_size, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche de convolution, le réseau est déjà plein\n");
        return;
    }
    int r = network->dim[n-1][1];
    int c = nb_filter;
    network->kernel[n]->nn = NULL;
    network->kernel[n]->activation = activation;
    network->kernel[n]->cnn = (Kernel_cnn*)malloc(sizeof(Kernel_cnn));
    Kernel_cnn* cnn = network->kernel[n]->cnn;

    cnn->k_size = kernel_size;
    cnn->rows = r;
    cnn->columns = c;
    cnn->w = (float****)malloc(sizeof(float***)*r);
    cnn->d_w = (float****)malloc(sizeof(float***)*r);
    for (int i=0; i < r; i++) {
        cnn->w[i] = (float***)malloc(sizeof(float**)*c);
        cnn->d_w[i] = (float***)malloc(sizeof(float**)*c);
        for (int j=0; j < c; j++) {
            cnn->w[i][j] = (float**)malloc(sizeof(float*)*kernel_size);
            cnn->d_w[i][j] = (float**)malloc(sizeof(float*)*kernel_size);
            for (int k=0; k < kernel_size; k++) {
                cnn->w[i][j][k] = (float*)malloc(sizeof(float)*kernel_size);
                cnn->d_w[i][j][k] = (float*)malloc(sizeof(float)*kernel_size);
            }
        }
    }
    cnn->bias = (float***)malloc(sizeof(float**)*c);
    cnn->d_bias = (float***)malloc(sizeof(float**)*c);
    for (int i=0; i < c; i++) {
        cnn->bias[i] = (float**)malloc(sizeof(float*)*kernel_size);
        cnn->d_bias[i] = (float**)malloc(sizeof(float*)*kernel_size);
        for (int j=0; j < kernel_size; j++) {
            cnn->bias[i][j] = (float*)malloc(sizeof(float)*kernel_size);
            cnn->d_bias[i][j] = (float*)malloc(sizeof(float)*kernel_size);
        }
    }
    create_a_cube_input_layer(network, n, c, network->dim[n-1][0] - 2*(kernel_size/2));
    int n_int = network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1];
    int n_out = network->dim[n][0]*network->dim[n][0]*network->dim[n][1];
    initialisation_3d_matrix(network->initialisation, cnn->bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_3d_matrix(ZERO, cnn->d_bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(network->initialisation, cnn->w, r, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(ZERO, cnn->d_w, r, c, kernel_size, kernel_size, n_int+n_out);
    network->size++;
}

void add_dense(Network* network, int input_units, int output_units, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n]->cnn = NULL;
    network->kernel[n]->nn = (Kernel_nn*)malloc(sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[n]->nn;
    network->kernel[n]->activation = activation;
    nn->input_units = input_units;
    nn->output_units = output_units;
    nn->bias = (float*)malloc(sizeof(float)*output_units);
    nn->d_bias = (float*)malloc(sizeof(float)*output_units);
    nn->weights = (float**)malloc(sizeof(float*)*input_units);
    nn->d_weights = (float**)malloc(sizeof(float*)*input_units);
    for (int i=0; i < input_units; i++) {
        nn->weights[i] = (float*)malloc(sizeof(float)*output_units);
        nn->d_weights[i] = (float*)malloc(sizeof(float)*output_units);
    }
    initialisation_1d_matrix(network->initialisation, nn->bias, output_units, output_units+input_units);
    initialisation_1d_matrix(ZERO, nn->d_bias, output_units, output_units+input_units);
    initialisation_2d_matrix(network->initialisation, nn->weights, input_units, output_units, output_units+input_units);
    initialisation_2d_matrix(ZERO, nn->d_weights, input_units, output_units, output_units+input_units);
    create_a_line_input_layer(network, n, output_units);
    network->size++;
}