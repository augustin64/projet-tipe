#include <stdio.h>
#include <stdlib.h>
#include "creation.h"
#include "function.h"
#include "initialisation.h"

Network* create_network(int max_size, int dropout, int initialisation, int input_dim, int input_depth) {
    if (dropout<0 || dropout>100) {
        printf("Erreur, la probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = malloc(sizeof(Network));
    network->max_size = max_size;
    network->dropout = dropout;
    network->initialisation = initialisation;
    network->size = 1;
    network->input = malloc(sizeof(float***)*max_size);
    network->kernel = malloc(sizeof(Kernel)*(max_size-1));
    create_a_cube_input_layer(network, 0, input_depth, input_dim);
    int i, j;
    network->dim = malloc(sizeof(int*)*max_size);
    for (i=0; i<max_size; i++) {
        network->dim[i] = malloc(sizeof(int)*2);
    }
    network->dim[0][0] = input_dim;
    network->dim[0][1] = input_depth;
    return network;
}

Network* create_network_lenet5(int dropout, int activation, int initialisation) {
    Network* network;
    network = create_network(8, dropout, initialisation, 32, 1);
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
    int i, j;
    network->input[pos] = malloc(sizeof(float**)*depth);
    for (i=0; i<depth; i++) {
        network->input[pos][i] = malloc(sizeof(float*)*dim);
        for (j=0; j<dim; j++) {
            network->input[pos][i][j] = malloc(sizeof(float)*dim);
        }
    }
    network->dim[pos][0] = dim;
    network->dim[pos][1] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    int i;
    network->input[pos] = malloc(sizeof(float**));
    network->input[pos][0] = malloc(sizeof(float*));
    network->input[pos][0][0] = malloc(sizeof(float)*dim);
}

void add_average_pooling(Network* network, int kernel_size, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = NULL;
    network->kernel[n].activation = activation + 100*kernel_size;
    create_a_cube_input_layer(network, n, network->dim[n-1][1], network->dim[n-1][0]/2);
    network->size++;
}

void add_average_pooling_flatten(Network* network, int kernel_size, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = NULL;
    network->kernel[n].activation = activation + 100*kernel_size;
    int dim = (network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1])/(kernel_size*kernel_size);
    create_a_line_input_layer(network, n, dim);
    network->size++;
}

void add_convolution(Network* network, int nb_filter, int kernel_size, int activation) {
    int n = network->size, i, j, k;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche de convolution, le réseau est déjà plein\n");
        return;
    }
    int r = network->dim[n-1][1];
    int c = nb_filter;
    network->kernel[n].nn = NULL;
    network->kernel[n].cnn = malloc(sizeof(Kernel_cnn));
    network->kernel[n].activation = activation;
    network->kernel[n].cnn->k_size = kernel_size;
    network->kernel[n].cnn->rows = r;
    network->kernel[n].cnn->columns = c;
    network->kernel[n].cnn->w = malloc(sizeof(float***)*r);
    network->kernel[n].cnn->d_w = malloc(sizeof(float***)*r);
    for (i=0; i<r; i++) {
        network->kernel[n].cnn->w[i] = malloc(sizeof(float**)*c);
        network->kernel[n].cnn->d_w[i] = malloc(sizeof(float**)*c);
        for (j=0; j<c; j++) {
            network->kernel[n].cnn->w[i][j] = malloc(sizeof(float*)*kernel_size);
            network->kernel[n].cnn->d_w[i][j] = malloc(sizeof(float*)*kernel_size);
            for (k=0; k<kernel_size; k++) {
                network->kernel[n].cnn->w[i][j][k] = malloc(sizeof(float)*kernel_size);
                network->kernel[n].cnn->d_w[i][j][k] = malloc(sizeof(float)*kernel_size);
            }
        }
    }
    network->kernel[n].cnn->bias = malloc(sizeof(float**)*c);
    network->kernel[n].cnn->d_bias = malloc(sizeof(float**)*c);
    for (i=0; i<c; i++) {
        network->kernel[n].cnn->bias[i] = malloc(sizeof(float*)*kernel_size);
        network->kernel[n].cnn->d_bias[i] = malloc(sizeof(float*)*kernel_size);
        for (j=0; j<kernel_size; j++) {
            network->kernel[n].cnn->bias[i][j] = malloc(sizeof(float)*kernel_size);
            network->kernel[n].cnn->d_bias[i][j] = malloc(sizeof(float)*kernel_size);
        }
    }
    create_a_cube_input_layer(network, n, c, network->dim[n-1][0] - 2*(kernel_size/2));
    int n_int = network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1];
    int n_out = network->dim[n][0]*network->dim[n][0]*network->dim[n][1];
    initialisation_3d_matrix(network->initialisation, network->kernel[n].cnn->bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_3d_matrix(ZERO, network->kernel[n].cnn->d_bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(network->initialisation, network->kernel[n].cnn->w, r, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(ZERO, network->kernel[n].cnn->d_w, r, c, kernel_size, kernel_size, n_int+n_out);
    network->size++;
}

void add_dense(Network* network, int input_units, int output_units, int activation) {
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = malloc(sizeof(Kernel_nn));
    network->kernel[n].activation = activation;
    network->kernel[n].nn->input_units = input_units;
    network->kernel[n].nn->output_units = output_units;
    network->kernel[n].nn->bias = malloc(sizeof(float)*output_units);
    network->kernel[n].nn->d_bias = malloc(sizeof(float)*output_units);
    network->kernel[n].nn->weights = malloc(sizeof(float*)*input_units);
    network->kernel[n].nn->d_weights = malloc(sizeof(float*)*input_units);
    for (int i=0; i<input_units; i++) {
        network->kernel[n].nn->weights[i] = malloc(sizeof(float)*output_units);
        network->kernel[n].nn->d_weights[i] = malloc(sizeof(float)*output_units);
    }
    initialisation_1d_matrix(network->initialisation, network->kernel[n].nn->bias, output_units, output_units+input_units);
    initialisation_1d_matrix(ZERO, network->kernel[n].nn->d_bias, output_units, output_units+input_units);
    initialisation_2d_matrix(network->initialisation, network->kernel[n].nn->weights, input_units, output_units, output_units+input_units);
    initialisation_2d_matrix(ZERO, network->kernel[n].nn->d_weights, input_units, output_units, output_units+input_units);
    create_a_line_input_layer(network, n, output_units);
    network->size++;
}