#include <stdlib.h>
#include <stdio.h>
#include "include/free.h"

void free_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim; j++) {
            free(network->input[pos][i][j]);
        }
        free(network->input[pos][i]);
    }
    free(network->input[pos]);
}

void free_a_line_input_layer(Network* network, int pos) {
    free(network->input[pos][0][0]);
    free(network->input[pos][0]);
    free(network->input[pos]);
}

void free_average_pooling(Network* network, int pos) {
    free_a_cube_input_layer(network, pos, network->depth[pos-1], network->width[pos-1]/2);
}

void free_average_pooling_flatten(Network* network, int pos) {
    free_a_line_input_layer(network, pos);
}

void free_convolution(Network* network, int pos) {
    int c = network->kernel[pos]->cnn->columns;
    int k_size = network->kernel[pos]->cnn->k_size;
    int r = network->kernel[pos]->cnn->rows;
    free_a_cube_input_layer(network, pos, c, network->width[pos-1] - 2*(k_size/2));
    for (int i=0; i < c; i++) {
        for (int j=0; j < k_size; j++) {
            free(network->kernel[pos]->cnn->bias[i][j]);
            free(network->kernel[pos]->cnn->d_bias[i][j]);
        }
        free(network->kernel[pos]->cnn->bias[i]);
        free(network->kernel[pos]->cnn->d_bias[i]);
    }
    free(network->kernel[pos]->cnn->bias);
    free(network->kernel[pos]->cnn->d_bias);

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                free(network->kernel[pos]->cnn->w[i][j][k]);
                free(network->kernel[pos]->cnn->d_w[i][j][k]);
            }
            free(network->kernel[pos]->cnn->w[i][j]);
            free(network->kernel[pos]->cnn->d_w[i][j]);
        }
        free(network->kernel[pos]->cnn->w[i]);
        free(network->kernel[pos]->cnn->d_w[i]);
    }
    free(network->kernel[pos]->cnn->w);
    free(network->kernel[pos]->cnn->d_w);

    free(network->kernel[pos]->cnn);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos);
    int dim = network->kernel[pos]->nn->output_units;
    for (int i=0; i < dim; i++) {
        free(network->kernel[pos]->nn->weights[i]);
        free(network->kernel[pos]->nn->d_weights[i]);
    }
    free(network->kernel[pos]->nn->weights);
    free(network->kernel[pos]->nn->d_weights);

    free(network->kernel[pos]->nn->bias);
    free(network->kernel[pos]->nn->d_bias);

    free(network->kernel[pos]->nn);
}

void free_network_creation(Network* network) {
    free_a_cube_input_layer(network, 0, network->depth[0], network->width[0]);

    for (int i=0; i < network->max_size; i++) {
        free(network->dim[i]);
    }
    free(network->dim);

    free(network->kernel);
    free(network->input);

    free(network);
}

void free_network_lenet5(Network* network) {
    free_dense(network, 6);
    free_dense(network, 5);
    free_dense(network, 4);
    free_average_pooling_flatten(network, 3);
    free_convolution(network, 2);
    free_average_pooling(network, 1);
    free_convolution(network, 0);
    free_network_creation(network);
    if (network->size != network->max_size) {
        printf("Attention, le réseau LeNet5 n'est pas complet");
    }
}
