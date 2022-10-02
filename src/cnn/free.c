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

void free_2d_average_pooling(Network* network, int pos) {
    free_a_cube_input_layer(network, pos+1, network->depth[pos-1], network->width[pos-1]/2);
}

void free_convolution(Network* network, int pos) {
    Kernel_cnn* k_pos = network->kernel[pos]->cnn;
    int c = k_pos->columns;
    int k_size = k_pos->k_size;
    int r = k_pos->rows;
    int bias_size = network->width[pos-1]-k_size+1; // Not sure of the value
    free_a_cube_input_layer(network, pos+1, c, bias_size);
    for (int i=0; i < c; i++) {
        for (int j=0; j < bias_size; j++) {
            free(k_pos->bias[i][j]);
            free(k_pos->d_bias[i][j]);
        }
        free(k_pos->bias[i]);
        free(k_pos->d_bias[i]);
    }
    free(k_pos->bias);
    free(k_pos->d_bias);

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                free(k_pos->w[i][j][k]);
                free(k_pos->d_w[i][j][k]);
            }
            free(k_pos->w[i][j]);
            free(k_pos->d_w[i][j]);
        }
        free(k_pos->w[i]);
        free(k_pos->d_w[i]);
    }
    free(k_pos->w);
    free(k_pos->d_w);

    free(k_pos);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        free(k_pos->weights[i]);
        free(k_pos->d_weights[i]);
    }
    free(k_pos->weights);
    free(k_pos->d_weights);

    free(k_pos->bias);
    free(k_pos->d_bias);

    free(k_pos);
}

void free_dense_linearisation(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        free(k_pos->weights[i]);
        free(k_pos->d_weights[i]);
    }
    free(k_pos->weights);
    free(k_pos->d_weights);

    free(k_pos->bias);
    free(k_pos->d_bias);

    free(k_pos);
}

void free_network_creation(Network* network) {
    free_a_cube_input_layer(network, 0, network->depth[0], network->width[0]);
    for (int i=0; i<network->max_size; i++)
        free(network->kernel[i]);
    free(network->width);
    free(network->depth);
    free(network->kernel);
    free(network->input);

    free(network);
}

void free_network_lenet5(Network* network) {
    free_dense(network, 6);
    free_dense(network, 5);
    free_dense_linearisation(network, 4);
    free_average_pooling(network, 3);
    free_convolution(network, 2);
    free_average_pooling(network, 1);
    free_convolution(network, 0);
    free_network_creation(network);
    if (network->size != network->max_size) {
        printf("033[33;1m[WARNING]\033[0m Le réseau LeNet5 est incomplet");
    }
}
