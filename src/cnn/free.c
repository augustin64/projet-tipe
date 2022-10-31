#include <stdlib.h>
#include <stdio.h>

#include "include/free.h"

void free_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim; j++) {
            free(network->input[pos][i][j]);
            free(network->input_z[pos][i][j]);
        }
        free(network->input[pos][i]);
        free(network->input_z[pos][i]);
    }
    free(network->input[pos]);
    free(network->input_z[pos]);
}

void free_a_line_input_layer(Network* network, int pos) {
    free(network->input[pos][0][0]);
    free(network->input_z[pos][0][0]);
    free(network->input[pos][0]);
    free(network->input_z[pos][0]);
    free(network->input[pos]);
    free(network->input_z[pos]);
}

void free_2d_average_pooling(Network* network, int pos) {
    free_a_cube_input_layer(network, pos+1, network->depth[pos+1], network->width[pos+1]);
}

void free_convolution(Network* network, int pos) {
    Kernel_cnn* k_pos = network->kernel[pos]->cnn;
    int c = k_pos->columns;
    int k_size = k_pos->k_size;
    int r = k_pos->rows;
    int bias_size = network->width[pos+1]; // Not sure of the value
    free_a_cube_input_layer(network, pos+1, network->depth[pos+1], network->width[pos+1]);
    for (int i=0; i < c; i++) {
        for (int j=0; j < bias_size; j++) {
            free(k_pos->bias[i][j]);
            free(k_pos->d_bias[i][j]);
            free(k_pos->last_d_bias[i][j]);
        }
        free(k_pos->bias[i]);
        free(k_pos->d_bias[i]);
        free(k_pos->last_d_bias[i]);
    }
    free(k_pos->bias);
    free(k_pos->d_bias);
    free(k_pos->last_d_bias);

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                free(k_pos->w[i][j][k]);
                free(k_pos->d_w[i][j][k]);
                free(k_pos->last_d_w[i][j][k]);
            }
            free(k_pos->w[i][j]);
            free(k_pos->d_w[i][j]);
            free(k_pos->last_d_w[i][j]);
        }
        free(k_pos->w[i]);
        free(k_pos->d_w[i]);
        free(k_pos->last_d_w[i]);
    }
    free(k_pos->w);
    free(k_pos->d_w);
    free(k_pos->last_d_w);

    free(k_pos);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        free(k_pos->weights[i]);
        free(k_pos->d_weights[i]);
        free(k_pos->last_d_weights[i]);
    }
    free(k_pos->weights);
    free(k_pos->d_weights);
    free(k_pos->last_d_weights);

    free(k_pos->bias);
    free(k_pos->d_bias);
    free(k_pos->last_d_bias);

    free(k_pos);
}

void free_dense_linearisation(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        free(k_pos->weights[i]);
        free(k_pos->d_weights[i]);
        free(k_pos->last_d_weights[i]);
    }
    free(k_pos->weights);
    free(k_pos->d_weights);
    free(k_pos->last_d_weights);

    free(k_pos->bias);
    free(k_pos->d_bias);
    free(k_pos->last_d_bias);

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
    free(network->input_z);

    free(network);
}

void free_network(Network* network) {
    for (int i=network->size-2; i>=0; i--) {
        if (network->kernel[i]->cnn != NULL) { // Convolution
            free_convolution(network, i);
        } else if (network->kernel[i]->nn != NULL) {
            if (network->depth[i]==1) { // Dense non linearised
                free_dense(network, i);
            } else { // Dense lineariation
                free_dense_linearisation(network, i);
            }
        } else { // Pooling
            free_2d_average_pooling(network, i);
        }
    }
    free_network_creation(network);
}
