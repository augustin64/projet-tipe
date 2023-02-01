#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

#include "include/free.h"

void free_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim; j++) {
            gree(network->input[pos][i][j]);
            gree(network->input_z[pos][i][j]);
        }
        gree(network->input[pos][i]);
        gree(network->input_z[pos][i]);
    }
    gree(network->input[pos]);
    gree(network->input_z[pos]);
}

void free_a_line_input_layer(Network* network, int pos) {
    gree(network->input[pos][0][0]);
    gree(network->input_z[pos][0][0]);
    gree(network->input[pos][0]);
    gree(network->input_z[pos][0]);
    gree(network->input[pos]);
    gree(network->input_z[pos]);
}

void free_2d_pooling(Network* network, int pos) {
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
            gree(k_pos->bias[i][j]);
            gree(k_pos->d_bias[i][j]);
        }
        gree(k_pos->bias[i]);
        gree(k_pos->d_bias[i]);
    }
    gree(k_pos->bias);
    gree(k_pos->d_bias);

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                gree(k_pos->w[i][j][k]);
                gree(k_pos->d_w[i][j][k]);
            }
            gree(k_pos->w[i][j]);
            gree(k_pos->d_w[i][j]);
        }
        gree(k_pos->w[i]);
        gree(k_pos->d_w[i]);
    }
    gree(k_pos->w);
    gree(k_pos->d_w);

    gree(k_pos);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i]);
        gree(k_pos->d_weights[i]);
    }
    gree(k_pos->weights);
    gree(k_pos->d_weights);

    gree(k_pos->bias);
    gree(k_pos->d_bias);

    gree(k_pos);
}

void free_dense_linearisation(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->input_units;
    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i]);
        gree(k_pos->d_weights[i]);
    }
    gree(k_pos->weights);
    gree(k_pos->d_weights);

    gree(k_pos->bias);
    gree(k_pos->d_bias);

    gree(k_pos);
}

void free_network_creation(Network* network) {
    free_a_cube_input_layer(network, 0, network->depth[0], network->width[0]);
    for (int i=0; i < network->max_size-1; i++)
        gree(network->kernel[i]);
    gree(network->width);
    gree(network->depth);
    gree(network->kernel);
    gree(network->input);
    gree(network->input_z);

    gree(network);
}

void free_network(Network* network) {
    for (int i=network->size-2; i>=0; i--) {
        if (network->kernel[i]->cnn != NULL) { // Convolution
            free_convolution(network, i);
        } else if (network->kernel[i]->nn != NULL) {
            if (network->kernel[i]->linearisation == 0) { // Dense non linearised
                free_dense(network, i);
            } else { // Dense lineariation
                free_dense_linearisation(network, i);
            }
        } else { // Pooling
            free_2d_pooling(network, i);
        }
    }
    free_network_creation(network);
}
