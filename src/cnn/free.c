#include <stdlib.h>
#include <stdio.h>

#include "../include/memory_management.h"

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

void free_pooling(Network* network, int pos) {
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
            #ifdef ADAM_CNN_BIAS
            gree(k_pos->s_d_bias[i][j]);
            gree(k_pos->v_d_bias[i][j]);
            #endif
        }
        gree(k_pos->bias[i]);
        gree(k_pos->d_bias[i]);
        #ifdef ADAM_CNN_BIAS
        gree(k_pos->s_d_bias[i]);
        gree(k_pos->v_d_bias[i]);
        #endif
    }
    gree(k_pos->bias);
    gree(k_pos->d_bias);
    #ifdef ADAM_CNN_BIAS
    gree(k_pos->s_d_bias);
    gree(k_pos->v_d_bias);
    #endif

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                gree(k_pos->weights[i][j][k]);
                gree(k_pos->d_weights[i][j][k]);
                #ifdef ADAM_CNN_WEIGHTS
                gree(k_pos->s_d_weights[i][j][k]);
                gree(k_pos->v_d_weights[i][j][k]);
                #endif
            }
            gree(k_pos->weights[i][j]);
            gree(k_pos->d_weights[i][j]);
            #ifdef ADAM_CNN_WEIGHTS
            gree(k_pos->s_d_weights[i][j]);
            gree(k_pos->v_d_weights[i][j]);
            #endif
        }
        gree(k_pos->weights[i]);
        gree(k_pos->d_weights[i]);
        #ifdef ADAM_CNN_WEIGHTS
        gree(k_pos->s_d_weights[i]);
        gree(k_pos->v_d_weights[i]);
        #endif
    }
    gree(k_pos->weights);
    gree(k_pos->d_weights);
    #ifdef ADAM_CNN_WEIGHTS
    gree(k_pos->s_d_weights);
    gree(k_pos->v_d_weights);
    #endif

    gree(k_pos);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->size_input;
    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i]);
        gree(k_pos->d_weights[i]);
        #ifdef ADAM_DENSE_WEIGHTS
        gree(k_pos->s_d_weights[i]);
        gree(k_pos->v_d_weights[i]);
        #endif
    }
    gree(k_pos->weights);
    gree(k_pos->d_weights);
    #ifdef ADAM_DENSE_WEIGHTS
    gree(k_pos->s_d_weights);
    gree(k_pos->v_d_weights);
    #endif

    gree(k_pos->bias);
    gree(k_pos->d_bias);
    #ifdef ADAM_DENSE_BIAS
    gree(k_pos->s_d_bias);
    gree(k_pos->v_d_bias);
    #endif

    gree(k_pos);
}

void free_dense_linearisation(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->size_input;
    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i]);
        gree(k_pos->d_weights[i]);
        #ifdef ADAM_DENSE_WEIGHTS
        gree(k_pos->s_d_weights[i]);
        gree(k_pos->v_d_weights[i]);
        #endif
    }
    gree(k_pos->weights);
    gree(k_pos->d_weights);
    #ifdef ADAM_DENSE_WEIGHTS
    gree(k_pos->s_d_weights);
    gree(k_pos->v_d_weights);
    #endif

    gree(k_pos->bias);
    gree(k_pos->d_bias);
    #ifdef ADAM_DENSE_BIAS
    gree(k_pos->s_d_bias);
    gree(k_pos->v_d_bias);
    #endif

    gree(k_pos);
}

void free_network_creation(Network* network) {
    free_a_cube_input_layer(network, 0, network->depth[0], network->width[0]);
    for (int i=0; i < network->max_size-1; i++) {
        gree(network->kernel[i]);
    }
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
            if (network->kernel[i]->linearisation == DOESNT_LINEARISE) { // Dense non linearized
                free_dense(network, i);
            } else { // Dense linearisation
                free_dense_linearisation(network, i);
            }
        } else { // Pooling
            free_pooling(network, i);
        }
    }
    free_network_creation(network);
}
