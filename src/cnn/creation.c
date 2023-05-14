#include <stdio.h>
#include <stdlib.h>

#include "../common/include/memory_management.h"
#include "../common/include/colors.h"
#include "../common/include/utils.h"
#include "include/initialisation.h"
#include "include/function.h"

#include "include/creation.h"

Network* create_network(int max_size, float learning_rate, int dropout, int initialisation, int input_width, int input_depth) {
    if (dropout < 0 || dropout > 100) {
        printf_error("La probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = (Network*)nalloc(1, sizeof(Network));
    network->learning_rate = learning_rate;
    network->max_size = max_size;
    network->dropout = dropout;
    network->initialisation = initialisation;
    network->size = 1;
    network->input = (float****)nalloc(max_size, sizeof(float***));
    network->input_z = (float****)nalloc(max_size, sizeof(float***));
    network->kernel = (Kernel**)nalloc(max_size-1, sizeof(Kernel*));
    network->width = (int*)nalloc(max_size, sizeof(int*));
    network->depth = (int*)nalloc(max_size, sizeof(int*));
    for (int i=0; i < max_size-1; i++) {
        network->kernel[i] = (Kernel*)nalloc(1, sizeof(Kernel));
    }
    network->width[0] = input_width;
    network->depth[0] = input_depth;
    create_a_cube_input_layer(network, 0, input_depth, input_width);
    create_a_cube_input_z_layer(network, 0, input_depth, input_width);
    return network;
}

Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth) {
    Network* network = create_network(8, learning_rate, dropout, initialisation, input_width, input_depth);
    add_convolution(network, 5, 6, 1, 0, activation);
    add_average_pooling(network, 2, 2, 0);
    add_convolution(network, 5, 16, 1, 0, activation);
    add_average_pooling(network, 2, 2, 0);
    add_dense_linearisation(network, 120, activation);
    add_dense(network, 84, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}

Network* create_network_alexnet(float learning_rate, int dropout, int activation, int initialisation, int size_output) {
    Network* network = create_network(12, learning_rate, dropout, initialisation, 227, 3);
    add_convolution(network, 11, 96, 4, 0, activation);
    add_average_pooling(network, 3, 2, 0);
    add_convolution(network, 5, 256, 1, 2, activation);
    add_average_pooling(network, 3, 2, 0);
    add_convolution(network, 3, 384, 1, 1, activation);
    add_convolution(network, 3, 384, 1, 1, activation);
    add_convolution(network, 3, 256, 1, 1, activation);
    add_average_pooling(network, 3, 2, 0);
    add_dense_linearisation(network, 4096, activation);
    add_dense(network, 4096, activation);
    add_dense(network, size_output, activation);
    return network;
}

Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth) {
    Network* network = create_network(3, learning_rate, dropout, initialisation, input_width, input_depth);
    add_dense_linearisation(network, 80, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}

void create_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    network->input[pos] = (float***)nalloc(depth, sizeof(float**));
    for (int i=0; i < depth; i++) {
        network->input[pos][i] = (float**)nalloc(dim, sizeof(float*));
        for (int j=0; j < dim; j++) {
            network->input[pos][i][j] = (float*)nalloc(dim, sizeof(float));
        }
    }
    network->width[pos] = dim;
    network->depth[pos] = depth;
}

void create_a_cube_input_z_layer(Network* network, int pos, int depth, int dim) {
    network->input_z[pos] = (float***)nalloc(depth, sizeof(float**));
    for (int i=0; i < depth; i++) {
        network->input_z[pos][i] = (float**)nalloc(dim, sizeof(float*));
        for (int j=0; j < dim; j++) {
            network->input_z[pos][i][j] = (float*)nalloc(dim, sizeof(float));
        }
    }
    network->width[pos] = dim;
    network->depth[pos] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    network->input[pos] = (float***)nalloc(1, sizeof(float**));
    network->input[pos][0] = (float**)nalloc(1, sizeof(float*));
    network->input[pos][0][0] = (float*)nalloc(dim, sizeof(float));
    network->width[pos] = dim;
    network->depth[pos] = 1;
}

void create_a_line_input_z_layer(Network* network, int pos, int dim) {
    network->input_z[pos] = (float***)nalloc(1, sizeof(float**));
    network->input_z[pos][0] = (float**)nalloc(1, sizeof(float*));
    network->input_z[pos][0][0] = (float*)nalloc(dim, sizeof(float));
    network->width[pos] = dim;
    network->depth[pos] = 1;
}

void add_average_pooling(Network* network, int kernel_size, int stride, int padding) {
    int n = network->size;
    int k_pos = n-1;
    if (network->max_size == n) {
        printf_error("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    int input_width = network->width[k_pos];
    int output_width = (2*padding + input_width - (kernel_size - stride))/stride;

    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->stride = stride;
    network->kernel[k_pos]->padding = padding;
    network->kernel[k_pos]->activation = IDENTITY; // Ne contient pas de fonction d'activation
    network->kernel[k_pos]->linearisation = DOESNT_LINEARISE;
    network->kernel[k_pos]->pooling = AVG_POOLING;

    create_a_cube_input_layer(network, n, network->depth[n-1], output_width);
    create_a_cube_input_z_layer(network, n, network->depth[n-1], output_width);
    network->size++;
}

void add_max_pooling(Network* network, int kernel_size, int stride, int padding) {
    int n = network->size;
    int k_pos = n-1;
    if (network->max_size == n) {
        printf_error("Impossible de rajouter une couche de max pooling, le réseau est déjà plein\n");
        return;
    }
    int input_width = network->width[k_pos];
    int output_width = (2*padding + input_width - (kernel_size - stride))/stride;

    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->activation = IDENTITY; // Ne contient pas de fonction d'activation
    network->kernel[k_pos]->linearisation = DOESNT_LINEARISE;
    network->kernel[k_pos]->pooling = MAX_POOLING;

    create_a_cube_input_layer(network, n, network->depth[n-1], output_width);
    create_a_cube_input_z_layer(network, n, network->depth[n-1], output_width);
    network->size++;
}

void add_convolution(Network* network, int kernel_size, int number_of_kernels, int stride, int padding, int activation) {
    int n = network->size;
    int k_pos = n-1;
    if (network->max_size == n) {
        printf_error("Impossible de rajouter une couche de convolution, le réseau est déjà plein \n");
        return;
    }
    int input_depth = network->depth[k_pos];
    int input_width = network->width[k_pos];

    int output_width = (2*padding + input_width - (kernel_size - stride))/stride;
    int output_depth = number_of_kernels;

    int bias_size = output_width;

    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->stride = stride;
    network->kernel[k_pos]->padding = padding;
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = DOESNT_LINEARISE;
    network->kernel[k_pos]->pooling = NO_POOLING;

    network->kernel[k_pos]->cnn = (Kernel_cnn*)nalloc(1, sizeof(Kernel_cnn));
    Kernel_cnn* cnn = network->kernel[k_pos]->cnn;
    cnn->k_size = kernel_size;
    cnn->rows = input_depth;
    cnn->columns = output_depth;

    cnn->weights = (float****)nalloc(input_depth, sizeof(float***));
    cnn->d_weights = (float****)nalloc(input_depth, sizeof(float***));
    #ifdef ADAM_CNN_WEIGHTS
    cnn->s_d_weights = (float****)nalloc(input_depth, sizeof(float***));
    cnn->v_d_weights = (float****)nalloc(input_depth, sizeof(float***));
    #endif
    for (int i=0; i < input_depth; i++) {
        cnn->weights[i] = (float***)nalloc(output_depth, sizeof(float**));
        cnn->d_weights[i] = (float***)nalloc(output_depth, sizeof(float**));
        #ifdef ADAM_CNN_WEIGHTS
        cnn->s_d_weights[i] = (float***)nalloc(output_depth, sizeof(float**));
        cnn->v_d_weights[i] = (float***)nalloc(output_depth, sizeof(float**));
        #endif
        for (int j=0; j < output_depth; j++) {
            cnn->weights[i][j] = (float**)nalloc(kernel_size, sizeof(float*));
            cnn->d_weights[i][j] = (float**)nalloc(kernel_size, sizeof(float*));
            #ifdef ADAM_CNN_WEIGHTS
            cnn->s_d_weights[i][j] = (float**)nalloc(kernel_size, sizeof(float*));
            cnn->v_d_weights[i][j] = (float**)nalloc(kernel_size, sizeof(float*));
            #endif
            for (int k=0; k < kernel_size; k++) {
                cnn->weights[i][j][k] = (float*)nalloc(kernel_size, sizeof(float));
                cnn->d_weights[i][j][k] = (float*)nalloc(kernel_size, sizeof(float));
                #ifdef ADAM_CNN_WEIGHTS
                cnn->s_d_weights[i][j][k] = (float*)nalloc(kernel_size, sizeof(float));
                cnn->v_d_weights[i][j][k] = (float*)nalloc(kernel_size, sizeof(float));
                #endif
                for (int l=0; l < kernel_size; l++) {
                    cnn->d_weights[i][j][k][l] = 0.;
                    #ifdef ADAM_CNN_WEIGHTS
                    cnn->s_d_weights[i][j][k][l] = 0.;
                    cnn->v_d_weights[i][j][k][l] = 0.;
                    #endif
                }
            }
        }
    }

    cnn->bias = (float***)nalloc(output_depth, sizeof(float**));
    cnn->d_bias = (float***)nalloc(output_depth, sizeof(float**));
    #ifdef ADAM_CNN_BIAS
    cnn->s_d_bias = (float***)nalloc(output_depth, sizeof(float**));
    cnn->v_d_bias = (float***)nalloc(output_depth, sizeof(float**));
    #endif
    for (int i=0; i < output_depth; i++) {
        cnn->bias[i] = (float**)nalloc(bias_size, sizeof(float*));
        cnn->d_bias[i] = (float**)nalloc(bias_size, sizeof(float*));
        #ifdef ADAM_CNN_BIAS
        cnn->s_d_bias[i] = (float**)nalloc(bias_size, sizeof(float*));
        cnn->v_d_bias[i] = (float**)nalloc(bias_size, sizeof(float*));
        #endif
        for (int j=0; j < bias_size; j++) {
            cnn->bias[i][j] = (float*)nalloc(bias_size, sizeof(float));
            cnn->d_bias[i][j] = (float*)nalloc(bias_size, sizeof(float));
            #ifdef ADAM_CNN_BIAS
            cnn->s_d_bias[i][j] = (float*)nalloc(bias_size, sizeof(float));
            cnn->v_d_bias[i][j] = (float*)nalloc(bias_size, sizeof(float));
            #endif
            for (int k=0; k < bias_size; k++) {
                cnn->d_bias[i][j][k] = 0.;
                #ifdef ADAM_CNN_BIAS
                cnn->s_d_bias[i][j][k] = 0.;
                cnn->v_d_bias[i][j][k] = 0.;
                #endif
            }
        }
    }

    int n_in = network->width[n-1]*network->width[n-1]*network->depth[n-1];
    int n_out = network->width[n]*network->width[n]*network->depth[n];
    initialisation_3d_matrix(network->initialisation, cnn->bias, output_depth, output_width, output_width, n_in, n_out);
    initialisation_4d_matrix(network->initialisation, cnn->weights, input_depth, output_depth, kernel_size, kernel_size, n_in, n_out);
    create_a_cube_input_layer(network, n, output_depth, bias_size);
    create_a_cube_input_z_layer(network, n, output_depth, bias_size);
    network->size++;
}

void add_dense(Network* network, int size_output, int activation) {
    int n = network->size;
    int k_pos = n-1;
    int size_input = network->width[k_pos];
    if (network->max_size == n) {
        printf_error("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->stride = -1; // N'est pas utilisé dans une couche dense
    network->kernel[k_pos]->padding = -1; // N'est pas utilisé dans une couche dense
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = DOESNT_LINEARISE;
    network->kernel[k_pos]->pooling = NO_POOLING;

    network->kernel[k_pos]->nn = (Kernel_nn*)nalloc(1, sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    nn->size_input = size_input;
    nn->size_output = size_output;

    nn->bias = (float*)nalloc(size_output, sizeof(float));
    nn->d_bias = (float*)nalloc(size_output, sizeof(float));
    #ifdef ADAM_DENSE_BIAS
    nn->s_d_bias = (float*)nalloc(size_output, sizeof(float));
    nn->v_d_bias = (float*)nalloc(size_output, sizeof(float));
    #endif
    for (int i=0; i < size_output; i++) {
        nn->d_bias[i] = 0.;
        #ifdef ADAM_DENSE_BIAS
        nn->s_d_bias[i] = 0.;
        nn->v_d_bias[i] = 0.;
        #endif
    }

    nn->weights = (float**)nalloc(size_input, sizeof(float*));
    nn->d_weights = (float**)nalloc(size_input, sizeof(float*));
    #ifdef ADAM_DENSE_WEIGHTS
    nn->s_d_weights = (float**)nalloc(size_input, sizeof(float*));
    nn->v_d_weights = (float**)nalloc(size_input, sizeof(float*));
    #endif
    for (int i=0; i < size_input; i++) {
        nn->weights[i] = (float*)nalloc(size_output, sizeof(float));
        nn->d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        #ifdef ADAM_DENSE_WEIGHTS
        nn->s_d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        nn->v_d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        #endif
        for (int j=0; j < size_output; j++) {
            nn->d_weights[i][j] = 0.;
            #ifdef ADAM_DENSE_WEIGHTS
            nn->s_d_weights[i][j] = 0.;
            nn->v_d_weights[i][j] = 0.;
            #endif
        }
    }

    initialisation_1d_matrix(network->initialisation, nn->bias, size_output, size_input, size_output);
    initialisation_2d_matrix(network->initialisation, nn->weights, size_input, size_output, size_input, size_output);
    create_a_line_input_layer(network, n, size_output);
    create_a_line_input_z_layer(network, n, size_output);
    network->size++;
}

void add_dense_linearisation(Network* network, int size_output, int activation) {
    // Can replace size_input by a research of this dim

    int n = network->size;
    int k_pos = n-1;
    int size_input = network->depth[k_pos]*network->width[k_pos]*network->width[k_pos];
    if (network->max_size == n) {
        printf_error("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->stride = -1; // N'est pas utilisé dans une couche dense
    network->kernel[k_pos]->padding = -1; // N'est pas utilisé dans une couche dense
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = DO_LINEARISE;
    network->kernel[k_pos]->pooling = NO_POOLING;

    network->kernel[k_pos]->nn = (Kernel_nn*)nalloc(1, sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    nn->size_input = size_input;
    nn->size_output = size_output;

    nn->bias = (float*)nalloc(size_output, sizeof(float));
    nn->d_bias = (float*)nalloc(size_output, sizeof(float));
    #ifdef ADAM_DENSE_BIAS
    nn->s_d_bias = (float*)nalloc(size_output, sizeof(float));
    nn->v_d_bias = (float*)nalloc(size_output, sizeof(float));
    #endif
    for (int i=0; i < size_output; i++) {
        nn->d_bias[i] = 0.;
        #ifdef ADAM_DENSE_BIAS
        nn->s_d_bias[i] = 0.;
        nn->v_d_bias[i] = 0.;
        #endif
    }

    nn->weights = (float**)nalloc(size_input, sizeof(float*));
    nn->d_weights = (float**)nalloc(size_input, sizeof(float*));
    #ifdef ADAM_DENSE_WEIGHTS
    nn->s_d_weights = (float**)nalloc(size_input, sizeof(float*));
    nn->v_d_weights = (float**)nalloc(size_input, sizeof(float*));
    #endif
    for (int i=0; i < size_input; i++) {
        nn->weights[i] = (float*)nalloc(size_output, sizeof(float));
        nn->d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        #ifdef ADAM_DENSE_WEIGHTS
        nn->s_d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        nn->v_d_weights[i] = (float*)nalloc(size_output, sizeof(float));
        #endif
        for (int j=0; j < size_output; j++) {
            nn->d_weights[i][j] = 0.;
            #ifdef ADAM_DENSE_WEIGHTS
            nn->s_d_weights[i][j] = 0.;
            nn->v_d_weights[i][j] = 0.;
            #endif
        }
    }

    initialisation_1d_matrix(network->initialisation, nn->bias, size_output, size_input, size_output);
    initialisation_2d_matrix(network->initialisation, nn->weights, size_input, size_output, size_input, size_output);
    create_a_line_input_layer(network, n, size_output);
    create_a_line_input_z_layer(network, n, size_output);
    network->size++;
}