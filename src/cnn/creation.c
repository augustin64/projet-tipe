#include <stdio.h>
#include <stdlib.h>

#include "../include/memory_management.h"
#include "include/initialisation.h"
#include "include/function.h"
#include "../include/utils.h"

#include "include/creation.h"

Network* create_network(int max_size, float learning_rate, int dropout, int initialisation, int input_dim, int input_depth) {
    if (dropout < 0 || dropout > 100) {
        printf("Erreur, la probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = (Network*)nalloc(sizeof(Network));
    network->learning_rate = learning_rate;
    network->max_size = max_size;
    network->dropout = dropout;
    network->initialisation = initialisation;
    network->size = 1;
    network->input = (float****)nalloc(sizeof(float***)*max_size);
    network->input_z = (float****)nalloc(sizeof(float***)*max_size);
    network->kernel = (Kernel**)nalloc(sizeof(Kernel*)*(max_size-1));
    network->width = (int*)nalloc(sizeof(int*)*max_size);
    network->depth = (int*)nalloc(sizeof(int*)*max_size);
    for (int i=0; i < max_size-1; i++) {
        network->kernel[i] = (Kernel*)nalloc(sizeof(Kernel));
    }
    network->kernel[0]->linearisation = 0;
    network->width[0] = input_dim;
    network->depth[0] = input_depth;
    network->kernel[0]->nn = NULL;
    network->kernel[0]->cnn = NULL;
    create_a_cube_input_layer(network, 0, input_depth, input_dim);
    create_a_cube_input_z_layer(network, 0, input_depth, input_dim);
    return network;
}

Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth) {
    Network* network = create_network(8, learning_rate, dropout, initialisation, input_dim, input_depth);
    network->kernel[0]->activation = activation;
    add_convolution(network, 6, 28, activation);
    add_2d_average_pooling(network, 14);
    add_convolution(network, 16, 10, activation);
    add_2d_average_pooling(network, 5);
    add_dense_linearisation(network, 120, activation);
    add_dense(network, 84, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}

Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth) {
    Network* network = create_network(3, learning_rate, dropout, initialisation, input_dim, input_depth);
    network->kernel[0]->activation = activation;
    add_dense_linearisation(network, 80, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}

void create_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    network->input[pos] = (float***)nalloc(sizeof(float**)*depth);
    for (int i=0; i < depth; i++) {
        network->input[pos][i] = (float**)nalloc(sizeof(float*)*dim);
        for (int j=0; j < dim; j++) {
            network->input[pos][i][j] = (float*)nalloc(sizeof(float)*dim);
        }
    }
    network->width[pos] = dim;
    network->depth[pos] = depth;
}

void create_a_cube_input_z_layer(Network* network, int pos, int depth, int dim) {
    network->input_z[pos] = (float***)nalloc(sizeof(float**)*depth);
    for (int i=0; i < depth; i++) {
        network->input_z[pos][i] = (float**)nalloc(sizeof(float*)*dim);
        for (int j=0; j < dim; j++) {
            network->input_z[pos][i][j] = (float*)nalloc(sizeof(float)*dim);
        }
    }
    network->width[pos] = dim;
    network->depth[pos] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    network->input[pos] = (float***)nalloc(sizeof(float**));
    network->input[pos][0] = (float**)nalloc(sizeof(float*));
    network->input[pos][0][0] = (float*)nalloc(sizeof(float)*dim);
    network->width[pos] = dim;
    network->depth[pos] = 1;
}

void create_a_line_input_z_layer(Network* network, int pos, int dim) {
    network->input_z[pos] = (float***)nalloc(sizeof(float**));
    network->input_z[pos][0] = (float**)nalloc(sizeof(float*));
    network->input_z[pos][0][0] = (float*)nalloc(sizeof(float)*dim);
    network->width[pos] = dim;
    network->depth[pos] = 1;
}

void add_2d_average_pooling(Network* network, int dim_output) {
    int n = network->size;
    int k_pos = n-1;
    int dim_input = network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    if (dim_input%dim_output != 0) {
        printf("Erreur de dimension dans l'average pooling\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->activation = IDENTITY; // Ne contient pas de fonction d'activation
    network->kernel[k_pos]->linearisation = 0;
    network->kernel[k_pos]->pooling = 1;
    create_a_cube_input_layer(network, n, network->depth[n-1], network->width[n-1]/2);
    create_a_cube_input_z_layer(network, n, network->depth[n-1], network->width[n-1]/2); // Will it be used ?
    network->size++;
}

void add_2d_max_pooling(Network* network, int dim_output) {
    int n = network->size;
    int k_pos = n-1;
    int dim_input = network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche de max pooling, le réseau est déjà plein\n");
        return;
    }
    if (dim_input%dim_output != 0) {
        printf("Erreur de dimension dans le max pooling\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->activation = IDENTITY; // Ne contient pas de fonction d'activation
    network->kernel[k_pos]->linearisation = 0;
    network->kernel[k_pos]->pooling = 2;
    create_a_cube_input_layer(network, n, network->depth[n-1], network->width[n-1]/2);
    create_a_cube_input_z_layer(network, n, network->depth[n-1], network->width[n-1]/2); // Will it be used ?
    network->size++;
}

void add_convolution(Network* network, int depth_output, int dim_output, int activation) {
    int n = network->size;
    int k_pos = n-1;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche de convolution, le réseau est déjà plein \n");
        return;
    }
    int depth_input = network->depth[k_pos];
    int dim_input = network->width[k_pos];

    int bias_size = dim_output;
    int kernel_size = dim_input - dim_output +1;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = 0;
    network->kernel[k_pos]->pooling = 0;
    network->kernel[k_pos]->cnn = (Kernel_cnn*)nalloc(sizeof(Kernel_cnn));
    Kernel_cnn* cnn = network->kernel[k_pos]->cnn;

    cnn->k_size = kernel_size;
    cnn->rows = depth_input;
    cnn->columns = depth_output;
    cnn->weights = (float****)nalloc(sizeof(float***)*depth_input);
    cnn->d_weights = (float****)nalloc(sizeof(float***)*depth_input);
    for (int i=0; i < depth_input; i++) {
        cnn->weights[i] = (float***)nalloc(sizeof(float**)*depth_output);
        cnn->d_weights[i] = (float***)nalloc(sizeof(float**)*depth_output);
        for (int j=0; j < depth_output; j++) {
            cnn->weights[i][j] = (float**)nalloc(sizeof(float*)*kernel_size);
            cnn->d_weights[i][j] = (float**)nalloc(sizeof(float*)*kernel_size);
            for (int k=0; k < kernel_size; k++) {
                cnn->weights[i][j][k] = (float*)nalloc(sizeof(float)*kernel_size);
                cnn->d_weights[i][j][k] = (float*)nalloc(sizeof(float)*kernel_size);
                for (int l=0; l < kernel_size; l++) {
                    cnn->d_weights[i][j][k][l] = 0.;
                }
            }
        }
    }
    cnn->bias = (float***)nalloc(sizeof(float**)*depth_output);
    cnn->d_bias = (float***)nalloc(sizeof(float**)*depth_output);
    for (int i=0; i < depth_output; i++) {
        cnn->bias[i] = (float**)nalloc(sizeof(float*)*bias_size);
        cnn->d_bias[i] = (float**)nalloc(sizeof(float*)*bias_size);
        for (int j=0; j < bias_size; j++) {
            cnn->bias[i][j] = (float*)nalloc(sizeof(float)*bias_size);
            cnn->d_bias[i][j] = (float*)nalloc(sizeof(float)*bias_size);
            for (int k=0; k < bias_size; k++) {
                cnn->d_bias[i][j][k] = 0.;
            }
        }
    }
    int n_in = network->width[n-1]*network->width[n-1]*network->depth[n-1];
    int n_out = network->width[n]*network->width[n]*network->depth[n];
    initialisation_3d_matrix(network->initialisation, cnn->bias, depth_output, dim_output, dim_output, n_in, n_out);
    initialisation_4d_matrix(network->initialisation, cnn->weights, depth_input, depth_output, kernel_size, kernel_size, n_in, n_out);
    create_a_cube_input_layer(network, n, depth_output, bias_size);
    create_a_cube_input_z_layer(network, n, depth_output, bias_size);
    network->size++;
}

void add_dense(Network* network, int size_output, int activation) {
    int n = network->size;
    int k_pos = n-1;
    int size_input = network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = (Kernel_nn*)nalloc(sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = 0;
    network->kernel[k_pos]->pooling = 0;
    nn->size_input = size_input;
    nn->size_output = size_output;
    nn->bias = (float*)nalloc(sizeof(float)*size_output);
    nn->d_bias = (float*)nalloc(sizeof(float)*size_output);
    for (int i=0; i < size_output; i++) {
        nn->d_bias[i] = 0.;
    }

    nn->weights = (float**)nalloc(sizeof(float*)*size_input);
    nn->d_weights = (float**)nalloc(sizeof(float*)*size_input);
    for (int i=0; i < size_input; i++) {
        nn->weights[i] = (float*)nalloc(sizeof(float)*size_output);
        nn->d_weights[i] = (float*)nalloc(sizeof(float)*size_output);
        for (int j=0; j < size_output; j++) {
            nn->d_weights[i][j] = 0.;
        }
    }

    initialisation_1d_matrix(network->initialisation, nn->bias, size_output, size_input);
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
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = (Kernel_nn*)nalloc(sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    network->kernel[k_pos]->activation = activation;
    network->kernel[k_pos]->linearisation = 1;
    network->kernel[k_pos]->pooling = 0;
    nn->size_input = size_input;
    nn->size_output = size_output;

    nn->bias = (float*)nalloc(sizeof(float)*size_output);
    nn->d_bias = (float*)nalloc(sizeof(float)*size_output);
    for (int i=0; i < size_output; i++) {
        nn->d_bias[i] = 0.;
    }
    nn->weights = (float**)nalloc(sizeof(float*)*size_input);
    nn->d_weights = (float**)nalloc(sizeof(float*)*size_input);
    for (int i=0; i < size_input; i++) {
        nn->weights[i] = (float*)nalloc(sizeof(float)*size_output);
        nn->d_weights[i] = (float*)nalloc(sizeof(float)*size_output);
        for (int j=0; j < size_output; j++) {
            nn->d_weights[i][j] = 0.;
        }
    }
    initialisation_1d_matrix(network->initialisation, nn->bias, size_output, size_input);
    initialisation_2d_matrix(network->initialisation, nn->weights, size_input, size_output, size_input, size_output);
    create_a_line_input_layer(network, n, size_output);
    create_a_line_input_z_layer(network, n, size_output);
    network->size++;
}