#include <stdio.h>
#include <stdlib.h>

#include "include/initialisation.h"
#include "include/function.h"

#include "include/creation.h"

Network* create_network(int max_size, int learning_rate, int dropout, int initialisation, int input_dim, int input_depth) {
    if (dropout < 0 || dropout > 100) {
        printf("Erreur, la probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = (Network*)malloc(sizeof(Network)); 
    network->learning_rate = learning_rate;
    network->max_size = max_size; 
    network->dropout = dropout; 
    network->initialisation = initialisation; 
    network->size = 1; 
    network->input = (float****)malloc(sizeof(float***)*max_size); 
    network->kernel = (Kernel**)malloc(sizeof(Kernel*)*max_size); 
    network->width = (int*)malloc(sizeof(int*)*max_size); 
    network->depth = (int*)malloc(sizeof(int*)*max_size); 
    for (int i=0; i < max_size; i++) {
        network->kernel[i] = (Kernel*)malloc(sizeof(Kernel));
    }
    network->width[0] = input_dim; 
    network->depth[0] = input_depth; 
    network->kernel[0]->nn = NULL; 
    network->kernel[0]->cnn = NULL; 
    create_a_cube_input_layer(network, 0, input_depth, input_dim); 
    return network;
}

Network* create_network_lenet5(int learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth) {
    Network* network = create_network(8, learning_rate, dropout, initialisation, input_dim, input_depth); 
    network->kernel[0]->activation = activation;  
    network->kernel[0]->linearisation = 0;
    add_convolution(network, 6, 28, activation);
    add_2d_average_pooling(network, 14);
    add_convolution(network, 16, 10, activation);
    add_2d_average_pooling(network, 5);
    add_dense_linearisation(network, 120, activation);
    add_dense(network, 84, activation);
    add_dense(network, 10, SOFTMAX);
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
    network->width[pos] = dim;
    network->depth[pos] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    network->input[pos] = (float***)malloc(sizeof(float**));
    network->input[pos][0] = (float**)malloc(sizeof(float*));
    network->input[pos][0][0] = (float*)malloc(sizeof(float)*dim);
    network->width[pos] = dim;
    network->depth[pos] = 1;
}

void add_2d_average_pooling(Network* network, int dim_ouput) {
    int n = network->size;
    int k_pos = n-1;
    int dim_input = network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    int kernel_size = dim_input/dim_ouput;
    if (dim_input%dim_ouput != 0) {
        printf("Erreur de dimension dans l'average pooling\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = NULL;
    network->kernel[k_pos]->activation = 100*kernel_size; // Ne contient pas de fonction d'activation
    create_a_cube_input_layer(network, n, network->depth[n-1], network->width[n-1]/2);
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
    network->kernel[k_pos]->cnn = (Kernel_cnn*)malloc(sizeof(Kernel_cnn));
    Kernel_cnn* cnn = network->kernel[k_pos]->cnn;

    cnn->k_size = kernel_size;
    cnn->rows = depth_input;
    cnn->columns = depth_output;
    cnn->w = (float****)malloc(sizeof(float***)*depth_input);
    cnn->d_w = (float****)malloc(sizeof(float***)*depth_input);
    cnn->last_d_w = (float****)malloc(sizeof(float***)*depth_input);
    for (int i=0; i < depth_input; i++) {
        cnn->w[i] = (float***)malloc(sizeof(float**)*depth_output);
        cnn->d_w[i] = (float***)malloc(sizeof(float**)*depth_output);
        cnn->last_d_w[i] = (float***)malloc(sizeof(float**)*depth_output);
        for (int j=0; j < depth_output; j++) {
            cnn->w[i][j] = (float**)malloc(sizeof(float*)*kernel_size);
            cnn->d_w[i][j] = (float**)malloc(sizeof(float*)*kernel_size);
            cnn->last_d_w[i][j] = (float**)malloc(sizeof(float*)*kernel_size);
            for (int k=0; k < kernel_size; k++) {
                cnn->w[i][j][k] = (float*)malloc(sizeof(float)*kernel_size);
                cnn->d_w[i][j][k] = (float*)malloc(sizeof(float)*kernel_size);
                cnn->last_d_w[i][j][k] = (float*)malloc(sizeof(float)*kernel_size);
            }
        }
    }
    cnn->bias = (float***)malloc(sizeof(float**)*depth_output);
    cnn->d_bias = (float***)malloc(sizeof(float**)*depth_output);
    cnn->last_d_bias = (float***)malloc(sizeof(float**)*depth_output);
    for (int i=0; i < depth_output; i++) {
        cnn->bias[i] = (float**)malloc(sizeof(float*)*bias_size);
        cnn->d_bias[i] = (float**)malloc(sizeof(float*)*bias_size);
        cnn->last_d_bias[i] = (float**)malloc(sizeof(float*)*bias_size);
        for (int j=0; j < bias_size; j++) {
            cnn->bias[i][j] = (float*)malloc(sizeof(float)*bias_size);
            cnn->d_bias[i][j] = (float*)malloc(sizeof(float)*bias_size);
            cnn->last_d_bias[i][j] = (float*)malloc(sizeof(float)*bias_size);
        }
    }
    create_a_cube_input_layer(network, n, depth_output, bias_size);
    int n_int = network->width[n-1]*network->width[n-1]*network->depth[n-1];
    int n_out = network->width[n]*network->width[n]*network->depth[n];
    /* Not currently used 
    initialisation_3d_matrix(network->initialisation, cnn->bias, depth_output, kernel_size, kernel_size, n_int+n_out);
    initialisation_3d_matrix(ZERO, cnn->d_bias, depth_output, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(network->initialisation, cnn->w, depth_input, depth_output, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(ZERO, cnn->d_w, depth_input, depth_output, kernel_size, kernel_size, n_int+n_out);
    */
    network->size++;
}

void add_dense(Network* network, int output_units, int activation) {
    int n = network->size;
    int k_pos = n-1;
    int input_units = network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = (Kernel_nn*)malloc(sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    network->kernel[k_pos]->activation = activation;
    nn->input_units = input_units;
    nn->output_units = output_units;
    nn->bias = (float*)malloc(sizeof(float)*output_units);
    nn->d_bias = (float*)malloc(sizeof(float)*output_units);
    nn->last_d_bias = (float*)malloc(sizeof(float)*output_units);
    nn->weights = (float**)malloc(sizeof(float*)*input_units);
    nn->d_weights = (float**)malloc(sizeof(float*)*input_units);
    nn->last_d_weights = (float**)malloc(sizeof(float*)*input_units);
    for (int i=0; i < input_units; i++) {
        nn->weights[i] = (float*)malloc(sizeof(float)*output_units);
        nn->d_weights[i] = (float*)malloc(sizeof(float)*output_units);
        nn->last_d_weights[i] = (float*)malloc(sizeof(float)*output_units);
    }
    create_a_line_input_layer(network, n, output_units);
    /* Not currently used
    initialisation_1d_matrix(network->initialisation, nn->bias, output_units, output_units+input_units);
    initialisation_1d_matrix(ZERO, nn->d_bias, output_units, output_units+input_units);
    initialisation_2d_matrix(network->initialisation, nn->weights, input_units, output_units, output_units+input_units);
    initialisation_2d_matrix(ZERO, nn->d_weights, input_units, output_units, output_units+input_units);
    */
    network->size++;
}

void add_dense_linearisation(Network* network, int output_units, int activation) {
    // Can replace input_units by a research of this dim

    int n = network->size;
    int k_pos = n-1;
    int input_units = network->depth[k_pos]*network->width[k_pos]*network->width[k_pos];
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[k_pos]->cnn = NULL;
    network->kernel[k_pos]->nn = (Kernel_nn*)malloc(sizeof(Kernel_nn));
    Kernel_nn* nn = network->kernel[k_pos]->nn;
    network->kernel[k_pos]->activation = activation;
    nn->input_units = input_units;
    nn->output_units = output_units;
    
    nn->bias = (float*)malloc(sizeof(float)*output_units);
    nn->d_bias = (float*)malloc(sizeof(float)*output_units);
    nn->last_d_bias = (float*)malloc(sizeof(float)*output_units);
    nn->weights = (float**)malloc(sizeof(float*)*input_units);
    nn->d_weights = (float**)malloc(sizeof(float*)*input_units);
    nn->last_d_weights = (float**)malloc(sizeof(float*)*input_units);
    for (int i=0; i < input_units; i++) {
        nn->weights[i] = (float*)malloc(sizeof(float)*output_units);
        nn->d_weights[i] = (float*)malloc(sizeof(float)*output_units);
        nn->last_d_weights[i] = (float*)malloc(sizeof(float)*output_units);
    }
    /* Not currently used
    initialisation_1d_matrix(network->initialisation, nn->bias, output_units, output_units+input_units);
    initialisation_1d_matrix(ZERO, nn->d_bias, output_units, output_units+input_units);
    initialisation_2d_matrix(network->initialisation, nn->weights, input_units, output_units, output_units+input_units);
    initialisation_2d_matrix(ZERO, nn->d_weights, input_units, output_units, output_units+input_units); */
    create_a_line_input_layer(network, n, output_units);

    network->size++;
}