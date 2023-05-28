#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "../common/include/memory_management.h"
#include "include/cnn.h"

#include "include/free.h"

void free_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim; j++) {
            gree(network->input[pos][i][j], true);
            gree(network->input_z[pos][i][j], true);
        }
        gree(network->input[pos][i], true);
        gree(network->input_z[pos][i], true);
    }
    gree(network->input[pos], true);
    gree(network->input_z[pos], true);
}

void free_a_line_input_layer(Network* network, int pos) {
    // Libère l'espace mémoire de network->input[pos] et network->input_z[pos]
    // lorsque ces couches sont denses (donc sont des matrice de dimension 1)
    gree(network->input[pos][0][0], true);
    gree(network->input_z[pos][0][0], true);
    gree(network->input[pos][0], true);
    gree(network->input_z[pos][0], true);
    gree(network->input[pos], true);
    gree(network->input_z[pos], true);
}

void free_pooling(Network* network, int pos) {
    // Le pooling n'alloue rien d'autre que l'input
    free_a_cube_input_layer(network, pos+1, network->depth[pos+1], network->width[pos+1]);
}

void free_convolution(Network* network, int pos) {
    Kernel_cnn* k_pos = network->kernel[pos]->cnn;
    int c = k_pos->columns;
    int k_size = k_pos->k_size;
    int r = k_pos->rows;
    int bias_size = network->width[pos+1];
    free_a_cube_input_layer(network, pos+1, network->depth[pos+1], network->width[pos+1]);

    // Partie toujours initialisée (donc à libérer)
    for (int i=0; i < c; i++) {
        for (int j=0; j < bias_size; j++) {
            gree(k_pos->bias[i][j], true);
        }
        gree(k_pos->bias[i], true);
    }
    gree(k_pos->bias, true);

    for (int i=0; i < r; i++) {
        for (int j=0; j < c; j++) {
            for (int k=0; k < k_size; k++) {
                gree(k_pos->weights[i][j][k], true);
            }
            gree(k_pos->weights[i][j], true);
        }
        gree(k_pos->weights[i], true);
    }
    gree(k_pos->weights, true);

    gree(k_pos, true);
}

void free_dense(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->size_input;

    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i], true);
    }
    gree(k_pos->weights, true);
    gree(k_pos->bias, true);

    gree(k_pos, true);
}

void free_dense_linearisation(Network* network, int pos) {
    free_a_line_input_layer(network, pos+1);
    Kernel_nn* k_pos = network->kernel[pos]->nn;
    int dim = k_pos->size_input;

    for (int i=0; i < dim; i++) {
        gree(k_pos->weights[i], true);
    }
    gree(k_pos->weights, true);
    gree(k_pos->bias, true);

    gree(k_pos, true);
}

void free_network_creation(Network* network) {
    // On libère l'input correspondant à l'image: input[0] (car elle n'appartient à aucune couche)
    free_a_cube_input_layer(network, 0, network->depth[0], network->width[0]);

    for (int i=0; i < network->max_size-1; i++) {
        gree(network->kernel[i], true);
    }
    gree(network->width, true);
    gree(network->depth, true);
    gree(network->kernel, true);
    gree(network->input, true);
    gree(network->input_z, true);

    gree(network, true);
}


void free_network(Network* network) {
    #if (defined(USE_CUDA) || defined(TEST_MEMORY_MANAGEMENT)) && defined(FREE_ALL_OPT)
        // Supprimer toute la mémoire allouée avec nalloc directement
        // Il n'y a alors plus besoin de parcourir tout le réseau,
        // mais il faut que TOUTE la mémoire du réseau ait été allouée de cette manière
        // et que cela soit le cas UNIQUEMENT pour la mémoire allouée au réseau

        // Représente un gain de 45mn sur VGG16
        (void)network;
        free_all_memory();
    #else
        for (int i=network->size-2; i>=0; i--) {
            if (network->kernel[i]->cnn != NULL) {
                // Convolution
                free_convolution(network, i);
            } 
            else if (network->kernel[i]->nn != NULL) {
                // Dense
                if (network->kernel[i]->linearisation == DOESNT_LINEARISE) {
                    // Dense normale
                    free_dense(network, i);
                } else {
                    // Dense qui linéarise
                    free_dense_linearisation(network, i);
                }
            } else {
                // Pooling
                free_pooling(network, i);
            }
        }
        free_network_creation(network);
    #endif
}

// ----------------------- Pour le d_network -----------------------

void free_d_convolution(Network* network, int pos) {
    Kernel_cnn* k_pos = network->kernel[pos]->cnn;
    D_Network* d_network = network->d_network;
    D_Kernel_cnn* d_k_pos = d_network->kernel[pos]->cnn;
    int c = k_pos->columns;
    int k_size = k_pos->k_size;
    int r = k_pos->rows;
    int bias_size = network->width[pos+1];

    if (network->finetuning == EVERYTHING) {
        for (int i=0; i < c; i++) {
            for (int j=0; j < bias_size; j++) {
                gree(d_k_pos->d_bias[i][j], true);
                #ifdef ADAM_CNN_BIAS
                gree(d_k_pos->s_d_bias[i][j], true);
                gree(d_k_pos->v_d_bias[i][j], true);
                #endif
            }
            gree(d_k_pos->d_bias[i], true);
            #ifdef ADAM_CNN_BIAS
            gree(d_k_pos->s_d_bias[i], true);
            gree(d_k_pos->v_d_bias[i], true);
            #endif
        }
        gree(d_k_pos->d_bias, true);
        #ifdef ADAM_CNN_BIAS
        gree(d_k_pos->s_d_bias, true);
        gree(d_k_pos->v_d_bias, true);
        #endif

        for (int i=0; i < r; i++) {
            for (int j=0; j < c; j++) {
                for (int k=0; k < k_size; k++) {
                    gree(d_k_pos->d_weights[i][j][k], true);
                    #ifdef ADAM_CNN_WEIGHTS
                    gree(d_k_pos->s_d_weights[i][j][k], true);
                    gree(d_k_pos->v_d_weights[i][j][k], true);
                    #endif
                }
                gree(d_k_pos->d_weights[i][j], true);
                #ifdef ADAM_CNN_WEIGHTS
                gree(d_k_pos->s_d_weights[i][j], true);
                gree(d_k_pos->v_d_weights[i][j], true);
                #endif
            }
            gree(d_k_pos->d_weights[i], true);
            #ifdef ADAM_CNN_WEIGHTS
            gree(d_k_pos->s_d_weights[i], true);
            gree(d_k_pos->v_d_weights[i], true);
            #endif
        }
        gree(d_k_pos->d_weights, true);
        #ifdef ADAM_CNN_WEIGHTS
        gree(d_k_pos->s_d_weights, true);
        gree(d_k_pos->v_d_weights, true);
        #endif
    }
}

void free_d_dense(Network* network, int pos) {
    D_Network* d_network = network->d_network;
    D_Kernel_nn* d_k_pos = d_network->kernel[pos]->nn;
    int dim = network->kernel[pos]->nn->size_input;
    for (int i=0; i < dim; i++) {
        gree(d_k_pos->d_weights[i], true);
        #ifdef ADAM_DENSE_WEIGHTS
        gree(d_k_pos->s_d_weights[i], true);
        gree(d_k_pos->v_d_weights[i], true);
        #endif
    }
    gree(d_k_pos->d_weights, true);
    #ifdef ADAM_DENSE_WEIGHTS
    gree(d_k_pos->s_d_weights, true);
    gree(d_k_pos->v_d_weights, true);
    #endif

    gree(d_k_pos->d_bias, true);
    #ifdef ADAM_DENSE_BIAS
    gree(d_k_pos->s_d_bias, true);
    gree(d_k_pos->v_d_bias, true);
    #endif
}

void free_d_dense_linearisation(Network* network, int pos) {
    D_Network* d_network = network->d_network;
    D_Kernel_nn* d_k_pos = d_network->kernel[pos]->nn;
    int dim = network->kernel[pos]->nn->size_input;

    if (network->finetuning <= NN_AND_LINEARISATION) {
        for (int i=0; i < dim; i++) {
            gree(d_k_pos->d_weights[i], true);
            #ifdef ADAM_DENSE_WEIGHTS
            gree(d_k_pos->s_d_weights[i], true);
            gree(d_k_pos->v_d_weights[i], true);
            #endif
        }
        gree(d_k_pos->d_weights, true);
        #ifdef ADAM_DENSE_WEIGHTS
        gree(d_k_pos->s_d_weights, true);
        gree(d_k_pos->v_d_weights, true);
        #endif

        gree(d_k_pos->d_bias, true);
        #ifdef ADAM_DENSE_BIAS
        gree(d_k_pos->s_d_bias, true);
        gree(d_k_pos->v_d_bias, true);
        #endif
    }

    gree(d_k_pos, true);
}

void free_d_network(Network* network) {
    D_Network* d_network = network->d_network;
    for (int i=0; i < network->max_size-1; i++) {
        D_Kernel* d_k_i = d_network->kernel[i];
        if (d_k_i->cnn) { // Convolution
            free_d_convolution(network, i);
        } else if (d_k_i->nn) { // Dense
            if (network->kernel[i]->linearisation == DOESNT_LINEARISE) { // Vecteur -> Vecteur
                free_d_dense(network, i);
            } else { // Matrice -> Vecteur
                free_d_dense_linearisation(network, i);
            }
        }
        gree(network->kernel[i], true);
    }
    gree(network->kernel, true);
    pthread_mutex_destroy(&(d_network->lock));
    gree(network, true);
}