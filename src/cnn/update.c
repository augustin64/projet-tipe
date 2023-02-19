#include <stdio.h>

#include "include/update.h"
#include "include/struct.h"

void update_weights(Network* network, Network* d_network) {
    int n = network->size;
    int input_depth, input_width, output_depth, output_width, k_size;
    Kernel* k_i;
    Kernel* dk_i;
    for (int i=0; i<(n-1); i++) {
        k_i = network->kernel[i];
        dk_i = d_network->kernel[i];
        input_depth = network->depth[i];
        input_width = network->width[i];
        output_depth = network->depth[i+1];
        output_width = network->width[i+1];

        if (k_i->cnn) { // Convolution
            Kernel_cnn* cnn = k_i->cnn;
            Kernel_cnn* d_cnn = dk_i->cnn;
            k_size = cnn->k_size;
            for (int a=0; a<input_depth; a++) {
                for (int b=0; b<output_depth; b++) {
                    for (int c=0; c<k_size; c++) {
                        for (int d=0; d<k_size; d++) {
                            cnn->weights[a][b][c][d] -= network->learning_rate * d_cnn->d_weights[a][b][c][d];
                            d_cnn->d_weights[a][b][c][d] = 0;

                            if (cnn->weights[a][b][c][d] > MAX_RESEAU)
                                cnn->weights[a][b][c][d] = MAX_RESEAU;
                            else if (cnn->weights[a][b][c][d] < -MAX_RESEAU)
                                cnn->weights[a][b][c][d] = -MAX_RESEAU;
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == 0) { // Vecteur -> Vecteur
                Kernel_nn* nn = k_i->nn;
                Kernel_nn* d_nn = dk_i->nn;
                for (int a=0; a<input_width; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->weights[a][b] -= network->learning_rate * d_nn->d_weights[a][b];
                        d_nn->d_weights[a][b] = 0;
                    }
                }
            } else { // Matrice -> vecteur
                Kernel_nn* nn = k_i->nn;
                Kernel_nn* d_nn = dk_i->nn;
                int input_size = input_width*input_width*input_depth;
                for (int a=0; a<input_size; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->weights[a][b] -= network->learning_rate * d_nn->d_weights[a][b];
                        d_nn->d_weights[a][b] = 0;

                        if (nn->weights[a][b] > MAX_RESEAU)
                            nn->weights[a][b] = MAX_RESEAU;
                        else if (nn->weights[a][b] < -MAX_RESEAU)
                            nn->weights[a][b] = -MAX_RESEAU;
                    }
                }
            }
        } else { // Pooling
            (void)0; // Ne rien faire pour la couche pooling
        }
    }
}

void update_bias(Network* network, Network* d_network) {

    int n = network->size;
    int output_width, output_depth;
    Kernel* k_i;
    Kernel* dk_i;
    for (int i=0; i<(n-1); i++) {
        k_i = network->kernel[i];
        dk_i = d_network->kernel[i];
        output_width = network->width[i+1];
        output_depth = network->depth[i+1];

        if (k_i->cnn) { // Convolution
            Kernel_cnn* cnn = k_i->cnn;
            Kernel_cnn* d_cnn = dk_i->cnn;
            for (int a=0; a<output_depth; a++) {
                for (int b=0; b<output_width; b++) {
                    for (int c=0; c<output_width; c++) {
                        cnn->bias[a][b][c] -= network->learning_rate * d_cnn->d_bias[a][b][c];
                        d_cnn->d_bias[a][b][c] = 0;

                        if (cnn->bias[a][b][c] > MAX_RESEAU)
                            cnn->bias[a][b][c] = MAX_RESEAU;
                        else if (cnn->bias[a][b][c] < -MAX_RESEAU)
                            cnn->bias[a][b][c] = -MAX_RESEAU;
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            Kernel_nn* nn = k_i->nn;
            Kernel_nn* d_nn = dk_i->nn;
            for (int a=0; a<output_width; a++) {
                nn->bias[a] -= network->learning_rate * d_nn->d_bias[a];
                d_nn->d_bias[a] = 0;

                if (nn->bias[a] > MAX_RESEAU)
                    nn->bias[a] = MAX_RESEAU;
                else if (nn->bias[a] < -MAX_RESEAU)
                    nn->bias[a] = -MAX_RESEAU;
            }
        } else { // Pooling
            (void)0; // Ne rien faire pour la couche pooling
        }
    }
}

void reset_d_weights(Network* network) {
    int n = network->size;
    int input_depth, input_width, output_depth, output_width;
    Kernel* k_i;
    Kernel* k_i_1;
    for (int i=0; i<(n-1); i++) {
        k_i = network->kernel[i];
        k_i_1 = network->kernel[i+1];
        input_depth = network->depth[i];
        input_width = network->width[i];
        output_depth = network->depth[i+1];
        output_width = network->width[i+1];

        if (k_i->cnn) { // Convolution
            Kernel_cnn* cnn = k_i_1->cnn;
            int k_size = cnn->k_size;
            for (int a=0; a<input_depth; a++) {
                for (int b=0; b<output_depth; b++) {
                    for (int c=0; c<k_size; c++) {
                        for (int d=0; d<k_size; d++) {
                            cnn->d_weights[a][b][c][d] = 0;
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == 0) { // Vecteur -> Vecteur
                Kernel_nn* nn = k_i_1->nn;
                for (int a=0; a<input_width; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->d_weights[a][b] = 0;
                    }
                }
            } else { // Matrice -> vecteur
                Kernel_nn* nn = k_i_1->nn;
                int input_size = input_width*input_width*input_depth;
                for (int a=0; a<input_size; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->d_weights[a][b] = 0;
                    }
                }
            }
        } else { // Pooling
            (void)0; // Ne rien faire pour la couche pooling
        }
    }
}

void reset_d_bias(Network* network) {
    int n = network->size;
    int output_width, output_depth;
    Kernel* k_i;
    Kernel* k_i_1;
    for (int i=0; i<(n-1); i++) {
        k_i = network->kernel[i];
        k_i_1 = network->kernel[i+1];
        output_width = network->width[i+1];
        output_depth = network->depth[i+1];

        if (k_i->cnn) { // Convolution
            Kernel_cnn* cnn = k_i_1->cnn;
            for (int a=0; a<output_depth; a++) {
                for (int b=0; b<output_width; b++) {
                    for (int c=0; c<output_width; c++) {
                        cnn->d_bias[a][b][c] = 0;
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            Kernel_nn* nn = k_i_1->nn;
            for (int a=0; a<output_width; a++) {
                nn->d_bias[a] = 0;
            }
        } else { // Pooling
            (void)0; // Ne rien faire pour la couche pooling
        }
    }
}