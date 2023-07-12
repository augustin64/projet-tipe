#include <stdio.h>
#include <math.h>
#include <float.h>

#include "include/update.h"
#include "include/struct.h"
#include "include/cnn.h"

#include "include/config.h"

float clip(float a) {
    if (a > NETWORK_CLIP_VALUE) {
        return NETWORK_CLIP_VALUE;
    }
    if (a < -NETWORK_CLIP_VALUE) {
        return -NETWORK_CLIP_VALUE;
    }
    return a;
}

void update_weights(Network* network) {
    int n = network->size;
    D_Network* d_network = network->d_network;
    pthread_mutex_lock(&(d_network->lock));

    for (int i=0; i < (n-1); i++) {
        Kernel* k_i = network->kernel[i];
        D_Kernel* d_k_i = d_network->kernel[i];

        int input_depth = network->depth[i];
        int input_width = network->width[i];

        int output_depth = network->depth[i+1];
        int output_width = network->width[i+1];

        if (k_i->cnn) { // Convolution
            if (network->finetuning != EVERYTHING) {
                return; // Alors on a finit de backpropager
            }
            Kernel_cnn* cnn = k_i->cnn;
            D_Kernel_cnn* d_cnn = d_k_i->cnn;
            int k_size = cnn->k_size;
            for (int a=0; a < input_depth; a++) {
                for (int b=0; b < output_depth; b++) {
                    for (int c=0; c < k_size; c++) {
                        for (int d=0; d < k_size; d++) {
                            #ifdef ADAM_CNN_WEIGHTS
                            d_cnn->v_d_weights[a][b][c][d] = BETA_1*d_cnn->v_d_weights[a][b][c][d] + (1-BETA_1)*d_cnn->d_weights[a][b][c][d];
                            d_cnn->s_d_weights[a][b][c][d] = BETA_2*d_cnn->s_d_weights[a][b][c][d] + (1-BETA_2)*d_cnn->d_weights[a][b][c][d]*d_cnn->d_weights[a][b][c][d];
                            cnn->weights[a][b][c][d] -= ALPHA*(d_cnn->v_d_weights[a][b][c][d]/sqrt(d_cnn->s_d_weights[a][b][c][d]+Epsilon));
                            #else
                            cnn->weights[a][b][c][d] -= network->learning_rate * d_cnn->d_weights[a][b][c][d];
                            #endif
                            d_cnn->d_weights[a][b][c][d] = 0;

                            cnn->weights[a][b][c][d] = clip(cnn->weights[a][b][c][d]);
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == DOESNT_LINEARISE) { // Vecteur -> Vecteur
                Kernel_nn* nn = k_i->nn;
                D_Kernel_nn* d_nn = d_k_i->nn;

                for (int a=0; a < input_width; a++) {
                    for (int b=0; b < output_width; b++) {
                        #ifdef ADAM_DENSE_WEIGHTS
                        d_nn->v_d_weights[a][b] = BETA_1*d_nn->v_d_weights[a][b] + (1-BETA_1)*d_nn->d_weights[a][b];
                        d_nn->s_d_weights[a][b] = BETA_2*d_nn->s_d_weights[a][b] + (1-BETA_2)*d_nn->d_weights[a][b]*d_nn->d_weights[a][b];
                        nn->weights[a][b] -= ALPHA*(d_nn->v_d_weights[a][b]/sqrt(d_nn->s_d_weights[a][b]+Epsilon));
                        #else
                        nn->weights[a][b] -= network->learning_rate * d_nn->d_weights[a][b];
                        #endif
                        d_nn->d_weights[a][b] = 0;
                    }
                }
            } else { // Matrice -> vecteur
                if (network->finetuning == NN_ONLY) {
                    return; // Alors on a finit de backpropager
                }
                Kernel_nn* nn = k_i->nn;
                D_Kernel_nn* d_nn = d_k_i->nn;

                int size_input = input_width*input_width*input_depth;

                for (int a=0; a < size_input; a++) {
                    for (int b=0; b < output_width; b++) {
                        #ifdef ADAM_DENSE_WEIGHTS
                        d_nn->v_d_weights[a][b] = BETA_1*d_nn->v_d_weights[a][b] + (1-BETA_1)*d_nn->d_weights[a][b];
                        d_nn->s_d_weights[a][b] = BETA_2*d_nn->s_d_weights[a][b] + (1-BETA_2)*d_nn->d_weights[a][b]*d_nn->d_weights[a][b];
                        nn->weights[a][b] -= ALPHA*(d_nn->v_d_weights[a][b]/sqrt(d_nn->s_d_weights[a][b]+Epsilon));
                        #else
                        nn->weights[a][b] -= network->learning_rate * d_nn->d_weights[a][b];
                        #endif
                        d_nn->d_weights[a][b] = 0;

                        nn->weights[a][b] = clip(nn->weights[a][b]);
                    }
                }
            }
        }
        // Une couche de pooling ne nécessite pas de traitement
    }
    pthread_mutex_unlock(&(d_network->lock));
}

void update_bias(Network* network) {
    int n = network->size;
    D_Network* d_network = network->d_network;

    for (int i=0; i < (n-1); i++) {
        Kernel* k_i = network->kernel[i];
        D_Kernel* d_k_i = d_network->kernel[i];
        int output_width = network->width[i+1];
        int output_depth = network->depth[i+1];

        if (k_i->cnn) { // Convolution
            if (network->finetuning != EVERYTHING) {
                return; // Alors on a finit de backpropager
            }
            Kernel_cnn* cnn = k_i->cnn;
            D_Kernel_cnn* d_cnn = d_k_i->cnn;

            for (int a=0; a < output_depth; a++) {
                for (int b=0; b < output_width; b++) {
                    for (int c=0; c < output_width; c++) {
                        #ifdef ADAM_CNN_BIAS
                        d_cnn->v_d_bias[a][b][c] = BETA_1*d_cnn->v_d_bias[a][b][c] + (1-BETA_1)*d_cnn->d_bias[a][b][c];
                        d_cnn->s_d_bias[a][b][c] = BETA_2*d_cnn->s_d_bias[a][b][c] + (1-BETA_2)*d_cnn->d_bias[a][b][c]*d_cnn->d_bias[a][b][c];
                        cnn->bias[a][b][c] -= ALPHA*(d_cnn->v_d_bias[a][b][c]/sqrt(d_cnn->s_d_bias[a][b][c]+Epsilon));
                        #else
                        cnn->bias[a][b][c] -= network->learning_rate * d_cnn->d_bias[a][b][c];
                        #endif
                        d_cnn->d_bias[a][b][c] = 0;

                        cnn->bias[a][b][c] = clip(cnn->bias[a][b][c]);
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == DO_LINEARISE) {// Matrice -> vecteur
                if (network->finetuning == NN_ONLY) {
                    return; // Alors on a finit de backpropager
                }
            }
            Kernel_nn* nn = k_i->nn;
            D_Kernel_nn* d_nn = d_k_i->nn;

            for (int a=0; a < output_width; a++) {
                #ifdef ADAM_DENSE_BIAS
                d_nn->v_d_bias[a] = BETA_1*d_nn->v_d_bias[a] + (1-BETA_1)*d_nn->d_bias[a];
                d_nn->s_d_bias[a] = BETA_2*d_nn->s_d_bias[a] + (1-BETA_2)*d_nn->d_bias[a]*d_nn->d_bias[a];
                nn->bias[a] -= ALPHA*(d_nn->v_d_bias[a]/sqrt(d_nn->s_d_bias[a]+Epsilon));
                #else
                nn->bias[a] -= network->learning_rate * d_nn->d_bias[a];
                #endif
                d_nn->d_bias[a] = 0;

                nn->bias[a] = clip(nn->bias[a]);
            }
        }
        // Une couche de pooling ne nécessite pas de traitement
    }
}

void reset_d_weights(Network* network) {
    int n = network->size;
    D_Network* d_network = network->d_network;

    for (int i=0; i < (n-1); i++) {
        Kernel* k_i = network->kernel[i];
        Kernel* k_i_1 = network->kernel[i+1];
        D_Kernel* d_k_i_1 = d_network->kernel[i+1];
        
        int input_depth = network->depth[i];
        int input_width = network->width[i];
        
        int output_depth = network->depth[i+1];
        int output_width = network->width[i+1];

        if (k_i->cnn) { // Convolution
            if (network->finetuning != EVERYTHING) {
                continue; // On n'a pas initialisé donc on n'a pas besoin de reset
            }
            D_Kernel_cnn* d_cnn = d_k_i_1->cnn;

            int k_size = k_i_1->cnn->k_size;

            for (int a=0; a < input_depth; a++) {
                for (int b=0; b < output_depth; b++) {
                    for (int c=0; c < k_size; c++) {
                        for (int d=0; d < k_size; d++) {
                            d_cnn->d_weights[a][b][c][d] = 0;
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == DOESNT_LINEARISE) { // Vecteur -> Vecteur
                D_Kernel_nn* d_nn = d_k_i_1->nn;

                for (int a=0; a < input_width; a++) {
                    for (int b=0; b < output_width; b++) {
                        d_nn->d_weights[a][b] = 0;
                    }
                }
            } else { // Matrice -> vecteur
                if (network->finetuning == NN_ONLY) {
                    continue; // On n'a pas initialisé donc on n'a pas besoin de reset
                }
                D_Kernel_nn* d_nn = d_k_i_1->nn;

                int size_input = input_width*input_width*input_depth;

                for (int a=0; a < size_input; a++) {
                    for (int b=0; b < output_width; b++) {
                        d_nn->d_weights[a][b] = 0;
                    }
                }
            }
        }
        // Une couche de pooling ne nécessite pas de traitement
    }
}

void reset_d_bias(Network* network) {
    int n = network->size;
    D_Network* d_network = network->d_network;
    
    for (int i=0; i < (n-1); i++) {
        Kernel* k_i = network->kernel[i];
        D_Kernel* d_k_i_1 = d_network->kernel[i+1];
        
        int output_width = network->width[i+1];
        int output_depth = network->depth[i+1];

        if (k_i->cnn) { // Convolution
            if (network->finetuning != EVERYTHING) {
                continue; // On n'a pas initialisé donc on n'a pas besoin de reset
            }
            D_Kernel_cnn* d_cnn = d_k_i_1->cnn;

            for (int a=0; a < output_depth; a++) {
                for (int b=0; b < output_width; b++) {
                    for (int c=0; c < output_width; c++) {
                        d_cnn->d_bias[a][b][c] = 0;
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (k_i->linearisation == DO_LINEARISE) {
                if (network->finetuning == NN_ONLY) {
                    continue; // On n'a pas initialisé donc on n'a pas besoin de reset
                }
            }
            D_Kernel_nn* d_nn = d_k_i_1->nn;

            for (int a=0; a < output_width; a++) {
                d_nn->d_bias[a] = 0;
            }
        }
        // Une couche de pooling ne nécessite pas de traitement
    }
}