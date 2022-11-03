
#include "include/update.h"
#include "include/struct.h"

void update_weights(Network* network) {
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
                            cnn->w[a][b][c][d] += cnn->d_w[a][b][c][d];
                            cnn->d_w[a][b][c][d] = 0;
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (input_depth==1) { // Vecteur -> Vecteur
                Kernel_nn* nn = k_i_1->nn;
                for (int a=0; a<input_width; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->weights[a][b] += nn->d_weights[a][b];
                        nn->d_weights[a][b] = 0;
                    }
                }
            } else { // Matrice -> vecteur
                Kernel_nn* nn = k_i_1->nn;
                int input_size = input_width*input_width*input_depth;
                for (int a=0; a<input_size; a++) {
                    for (int b=0; b<output_width; b++) {
                        nn->weights[a][b] += nn->d_weights[a][b];
                        nn->d_weights[a][b] = 0;
                    }
                }
            }
        } else { // Pooling
            (void)0; // Ne rien faire pour la couche pooling
        }
    }
}

void update_bias(Network* network) {
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
                        cnn->bias[a][b][c] += cnn->d_bias[a][b][c];
                        cnn->d_bias[a][b][c] = 0;
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            Kernel_nn* nn = k_i_1->nn;
            for (int a=0; a<output_width; a++) {
                nn->bias[a] += nn->d_bias[a];
                nn->d_bias[a] = 0;
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
                            cnn->d_w[a][b][c][d] = 0;
                        }
                    }
                }
            }
        } else if (k_i->nn) { // Full connection
            if (input_depth==1) { // Vecteur -> Vecteur
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