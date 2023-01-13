#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../include/colors.h"
#include "include/neuron_io.h"
#include "include/struct.h"

#define MAGIC_NUMBER 1012

#define bufferAdd(val) {buffer[indice_buffer] = val; indice_buffer++;}

void write_network(char* filename, Network* network) {
    FILE *ptr;
    int size = network->size;
    int type_couche[size];
    int indice_buffer = 0;

    ptr = fopen(filename, "wb");

    uint32_t buffer[(network->size)*3+4];

    bufferAdd(MAGIC_NUMBER);
    bufferAdd(size);
    bufferAdd(network->initialisation);
    bufferAdd(network->dropout);

    // Écriture du header
    for (int i=0; i < size; i++) {
        bufferAdd(network->width[i]);
        bufferAdd(network->depth[i]);
    }

    for (int i=0; i < size; i++) {
        if ((!network->kernel[i]->cnn)&&(!network->kernel[i]->nn)) {
            type_couche[i] = 2;
        } else if (!network->kernel[i]->cnn) {
            type_couche[i] = 1;
        } else {
            type_couche[i] = 0;
        }
        bufferAdd(type_couche[i]);
    }

    fwrite(buffer, sizeof(buffer), 1, ptr);

    // Écriture du pré-corps et corps
    for (int i=0; i < size; i++) {
        write_couche(network, i, type_couche[i], ptr);
    }

    fclose(ptr);
}


void write_couche(Network* network, int indice_couche, int type_couche, FILE* ptr) {
    Kernel* kernel = network->kernel[indice_couche];
    int indice_buffer = 0;
    if (type_couche == 0) { // Cas du CNN
        Kernel_cnn* cnn = kernel->cnn;
        int output_dim = network->width[indice_couche+1];

        // Écriture du pré-corps
        uint32_t pre_buffer[5];
        pre_buffer[0] = kernel->activation;
        pre_buffer[1] = kernel->linearisation;
        pre_buffer[2] = cnn->k_size;
        pre_buffer[3] = cnn->rows;
        pre_buffer[4] = cnn->columns;
        fwrite(pre_buffer, sizeof(pre_buffer), 1, ptr);

        // Écriture du corps
        float buffer[cnn->columns*(cnn->k_size*cnn->k_size*cnn->rows+output_dim*output_dim)];

        for (int i=0; i < cnn->columns; i++) {
            for (int j=0; j < output_dim; j++) {
                for (int k=0; k < output_dim; k++) {
                    bufferAdd(cnn->bias[i][j][k]);
                }
            }
        }
        for (int i=0; i < cnn->rows; i++) {
            for (int j=0; j < cnn->columns; j++) {
                for (int k=0; k < cnn->k_size; k++) {
                    for (int l=0; l < cnn->k_size; l++) {
                        bufferAdd(cnn->w[i][j][k][l]);
                    }
                }
            }
        }
        fwrite(buffer, sizeof(buffer), 1, ptr);
    } else if (type_couche == 1) { // Cas du NN
        Kernel_nn* nn = kernel->nn;

        // Écriture du pré-corps
        uint32_t pre_buffer[4];
        pre_buffer[0] = kernel->activation;
        pre_buffer[1] = kernel->linearisation;
        pre_buffer[2] = nn->input_units;
        pre_buffer[3] = nn->output_units;
        fwrite(pre_buffer, sizeof(pre_buffer), 1, ptr);

        // Écriture du corps
        float buffer[(1+nn->input_units)*nn->output_units];
        for (int i=0; i < nn->output_units; i++) {
            bufferAdd(nn->bias[i]);
        }
        for (int i=0; i < nn->input_units; i++) {
            for (int j=0; j < nn->output_units; j++) {
                bufferAdd(nn->weights[i][j]);
            }
        }
        fwrite(buffer, sizeof(buffer), 1, ptr);
    } else if (type_couche == 2) { // Cas du Pooling Layer
        uint32_t pre_buffer[2];
        pre_buffer[0] = kernel->activation; // Variable du pooling
        pre_buffer[1] = kernel->linearisation;
        fwrite(pre_buffer, sizeof(pre_buffer), 1, ptr);
    }
}


Network* read_network(char* filename) {
    FILE *ptr;
    Network* network = (Network*)malloc(sizeof(Network));

    ptr = fopen(filename, "rb");

    uint32_t magic;
    uint32_t size;
    uint32_t initialisation;
    uint32_t dropout;
    uint32_t tmp;

    fread(&magic, sizeof(uint32_t), 1, ptr);
    if (magic != MAGIC_NUMBER) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    // Lecture des constantes du réseau
    fread(&size, sizeof(uint32_t), 1, ptr);
    network->size = size;
    network->max_size = size;
    fread(&initialisation, sizeof(uint32_t), 1, ptr);
    network->initialisation = initialisation;
    fread(&dropout, sizeof(uint32_t), 1, ptr);
    network->dropout = dropout;

    // Lecture de la taille de l'entrée des différentes matrices
    network->width = (int*)malloc(sizeof(int)*size);
    network->depth = (int*)malloc(sizeof(int)*size);

    for (int i=0; i < (int)size; i++) {
        fread(&tmp, sizeof(uint32_t), 1, ptr);
        network->width[i] = tmp;
        fread(&tmp, sizeof(uint32_t), 1, ptr);
        network->depth[i] = tmp;
    }

    // Lecture du type de chaque couche
    uint32_t type_couche[size];

    for (int i=0; i < (int)size; i++) {
        fread(&tmp, sizeof(tmp), 1, ptr);
        type_couche[i] = tmp;
    }

    // Lecture de chaque couche
    network->kernel = (Kernel**)malloc(sizeof(Kernel*)*size);

    for (int i=0; i < (int)size; i++) {
        network->kernel[i] = read_kernel(type_couche[i], network->width[i+1], ptr);
    }

    network->input = (float****)malloc(sizeof(float***)*size);
    for (int i=0; i < (int)size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network->input[i] = (float***)malloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network->input[i][j] = (float**)malloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network->input[i][j][k] = (float*)malloc(sizeof(float)*network->width[i]);
                for (int l=0; l < network->width[i]; l++) {
                    network->input[i][j][k][l] = 0.;
                }
            }
        }
    }

    network->input_z = (float****)malloc(sizeof(float***)*size);
    for (int i=0; i < (int)size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network->input_z[i] = (float***)malloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network->input_z[i][j] = (float**)malloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network->input_z[i][j][k] = (float*)malloc(sizeof(float)*network->width[i]);
                for (int l=0; l < network->width[i]; l++) {
                    network->input_z[i][j][k][l] = 0.;
                }
            }
        }
    }
    
    fclose(ptr);
    return network;
}

Kernel* read_kernel(int type_couche, int output_dim, FILE* ptr) {
    Kernel* kernel = (Kernel*)malloc(sizeof(Kernel));
    if (type_couche == 0) { // Cas du CNN
        // Lecture du "Pré-corps"
        kernel->cnn = (Kernel_cnn*)malloc(sizeof(Kernel_cnn));
        kernel->nn = NULL;
        uint32_t buffer[5];
        fread(&buffer, sizeof(buffer), 1, ptr);
        
        kernel->activation = buffer[0];
        kernel->linearisation = buffer[1];
        kernel->cnn->k_size = buffer[2];
        kernel->cnn->rows = buffer[3];
        kernel->cnn->columns = buffer[4];

        // Lecture du corps
        Kernel_cnn* cnn = kernel->cnn;
        float tmp;

        cnn->bias = (float***)malloc(sizeof(float**)*cnn->columns);
        cnn->d_bias = (float***)malloc(sizeof(float**)*cnn->columns);
        for (int i=0; i < cnn->columns; i++) {
            cnn->bias[i] = (float**)malloc(sizeof(float*)*output_dim);
            cnn->d_bias[i] = (float**)malloc(sizeof(float*)*output_dim);
            for (int j=0; j < output_dim; j++) {
                cnn->bias[i][j] = (float*)malloc(sizeof(float)*output_dim);
                cnn->d_bias[i][j] = (float*)malloc(sizeof(float)*output_dim);
                for (int k=0; k < output_dim; k++) {
                    fread(&tmp, sizeof(tmp), 1, ptr);
                    cnn->bias[i][j][k] = tmp;
                    cnn->d_bias[i][j][k] = 0.;
                }
            }
        }

        cnn->w = (float****)malloc(sizeof(float***)*cnn->rows);
        cnn->d_w = (float****)malloc(sizeof(float***)*cnn->rows);
        for (int i=0; i < cnn->rows; i++) {
            cnn->w[i] = (float***)malloc(sizeof(float**)*cnn->columns);
            cnn->d_w[i] = (float***)malloc(sizeof(float**)*cnn->columns);
            for (int j=0; j < cnn->columns; j++) {
                cnn->w[i][j] = (float**)malloc(sizeof(float*)*cnn->k_size);
                cnn->d_w[i][j] = (float**)malloc(sizeof(float*)*cnn->k_size);
                for (int k=0; k < cnn->k_size; k++) {
                    cnn->w[i][j][k] = (float*)malloc(sizeof(float)*cnn->k_size);
                    cnn->d_w[i][j][k] = (float*)malloc(sizeof(float)*cnn->k_size);
                    for (int l=0; l < cnn->k_size; l++) {
                        fread(&tmp, sizeof(tmp), 1, ptr);
                        cnn->w[i][j][k][l] = tmp;
                        cnn->d_w[i][j][k][l] = 0.;
                    }
                }
            }
        }
    } else if (type_couche == 1) { // Cas du NN
        // Lecture du "Pré-corps"
        kernel->nn = (Kernel_nn*)malloc(sizeof(Kernel_nn));
        kernel->cnn = NULL;
        uint32_t buffer[4];
        fread(&buffer, sizeof(buffer), 1, ptr);

        kernel->activation = buffer[0];
        kernel->linearisation = buffer[1];
        kernel->nn->input_units = buffer[2];
        kernel->nn->output_units = buffer[3];

        // Lecture du corps
        Kernel_nn* nn = kernel->nn;
        float tmp;

        nn->bias = (float*)malloc(sizeof(float)*nn->output_units);
        nn->d_bias = (float*)malloc(sizeof(float)*nn->output_units);
        for (int i=0; i < nn->output_units; i++) {
            fread(&tmp, sizeof(tmp), 1, ptr);
            nn->bias[i] = tmp;
            nn->d_bias[i] = 0.;
        }

        nn->weights = (float**)malloc(sizeof(float*)*nn->input_units);
        nn->d_weights = (float**)malloc(sizeof(float*)*nn->input_units);
        for (int i=0; i < nn->input_units; i++) {
            nn->weights[i] = (float*)malloc(sizeof(float)*nn->output_units);
            nn->d_weights[i] = (float*)malloc(sizeof(float)*nn->output_units);
            for (int j=0; j < nn->output_units; j++) {
                fread(&tmp, sizeof(tmp), 1, ptr);
                nn->weights[i][j] = tmp;
                nn->d_weights[i][j] = 0.;
            }
        }
    } else if (type_couche == 2) { // Cas du Pooling Layer
        uint32_t pooling, linearisation;
        fread(&pooling, sizeof(pooling), 1, ptr);
        fread(&linearisation, sizeof(linearisation), 1, ptr);

        kernel->cnn = NULL;
        kernel->nn = NULL;
        kernel->activation = pooling;
        kernel->linearisation = linearisation;
    }
    return kernel;
}