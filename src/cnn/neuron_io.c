#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../include/memory_management.h"
#include "../include/colors.h"
#include "include/function.h"
#include "include/struct.h"

#include "include/neuron_io.h"

#define MAGIC_NUMBER 1012

#define bufferAdd(val) {buffer[indice_buffer] = val; indice_buffer++;}

void write_network(char* filename, Network* network) {
    FILE *ptr;
    int size = network->size;
    int type_couche[size-1];
    int indice_buffer = 0;

    ptr = fopen(filename, "wb");

    // Le buffer est composé de:
    // - MAGIC_NUMBER (1)
    // - size (2)
    // - network->initialisation (3)
    // - network->dropout (4)
    // - network->width[i] & network->depth[i] (4+network->size*2)
    // - type_couche[i] (3+network->size*3) - On exclue la dernière couche
    uint32_t buffer[(network->size)*3+3];

    bufferAdd(MAGIC_NUMBER);
    bufferAdd(size);
    bufferAdd(network->initialisation);
    bufferAdd(network->dropout);

    // Écriture du header
    for (int i=0; i < size; i++) {
        bufferAdd(network->width[i]);
        bufferAdd(network->depth[i]);
    }

    for (int i=0; i < size-1; i++) {
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
    for (int i=0; i < size-1; i++) {
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
        // We need to split in small buffers to keep some free memory in the computer
        for (int i=0; i < cnn->columns; i++) {
            indice_buffer = 0;
            float buffer[output_dim*output_dim];
            for (int j=0; j < output_dim; j++) {
                for (int k=0; k < output_dim; k++) {
                    bufferAdd(cnn->bias[i][j][k]);
                }
            }
            fwrite(buffer, sizeof(buffer), 1, ptr);
        }
        for (int i=0; i < cnn->rows; i++) {
            indice_buffer = 0;
            float buffer[cnn->columns*cnn->k_size*cnn->k_size];
            for (int j=0; j < cnn->columns; j++) {
                for (int k=0; k < cnn->k_size; k++) {
                    for (int l=0; l < cnn->k_size; l++) {
                        bufferAdd(cnn->w[i][j][k][l]);
                    }
                }
            }
            fwrite(buffer, sizeof(buffer), 1, ptr);
        }
    } else if (type_couche == 1) { // Cas du NN
        Kernel_nn* nn = kernel->nn;

        // Écriture du pré-corps
        uint32_t pre_buffer[4];
        pre_buffer[0] = kernel->activation;
        pre_buffer[1] = kernel->linearisation;
        pre_buffer[2] = nn->size_input;
        pre_buffer[3] = nn->size_output;
        fwrite(pre_buffer, sizeof(pre_buffer), 1, ptr);

        // Écriture du corps
        float buffer[nn->size_output];
        for (int i=0; i < nn->size_output; i++) {
            bufferAdd(nn->bias[i]);
        }
        fwrite(buffer, sizeof(buffer), 1, ptr);

        for (int i=0; i < nn->size_input; i++) {
            indice_buffer = 0;
            float buffer[nn->size_output];
            for (int j=0; j < nn->size_output; j++) {
                bufferAdd(nn->weights[i][j]);
            }
            fwrite(buffer, sizeof(buffer), 1, ptr);
        }
    } else if (type_couche == 2) { // Cas du Pooling Layer
        uint32_t pre_buffer[2];
        pre_buffer[0] = kernel->linearisation;
        pre_buffer[1] = kernel->pooling;
        fwrite(pre_buffer, sizeof(pre_buffer), 1, ptr);
    }
}


Network* read_network(char* filename) {
    FILE *ptr;
    Network* network = (Network*)nalloc(sizeof(Network));

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
    network->width = (int*)nalloc(sizeof(int)*size);
    network->depth = (int*)nalloc(sizeof(int)*size);

    for (int i=0; i < (int)size; i++) {
        fread(&tmp, sizeof(uint32_t), 1, ptr);
        network->width[i] = tmp;
        fread(&tmp, sizeof(uint32_t), 1, ptr);
        network->depth[i] = tmp;
    }

    // Lecture du type de chaque couche
    uint32_t type_couche[size-1];

    for (int i=0; i < (int)size-1; i++) {
        fread(&tmp, sizeof(tmp), 1, ptr);
        type_couche[i] = tmp;
    }

    // Lecture de chaque couche
    network->kernel = (Kernel**)nalloc(sizeof(Kernel*)*(size-1));

    for (int i=0; i < (int)size-1; i++) {
        network->kernel[i] = read_kernel(type_couche[i], network->width[i+1], ptr);
    }

    network->input = (float****)nalloc(sizeof(float***)*size);
    for (int i=0; i < (int)size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network->input[i] = (float***)nalloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network->input[i][j] = (float**)nalloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network->input[i][j][k] = (float*)nalloc(sizeof(float)*network->width[i]);
                for (int l=0; l < network->width[i]; l++) {
                    network->input[i][j][k][l] = 0.;
                }
            }
        }
    }

    network->input_z = (float****)nalloc(sizeof(float***)*size);
    for (int i=0; i < (int)size; i++) { // input[size][couche->depth][couche->dim][couche->dim]
        network->input_z[i] = (float***)nalloc(sizeof(float**)*network->depth[i]);
        for (int j=0; j < network->depth[i]; j++) {
            network->input_z[i][j] = (float**)nalloc(sizeof(float*)*network->width[i]);
            for (int k=0; k < network->width[i]; k++) {
                network->input_z[i][j][k] = (float*)nalloc(sizeof(float)*network->width[i]);
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
    Kernel* kernel = (Kernel*)nalloc(sizeof(Kernel));
    if (type_couche == 0) { // Cas du CNN
        // Lecture du "Pré-corps"
        kernel->cnn = (Kernel_cnn*)nalloc(sizeof(Kernel_cnn));
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

        cnn->bias = (float***)nalloc(sizeof(float**)*cnn->columns);
        cnn->d_bias = (float***)nalloc(sizeof(float**)*cnn->columns);
        for (int i=0; i < cnn->columns; i++) {
            cnn->bias[i] = (float**)nalloc(sizeof(float*)*output_dim);
            cnn->d_bias[i] = (float**)nalloc(sizeof(float*)*output_dim);
            for (int j=0; j < output_dim; j++) {
                cnn->bias[i][j] = (float*)nalloc(sizeof(float)*output_dim);
                cnn->d_bias[i][j] = (float*)nalloc(sizeof(float)*output_dim);
                for (int k=0; k < output_dim; k++) {
                    fread(&tmp, sizeof(tmp), 1, ptr);
                    cnn->bias[i][j][k] = tmp;
                    cnn->d_bias[i][j][k] = 0.;
                }
            }
        }

        cnn->w = (float****)nalloc(sizeof(float***)*cnn->rows);
        cnn->d_w = (float****)nalloc(sizeof(float***)*cnn->rows);
        for (int i=0; i < cnn->rows; i++) {
            cnn->w[i] = (float***)nalloc(sizeof(float**)*cnn->columns);
            cnn->d_w[i] = (float***)nalloc(sizeof(float**)*cnn->columns);
            for (int j=0; j < cnn->columns; j++) {
                cnn->w[i][j] = (float**)nalloc(sizeof(float*)*cnn->k_size);
                cnn->d_w[i][j] = (float**)nalloc(sizeof(float*)*cnn->k_size);
                for (int k=0; k < cnn->k_size; k++) {
                    cnn->w[i][j][k] = (float*)nalloc(sizeof(float)*cnn->k_size);
                    cnn->d_w[i][j][k] = (float*)nalloc(sizeof(float)*cnn->k_size);
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
        kernel->nn = (Kernel_nn*)nalloc(sizeof(Kernel_nn));
        kernel->cnn = NULL;
        uint32_t buffer[4];
        fread(&buffer, sizeof(buffer), 1, ptr);

        kernel->activation = buffer[0];
        kernel->linearisation = buffer[1];
        kernel->nn->size_input = buffer[2];
        kernel->nn->size_output = buffer[3];

        // Lecture du corps
        Kernel_nn* nn = kernel->nn;
        float tmp;

        nn->bias = (float*)nalloc(sizeof(float)*nn->size_output);
        nn->d_bias = (float*)nalloc(sizeof(float)*nn->size_output);
        for (int i=0; i < nn->size_output; i++) {
            fread(&tmp, sizeof(tmp), 1, ptr);
            nn->bias[i] = tmp;
            nn->d_bias[i] = 0.;
        }

        nn->weights = (float**)nalloc(sizeof(float*)*nn->size_input);
        nn->d_weights = (float**)nalloc(sizeof(float*)*nn->size_input);
        for (int i=0; i < nn->size_input; i++) {
            nn->weights[i] = (float*)nalloc(sizeof(float)*nn->size_output);
            nn->d_weights[i] = (float*)nalloc(sizeof(float)*nn->size_output);
            for (int j=0; j < nn->size_output; j++) {
                fread(&tmp, sizeof(tmp), 1, ptr);
                nn->weights[i][j] = tmp;
                nn->d_weights[i][j] = 0.;
            }
        }
    } else if (type_couche == 2) { // Cas du Pooling Layer
        uint32_t pooling, linearisation;
        fread(&linearisation, sizeof(linearisation), 1, ptr);
        fread(&pooling, sizeof(pooling), 1, ptr);

        kernel->cnn = NULL;
        kernel->nn = NULL;
        kernel->activation = IDENTITY;
        kernel->pooling = pooling;
        kernel->linearisation = linearisation;
    }
    return kernel;
}