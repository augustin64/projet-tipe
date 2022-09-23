#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/neuron_io.h"
#include "include/struct.h"

#define MAGIC_NUMBER 1012

void write_network(char* filename, Network* network) {
    FILE *ptr;
    int size = network->size;
    int type_couche[size];

    ptr = fopen(filename, "wb");

    uint32_t buffer[(network->size)*3+4];

    buffer[0] = MAGIC_NUMBER;
    buffer[1] = size;
    buffer[2] = network->initialisation;
    buffer[3] = network->dropout;

    for (int i=0; i < size; i++) {
        buffer[2*i+4] = network->width[i];
        buffer[2*i+5] = network->depth[i];
    }

    for (int i=0; i < size; i++) {
        if ((!network->kernel[i]->cnn)&&(!network->kernel[i]->nn)) {
            type_couche[i] = 2;
        } else if (!network->kernel[i]->cnn) {
            type_couche[i] = 1;
        } else {
            type_couche[i] = 0;
        }
        buffer[i+2*size+4] = type_couche[i];
    }

    fwrite(buffer, sizeof(buffer), 1, ptr);


    for (int i=0; i < size; i++) {
        write_couche(network->kernel[i], type_couche[i], ptr);
    }

    fclose(ptr);
}


void write_couche(Kernel* kernel, int type_couche, FILE* ptr) {
    uint32_t activation[1];
    int indice;

    activation[0] = kernel->activation;

    fwrite(activation, sizeof(activation), 1, ptr);
    if (type_couche == 0) {
        Kernel_cnn* cnn = kernel->cnn;
        float buffer[2*cnn->k_size*cnn->k_size*cnn->columns*(cnn->rows+1)];
        for (int i=0; i < cnn->columns; i++) {
            for (int j=0; j < cnn->k_size; j++) {
                for (int k=0; k < cnn->k_size; k++) {
                    indice = cnn->k_size*(i*cnn->k_size+j)+k;
                    buffer[indice] = cnn->bias[i][j][k];
                }
            }
        }
        int av_bias = cnn->columns*cnn->k_size*cnn->k_size;
        for (int i=0; i < cnn->rows; i++) {
            for (int j=0; j < cnn->columns; j++) {
                for (int k=0; k < cnn->k_size; k++) {
                    for (int l=0; l < cnn->k_size; l++) {
                        indice = ((i*cnn->columns+j)*cnn->k_size+k)*cnn->k_size+l+av_bias;
                        buffer[indice] = cnn->w[i][j][k][l];
                    }
                }
            }
        }
        fwrite(buffer, sizeof(buffer), 1, ptr);
    } else if (type_couche == 1) {
        Kernel_nn* nn = kernel->nn;
        float buffer[(1+nn->input_units)*nn->output_units];
        for (int i=0; i < nn->output_units; i++) {
            buffer[i] = nn->bias[i];
        }
        int av_bias = nn->output_units;
        for (int i=0; i < nn->input_units; i++) {
            for (int j=0; j < nn->output_units; j++) {
                buffer[i*nn->output_units+j+av_bias] = nn->weights[i][j];
            }
        }
        fwrite(buffer, sizeof(buffer), 1, ptr);
    }
}


Network* read_network(char* filename) {
    FILE *ptr;
    Network* network = (Network*)malloc(sizeof(Network));
    // TODO: malloc pour network -> input

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
    fread(&initialisation, sizeof(uint32_t), 1, ptr);
    network->initialisation = initialisation;
    fread(&dropout, sizeof(uint32_t), 1, ptr);
    network->dropout = dropout;

    // Lecture de la taille de l'entrée des différentes matrices
    network->width = (int*)malloc(sizeof(int)*size);
    network->depth = (int*)malloc(sizeof(int)*size);

    for (int i=0; i < (int)size; i++) {
        fread(&tmp, sizeof(tmp), 1, ptr);
        network->width[i] = tmp;
        fread(&tmp, sizeof(tmp), 1, ptr);
        network->depth[i+1] = tmp;
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
        network->kernel[i] = read_kernel(type_couche[i], ptr);
    }
    
    fclose(ptr);
    return network;
}

Kernel* read_kernel(int type_couche, FILE* ptr) {
    Kernel* kernel = (Kernel*)malloc(sizeof(Kernel));
    if (type_couche == 0) {
        // TODO: lecture d'un CNN
    } else if (type_couche == 1) {
        // TODO: lecture d'un NN
    } else if (type_couche == 2) {
        uint32_t pooling;
        fread(&pooling, sizeof(pooling), 1, ptr);

        kernel->cnn = NULL;
        kernel->nn = NULL;
        kernel->activation = pooling;
        kernel->linearisation = pooling; // TODO: mettre à 0 la variable inutile
    }
    return kernel;
}