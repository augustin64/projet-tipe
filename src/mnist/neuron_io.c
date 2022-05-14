#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/neuron.h"
#define MAGIC_NUMBER 2023



Neuron* read_neuron(uint32_t nb_weights, FILE *ptr) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    float activation;
    float bias;
    float tmp;

    fread(&activation, sizeof(float), 1, ptr);
    fread(&bias, sizeof(float), 1, ptr);

    neuron->bias = bias;

    neuron->z = 0.0;
    neuron->last_back_bias = 0.0;
    neuron->back_bias = 0.0;

    float* weights = (float*)malloc(sizeof(float)*nb_weights);

    neuron->last_back_weights = (float*)malloc(sizeof(float)*nb_weights);
    neuron->back_weights = (float*)malloc(sizeof(float)*nb_weights);
    neuron->weights = weights;

    for (int i=0; i < (int)nb_weights; i++) {
        fread(&tmp, sizeof(float), 1, ptr);
        neuron->weights[i] = tmp;
        neuron->back_weights[i] = 0.0;
        neuron->last_back_weights[i] = 0.0;
    }

    return neuron;
}


// Lit une couche de neurones
Neuron** read_neurons(uint32_t nb_neurons, uint32_t nb_weights, FILE *ptr) {
    Neuron** neurons = (Neuron**)malloc(sizeof(Neuron*)*nb_neurons);
    for (int i=0; i < (int)nb_neurons; i++) {
        neurons[i] = read_neuron(nb_weights, ptr);
    }
    return neurons;
}


// Charge l'entièreté du réseau neuronal depuis un fichier binaire
Network* read_network(char* filename) {
    FILE *ptr;
    Network* network = (Network*)malloc(sizeof(Network));
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t nb_layers;
    uint32_t tmp;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    if (magic_number != MAGIC_NUMBER) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&nb_layers, sizeof(uint32_t), 1, ptr);
    network->nb_layers = nb_layers;


    Layer** layers = (Layer**)malloc(sizeof(Layer*)*nb_layers);
    uint32_t nb_neurons_layer[nb_layers+1];

    network->layers = layers;

    for (int i=0; i < (int)nb_layers; i++) {
        layers[i] = (Layer*)malloc(sizeof(Layer));
        fread(&tmp, sizeof(tmp), 1, ptr);
        layers[i]->nb_neurons = tmp;
        nb_neurons_layer[i] = tmp;
    }
    nb_neurons_layer[nb_layers] = 0;

    for (int i=0; i < (int)nb_layers; i++) {
        layers[i]->neurons = read_neurons(layers[i]->nb_neurons, nb_neurons_layer[i+1], ptr);
    }

    fclose(ptr);
    return network;
}




// Écrit un neurone dans le fichier pointé par *ptr
void write_neuron(Neuron* neuron, int weights, FILE *ptr) {
    float buffer[weights+2];

    buffer[1] = neuron->bias;
    for (int i=0; i < weights; i++) {
        buffer[i+2] = neuron->weights[i];
    }

    fwrite(buffer, sizeof(buffer), 1, ptr);
}


// Stocke l'entièreté du réseau neuronal dans un fichier binaire
int write_network(char* filename, Network* network) {
    FILE *ptr;
    int nb_layers = network->nb_layers;
    int nb_neurons[nb_layers+1];

    ptr = fopen(filename, "wb");

    uint32_t buffer[nb_layers+2];

    buffer[0] = MAGIC_NUMBER;
    buffer[1] = nb_layers;
    for (int i=0; i < nb_layers; i++) {
        buffer[i+2] = network->layers[i]->nb_neurons;
        nb_neurons[i] = network->layers[i]->nb_neurons;
    }
    nb_neurons[nb_layers] = 0;
    fwrite(buffer, sizeof(buffer), 1, ptr);
    for (int i=0; i < nb_layers; i++) {
        for (int j=0; j < nb_neurons[i]; j++) {
            write_neuron(network->layers[i]->neurons[j], nb_neurons[i+1], ptr);
        }
    }

    fclose(ptr);
    return 1;
}
