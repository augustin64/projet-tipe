#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/test.h"

#include "../src/dense/include/neural_network.h"
#include "../src/dense/include/neuron_io.h"
#include "../src/common/include/colors.h"


Neuron* creer_neuron(int nb_sortants) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    if (nb_sortants != 0) {
        neuron->weights = (float*)malloc(sizeof(float)*nb_sortants);
        neuron->back_weights = (float*)malloc(sizeof(float)*nb_sortants);
        neuron->last_back_weights = (float*)malloc(sizeof(float)*nb_sortants);

        for (int i=0; i < nb_sortants; i++) {
            neuron->weights[i] = 0.5;
            neuron->back_weights[i] = 0.0;
            neuron->last_back_weights[i] = 0.0;
        }
        neuron->z = 0.0;
        neuron->bias = 0.0;
        neuron->back_bias = 0.0;
        neuron->last_back_bias = 0.0;
    }

    return neuron;
}


Layer* creer_layer(int nb_neurons, int nb_sortants) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    Neuron** tab = (Neuron**)malloc(sizeof(Neuron*)*nb_neurons);

    layer->nb_neurons = nb_neurons;
    layer->neurons = tab;

    for (int i=0; i < nb_neurons; i++) {
        tab[i] = creer_neuron(nb_sortants);
    }
    return layer;
};


Network* create_network(int nb_layers, int nb_max_neurons, int nb_min_neurons) {
    Network* network = (Network*)malloc(sizeof(Network));
    network->layers = (Layer**)malloc(sizeof(Layer*)*nb_layers);
    int nb_neurons[nb_layers+1];

    network->nb_layers = nb_layers;

    for (int i=0; i < nb_layers; i++) {
        nb_neurons[i] = i*(nb_min_neurons-nb_max_neurons)/(nb_layers-1) + nb_max_neurons;
    }
    nb_neurons[nb_layers] = 0;

    for (int i=0; i < nb_layers; i++) {
        network->layers[i] = creer_layer(nb_neurons[i], nb_neurons[i+1]);
    }
    return network;
}

int main() {
    _TEST_PRESENTATION("Dense: Lecture/Écriture")
    
    Network* network = create_network(5, 300, 10);
    _TEST_ASSERT(true, "Création du réseau");

    write_network((char*)".test-cache/neuron_io.bin", network);
    _TEST_ASSERT(true, "Écriture du réseau");
    
    Network* network2 = read_network((char*)".test-cache/neuron_io.bin");
    _TEST_ASSERT(true, "Accès en lecture");
    
    deletion_of_network(network);
    deletion_of_network(network2);
    _TEST_ASSERT(true, "Suppression des réseaux");

    return 0;
}