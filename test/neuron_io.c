#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/mnist/neuron_io.c"


Neuron* creer_neuron(int nb_sortants) {
    Neuron* neuron = malloc(2*sizeof(float*)+6*sizeof(float));
    neuron->weights = malloc(sizeof(float)*nb_sortants);
    neuron->back_weights = malloc(sizeof(float)*nb_sortants);
    neuron->last_back_weights = malloc(sizeof(float)*nb_sortants);

    for (int i=0; i < nb_sortants; i++) {
        neuron->weights[i] = 0.5;
        neuron->back_weights[i] = 0.0;
        neuron->last_back_weights[i] = 0.0;
    }
    neuron->z = 0.0;
    neuron->bias = 0.0;
    neuron->back_bias = 0.0;
    neuron->last_back_bias = 0.0;

    return neuron;
}


Layer* creer_layer(int nb_neurons, int nb_sortants) {
    Layer* layer = malloc(sizeof(int)+sizeof(Neuron**));
    Neuron** tab = malloc(sizeof(Neuron*)*nb_neurons);

    layer->nb_neurons = nb_neurons;
    layer->neurons = tab;

    for (int i=0; i<nb_neurons; i++) {
        tab[i] = creer_neuron(nb_sortants);
    }
    return layer;
};


Network* create_network(int nb_layers, int nb_max_neurons, int nb_min_neurons) {
    Network* network = malloc(sizeof(int)+sizeof(Layer**));
    network->layers = malloc(sizeof(Layer*)*nb_layers);
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
    Network* network = create_network(5, 300, 10);
    write_network(".test-cache/neuron_io.bin", network);
    Network* network2 = read_network(".test-cache/neuron_io.bin");
    return 1;
}