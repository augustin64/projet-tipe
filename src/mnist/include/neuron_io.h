#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "neuron.h"

#ifndef DEF_NEURON_IO_H
#define DEF_NEURON_IO_H

Neuron* read_neuron(uint32_t nb_weights, FILE *ptr);
Neuron** read_neurons(uint32_t nb_neurons, uint32_t nb_weights, FILE *ptr);
Network* read_network(char* filename);

void write_neuron(Neuron* neuron, int weights, FILE *ptr);
void write_network(char* filename, Network* network);

Neuron* read_delta_neuron(uint32_t nb_weights, FILE *ptr);
Neuron** read_delta_neurons(uint32_t nb_neurons, uint32_t nb_weights, FILE *ptr);
Network* read_delta_network(char* filename);

void write_delta_neuron(Neuron* neuron, int weights, FILE *ptr);
void write_delta_network(char* filename, Network* network);

#endif
