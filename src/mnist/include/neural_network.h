#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "neuron.h"

#ifndef DEF_NEURAL_NETWORK_H
#define DEF_NEURAL_NETWORK_H

float max(float a, float b);
float sigmoid(float x);
float sigmoid_derivative(float x);
float leaky_ReLU(float x);
float leaky_ReLU_derivative(float x);
void network_creation(Network* network, int* neurons_per_layer, int nb_layers);
void deletion_of_network(Network* network);
void forward_propagation(Network* network);
int* desired_output_creation(Network* network, int wanted_number);
void backward_propagation(Network* network, int* desired_output);
void network_modification(Network* network, uint32_t nb_modifs);
void network_initialisation(Network* network);
void patch_network(Network* network, Network* delta, uint32_t nb_modifs);
void patch_delta(Network* network, Network* delta, uint32_t nb_modifs);
Network* copy_network(Network* network);
float loss_computing(Network* network, int numero_voulu);

#ifdef __CUDACC__
Network* copy_network_cuda(Network* network);
#endif
#endif
