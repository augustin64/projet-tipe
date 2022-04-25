#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "struct/neuron.h"

#ifndef DEF_NEURAL_NETWORK_H
#define DEF_NEURAL_NETWORK_H

float max(float a, float b);
float sigmoid(float x);
float sigmoid_derivative(float x);
float leaky_ReLU(float x);
float leaky_ReLU_derivative(float x);
void network_creation(Network* network_neuronal, int* neurons_per_layer, int nb_layers);
void deletion_of_network(Network* network_neuronal);
void forward_propagation(Network* network_neuronal);
int* desired_output_creation(Network* network_neuronal, int wanted_number);
void backward_propagation(Network* network_neuronal, int* desired_output);
void network_modification(Network* network_neuronal, uint32_t nb_modifs);
void network_initialisation(Network* network_neuronal);
float loss_computing(Network* network, int numero_voulu);

#endif
