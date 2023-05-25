#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "neuron.h"

#ifndef DEF_NEURAL_NETWORK_H
#define DEF_NEURAL_NETWORK_H

#define LEARNING_RATE 0.1
// Retourne un nombre aléatoire entre 0 et 1
#define RAND_DOUBLE() ((double)rand())/((double)RAND_MAX)
//Coefficient leaking ReLU
#define COEFF_LEAKY_RELU 0.2
#define MAX_RESEAU 100000

#define PRINT_POIDS false
#define PRINT_BIAIS false

// Mettre à 1 pour désactiver
#define DROPOUT 1
#define ENTRY_DROPOUT 1


bool drop(float prob);

/*
* Fonction max pour les floats
*/
float max(float a, float b);

float sigmoid(float x);

float sigmoid_derivative(float x);

float leaky_ReLU(float x);

float leaky_ReLU_derivative(float x);

/*
* Remplace le pointeur par un réseau de neurones qu'elle crée
* et auquel elle alloue de la mémoire aux différentes variables
*/
void network_creation(Network* network, int* neurons_per_layer, int nb_layers);

/*
* Libère l'espace mémoire alloué dans le pointeur aux différentes
* variables dans la fonction 'creation_du_network'
*/
void deletion_of_network(Network* network);

/*
* Effectue une propagation en avant du réseau de neurones lorsque
* les données on été insérées dans la première couche. Le résultat
* de la propagation se trouve dans la dernière couche
*/
void forward_propagation(Network* network, bool is_training);

/*
* Renvoie la liste des sorties voulues à partir du nombre voulu
*/
int* desired_output_creation(Network* network, int wanted_number);

/*
* Effectue une propagation en arrière du réseau de neurones
* lorsqu'une forward_propagation a déjà été effectuée
*/
void backward_propagation(Network* network, int* desired_output);

/*
* Modifie les poids et le biais des neurones du réseau de neurones
* après une ou plusieurs backward_propagation
*/
void network_modification(Network* network, uint32_t nb_modifs);

/*
* Initialise les variables du réseau de neurones
*/
void network_initialisation(Network* network);

/*
* Les deux réseaux donnés sont supposés de même dimensions,
* Applique les modifications contenues dans delta à network
*/
void patch_network(Network* network, Network* delta, uint32_t nb_modifs);

/*
* Les deux réseaux donnés sont supposés de même dimensions
*/
void patch_delta(Network* network, Network* delta, uint32_t nb_modifs);

/*
* Renvoie une copie modifiable du réseau de neurones
*/
Network* copy_network(Network* network);

/*
* Renvoie l'erreur du réseau de neurones pour un numéro voulu
*/
float loss_computing(Network* network, int wanted_number);

#endif
