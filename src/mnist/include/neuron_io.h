#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "neuron.h"

#ifndef DEF_NEURON_IO_H
#define DEF_NEURON_IO_H



// Lecture d'un réseau neuronal 

/* 
* Lit un neurone
*/
Neuron* read_neuron(uint32_t nb_weights, FILE *ptr);

/*
* Lit une couche de neurones
*/
Neuron** read_neurons(uint32_t nb_neurons, uint32_t nb_weights, FILE *ptr);

/*
* Charge l'entièreté du réseau neuronal depuis un fichier binaire
*/
Network* read_network(char* filename);




// Écriture d'un réseau neuronal

/*
* Écrit un neurone dans le fichier pointé par *ptr
*/
void write_neuron(Neuron* neuron, int weights, FILE *ptr);

/*
* Stocke l'entièreté du réseau neuronal dans un fichier binaire
*/
void write_network(char* filename, Network* network);




// Lecture des calculs de la backpropagation d'un réseau neuronal

/*
* Lit un neurone
*/
Neuron* read_delta_neuron(uint32_t nb_weights, FILE *ptr);

/*
* Lit une couche de neurones
*/
Neuron** read_delta_neurons(uint32_t nb_neurons, uint32_t nb_weights, FILE *ptr);

/*
* Charge l'entièreté du réseau neuronal depuis un fichier binaire
*/
Network* read_delta_network(char* filename);




// Écriture des calculs de la backpropagation d'un réseau neuronal

/*
* Écrit les calculs de backpropagation effectués sur
* un neurone dans le fichier pointé par *ptr
*/
void write_delta_neuron(Neuron* neuron, int weights, FILE *ptr);

/*
* Enregistre les calculs de backpropagation effectués
* sur un réseau dans un fichier
*/
void write_delta_network(char* filename, Network* network);

#endif
