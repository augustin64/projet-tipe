#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "struct.h"

#ifndef DEF_NEURON_IO_H
#define DEF_NEURON_IO_H



// Écriture d'un réseau neuronal

/*
* Écrit un réseau neuronal dans un fichier donné
*/
void write_network(char* filename, Network* network);

/*
* Écrit une couche dans le fichier spécifié par le pointeur ptr
*/
void write_couche(Network* network, int indice_couche, int type_couche, FILE* ptr);


// Lecture d'un réseau neuronal

/*
* Lit un réseau neuronal dans un fichier donné
*/
Network* read_network(char* filename);

/*
* Lit une kernel dans le fichier spécifié par le pointeur ptr
*/
Kernel* read_kernel(int type_couche, int output_dim, FILE* ptr);
#endif