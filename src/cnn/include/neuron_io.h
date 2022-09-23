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
void write_couche(Kernel* kernel, int type_couche, FILE* ptr);
#endif