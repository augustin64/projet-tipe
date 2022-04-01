#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "struct/neuron.h"

#ifndef DEF_NEURON_IO_H
#define DEF_NEURON_IO_H

Neurone* lire_neurone(uint32_t nb_poids_sortants, FILE *ptr);
Neurone** lire_neurones(uint32_t nb_neurones, uint32_t nb_poids_sortants, FILE *ptr);
Reseau* lire_reseau(char* filename);
void ecrire_neurone(Neurone* neurone, int poids_sortants, FILE *ptr);
int ecrire_reseau(char* filename, Reseau* reseau);


#endif
