#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "struct.h"

#ifndef DEF_UTILS_H
#define DEF_UTILS_H

/*
* Échange deux éléments d'un tableau
*/
void swap(int* tab, int i, int j);

/*
* Mélange un tableau avec le mélange de Knuth
*/
void knuth_shuffle(int* tab, int n);

/*
* Vérifie si deux réseaux sont égaux
*/
bool equals_networks(Network* network1, Network* network2);

/*
 * Duplique un réseau
*/
Network* copy_network(Network* network);

#endif