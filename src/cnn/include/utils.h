#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "../../colors.h"
#include "struct.h"

#ifndef DEF_UTILS_H
#define DEF_UTILS_H

/*
* Vérifie si deux réseaux sont égaux
*/
bool equals_networks(Network* network1, Network* network2);

/*
 * Duplique un réseau
*/
Network* copy_network(Network* network);

#endif