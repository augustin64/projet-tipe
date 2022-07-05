#include "struct.h"

#ifndef DEF_FREE_H
#define DEF_FREE_H

/*
* Libère la mémoire allouée à une couche de type input cube
*/
void free_a_cube_input_layer(Network* network, int pos, int depth, int dim);

/*
* Libère la mémoire allouée à une couche de type input line
*/
void free_a_line_input_layer(Network* network, int pos);

/*
* Libère l'espace mémoie et supprime une couche d'average pooling classique
*/
void free_average_pooling(Network* network, int pos);

/*
* Libère l'espace mémoie et supprime une couche d'average pooling flatten
*/
void free_average_pooling_flatten(Network* network, int pos);

/*
* Libère l'espace mémoire et supprime une couche de convolution
*/
void free_convolution(Network* network, int pos);

/*
* Libère l'espace mémoire et supprime une couche dense
*/
void free_dense(Network* network, int pos);

/*
* Libère l'espace alloué dans la fonction 'create_network'
*/
void free_network_creation(Network* network);

/*
* Libère l'espace alloué dans la fonction 'create_network_lenet5'
*/
void free_network_lenet5(Network* network);

#endif