#include "struct.h"

#ifndef DEF_FREE_H
#define DEF_FREE_H

/*
* Libère la mémoire allouée à une couche de type input cube
* Donc free networkt->input[pos][i][j]
*/
void free_a_cube_input_layer(Network* network, int pos, int depth, int dim);

/*
* Libère la mémoire allouée à une couche de type input line
* Donc free networkt->input[pos][0][0]
*/
void free_a_line_input_layer(Network* network, int pos);

/*
* Libère l'espace mémoire alloué dans 'add_average_pooling' ou 'add_max_pooling' (creation.c)
*/
void free_pooling(Network* network, int pos);

/*
* Libère l'espace mémoire dans 'add_convolution' (creation.c)
*/
void free_convolution(Network* network, int pos);

/*
* Libère l'espace mémoire alloué dans 'add_dense' (creation.c)
*/
void free_dense(Network* network, int pos);

/*
* Libère l'espace mémoire alloué dans 'add_dense_linearisation' (creation.c)
*/
void free_dense_linearisation(Network* network, int pos);

/*
* Libère l'espace mémoire alloué dans 'create_network' (creation.c)
*/
void free_network_creation(Network* network);

/*
* Libère l'espace mémoire alloué à un réseau quelconque
*/
void free_network(Network* network);

#endif