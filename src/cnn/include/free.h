#include "struct.h"

#ifndef DEF_FREE_H
#define DEF_FREE_H

/*
* Libère l'espace mémoire de network->input[pos] et network->input_z[pos]
* lorsque ces couches sont non denses (donc sont des matrice de dimension 3)
* Libère donc l'espace mémoire alloué dans 'create_a_cube_input_layer' et create_a_cube_input_z_layer' (creation.c)
*/
void free_a_cube_input_layer(Network* network, int pos, int depth, int dim);

/*
* Libère l'espace mémoire de network->input[pos] et network->input_z[pos]
* lorsque ces couches sont denses (donc sont des matrice de dimension 1)
* Libère donc l'espace mémoire alloué dans 'create_a_line_input_layer' et create_a_line_input_z_layer' (creation.c)
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
* Libère entièrement l'espace mémoire alloué à un réseau quelconque
*/
void free_network(Network* network);

/*
* Libère l'espace mémoire alloué pour une d_convolution
*/
void free_d_convolution(Network* network, int pos);

/*
* Libère l'espace mémoire alloué pour une d_dense
*/
void free_d_dense(Network* network, int pos);

/*
* Libère l'espace mémoire alloué pour une d_dense_linearisation
*/
void free_d_dense_linearisation(Network* network, int pos);

/*
* Libère entièrement l'espace mémoire alloué dans 'create_d_network' (creation.c)
*/
void free_d_network_creation(Network* network, D_Network* d_network);

#endif