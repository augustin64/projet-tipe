#include "struct.h"
#include "initialisation.h"

#ifndef DEF_CREATION_H
#define DEF_CREATION_H

/*
* Créé un réseau qui peut contenir max_size couche (dont celle d'input et d'output)
*/
Network* create_network(int max_size, int dropout, int initialisation, int input_dim, int input_depth);

/*
* Renvoie un réseau suivant l'architecture LeNet5
*/
Network* create_network_lenet5(int dropout, int activation, int initialisation);

/*
* Créé et alloue de la mémoire à une couche de type input cube
*/
void create_a_cube_input_layer(Network* network, int pos, int depth, int dim); // CHECKED

/*
* Créé et alloue de la mémoire à une couche de type ligne
*/
void create_a_line_input_layer(Network* network, int pos, int dim);

/*
* Ajoute au réseau une couche d'average pooling valide de dimension dim*dim
*/
void add_2d_average_pooling(Network* network, int dim_input, int dim_ouput);

/*
* Ajoute au réseau une couche de convolution dim*dim et initialise les kernels
*/
void add_convolution(Network* network, int depth_input, int dim_input, int depth_output, int dim_output, int activation);

/*
* Ajoute au réseau une couche dense et initialise les poids et les biais
*/
void add_dense(Network* network, int input_units, int output_units, int activation);

/*
* Ajoute au réseau une couche dense qui aplatit
*/
void add_dense_linearisation(Network* network, int input_units, int output_units, int activation);

#endif