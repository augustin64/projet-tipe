#include "struct.h"
#include "initialisation.h"

#ifndef DEF_CREATION_H
#define DEF_CREATION_H

/*
* Créé un réseau qui peut contenir max_size couche (dont celle d'input et d'output)
*/
Network* create_network(int max_size, float learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth);

/*
* Renvoie un réseau suivant l'architecture LeNet5
*/
Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth);

/*
* Renvoie un réseau sans convolution, similaire à celui utilisé dans src/mnist
*/
Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_dim, int input_depth);

/*
* Créé et alloue de la mémoire à une couche de type input cube
*/
void create_a_cube_input_layer(Network* network, int pos, int depth, int dim);

/*
* Créé et alloue de la mémoire à une couche de type input_z cube
*/
void create_a_cube_input_z_layer(Network* network, int pos, int depth, int dim);

/*
* Créé et alloue de la mémoire à une couche de type ligne
*/
void create_a_line_input_layer(Network* network, int pos, int dim);

/*
* Ajoute au réseau une couche d'average pooling valide de dimension dim*dim
*/
void add_average_pooling(Network* network, int dim_output);

/*
* Ajoute au réseau une couche de max pooling valide de dimension dim*dim
*/
void add_max_pooling(Network* network, int dim_output);

/*
* Ajoute au réseau une couche de convolution dim*dim et initialise les kernels
*/
void add_convolution(Network* network, int depth_output, int dim_output, int activation);

/*
* Ajoute au réseau une couche dense et initialise les poids et les biais
*/
void add_dense(Network* network, int size_output, int activation);

/*
* Ajoute au réseau une couche dense qui aplatit
*/
void add_dense_linearisation(Network* network, int size_output, int activation);

#endif