#include "struct.h"
#include "initialisation.h"

#ifndef DEF_CREATION_H
#define DEF_CREATION_H

/*
* Créé un réseau qui peut contenir max_size couche (dont celle d'input et d'output)
*/
Network* create_network(int max_size, float learning_rate, int dropout, int initialisation, int input_width, int input_depth);

/*
* Renvoie un réseau suivant l'architecture LeNet5
*/
Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth);

/*
* Renvoie un réseau suivant l'architecture AlexNet
* C'est à dire en entrée 3x227x227 et une sortie de taille 'size_output'
*/
Network* create_network_alexnet(float learning_rate, int dropout, int activation, int initialisation, int size_output);

/*
* Renvoie un réseau suivant l'architecture VGG16 modifiée pour prendre en entrée 3x256x256
* et une sortie de taille 'size_output'
*/
Network* create_network_VGG16(float learning_rate, int dropout, int activation, int initialisation, int size_output);

/*
* Renvoie un réseau sans convolution, similaire à celui utilisé dans src/dense
*/
Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth);

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
* Ajoute au réseau une couche d'average pooling avec la taille de noyau (kernel_size), 
* le remplissage (padding) et le décalge (stride) choisis
*/
void add_average_pooling(Network* network, int kernel_size, int stride, int padding);

/*
* Ajoute au réseau une couche de max pooling avec la taille de noyau (kernel_size), 
* le remplissage (padding) et le décalge (stride) choisis
*/
void add_max_pooling(Network* network, int kernel_size, int stride, int padding);

/*
* Ajoute au réseau une couche de convolution avec la taille de noyau (kernel_size), 
* le remplissage (padding) et le décalge (stride) choisis. Le choix de la profondeur de 
* la couche suivante se fait avec number_of_kernels (= output_depth)
* Puis initialise les poids et les biais construits
*/
void add_convolution(Network* network, int kernel_size, int number_of_kernels, int stride, int padding, int activation);

/*
* Ajoute au réseau une couche dense et initialise les poids et les biais
*/
void add_dense(Network* network, int size_output, int activation);

/*
* Ajoute au réseau une couche dense qui aplatit
*/
void add_dense_linearisation(Network* network, int size_output, int activation);

#endif