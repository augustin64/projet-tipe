#include "struct.h"

#ifndef DEF_MAIN_H
#define DEF_MAIN_H


/*
* Renvoie si oui ou non (1 ou 0) le neurone va être abandonné
*/
int will_be_drop(int dropout_prob);

/*
* Écrit une image 28*28 au centre d'un tableau 32*32 et met à 0 le reste
*/
void write_image_in_network_32(int** image, int height, int width, float** input);

/*
* Propage en avant le cnn
*/
void forward_propagation(Network* network);

/*
* Propage en arrière le cnn
*/
void backward_propagation(Network* network, float wanted_number); //NOT FINISHED

/*
* Renvoie l'erreur du réseau neuronal pour une sortie
*/
float compute_cross_entropy_loss(float* output, float* wanted_output, int len);

/*
* On considère que la sortie voulue comporte 10 éléments
*/
float* generate_wanted_output(float wanted_number);

#endif