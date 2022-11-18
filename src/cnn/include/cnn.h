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
void backward_propagation(Network* network, float wanted_number);

/*
* Met à 0 chaque valeur de l'input avec une probabilité de dropout %
*/
void drop_neurones(float*** input, int depth, int dim1, int dim2, int dropout);

/*
* Copie les données de output dans output_a (Sachant que les deux matrices ont les mêmes dimensions)
*/
void copy_input_to_input_z(float*** output, float*** output_a, int output_depth, int output_rows, int output_columns);

/*
* Renvoie l'erreur du réseau neuronal pour une sortie (RMS)
*/
float compute_mean_squared_error(float* output, float* wanted_output, int len);

/*
* Renvoie l'erreur du réseau neuronal pour une sortie (CEL)
*/
float compute_cross_entropy_loss(float* output, float* wanted_output, int len);

/*
* On considère que la sortie voulue comporte 10 éléments
*/
float* generate_wanted_output(float wanted_number);

#endif