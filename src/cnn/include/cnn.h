#include "struct.h"

#ifndef DEF_MAIN_H
#define DEF_MAIN_H

/*
* Renvoie l'indice de l'élément de valeur maximale dans un tableau de flottants
* Utilisé pour trouver le neurone le plus activé de la dernière couche (résultat de la classification)
*/
int indice_max(float* tab, int n);

/*
* Renvoie si oui ou non (1 ou 0) le neurone va être abandonné
*/
int will_be_drop(int dropout_prob);

/*
* Écrit une image 28*28 au centre d'un tableau 32*32 et met à 0 le reste
*/
void write_image_in_network_32(int** image, int height, int width, float** input, bool random_offset);

/*
* Écrit une image linéarisée de img_width*img_width*img_depth pixels dans un tableau de taille size_input*size_input*3
* Les conditions suivantes doivent être respectées:
* - l'image est au plus de la même taille que input
* - la différence de taille entre input et l'image doit être un multiple de 2 (pour centrer l'image)
*/
void write_256_image_in_network(unsigned char* image, int img_width, int img_depth, int input_width, float*** input);

/*
* Propage en avant le cnn. Le dropout est actif que si le réseau est en phase d'apprentissage.
* 
*/
void forward_propagation(Network* network);

/*
* Propage en arrière le cnn
*/
void backward_propagation(Network* network, int wanted_number);

/*
* Implémente le dropout durant l'apprentissage en suivant le papier de recherche suivant:
* https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
* Cela se fait en mettant à 0 chaque valeur de l'input avec une probabilité de dropout%
*/
void drop_neurones(float*** input, int depth, int dim1, int dim2, int dropout);

/*
* Copie les données de output dans output_z (Sachant que les deux matrices ont les mêmes dimensions)
*/
void copy_input_to_input_z(float*** output, float*** output_z, int output_depth, int output_rows, int output_columns);

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
float* generate_wanted_output(int wanted_number, int size_output);

#endif