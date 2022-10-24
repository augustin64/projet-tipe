#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef DEF_MNIST_H
#define DEF_MNIST_H


uint32_t swap_endian(uint32_t val);

/*
* Renvoie le nombre d'éléments dans un set de labels au format IDX
*/
uint32_t read_mnist_labels_nb_images(char* filename);

/*
* Lit et renvoie une image dans un fichier spécifié par ptr
*/
int** read_image(unsigned int width, unsigned int height, FILE* ptr);

/*
* Renvoie les paramètres [nb_elem, width, height]
*/
int* read_mnist_images_parameters(char* filename);

/*
* Charge dans la mémoire des images du set de données MNIST au format IDX
*/
int*** read_mnist_images(char* filename);

/*
* Charge dans la mémoire des labels du set de données MNIST au format IDX
*/
unsigned int* read_mnist_labels(char* filename);

#endif