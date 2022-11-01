#ifndef DEF_PRINT_H
#define DEF_PRINT_H

#include "include/struct.h"

/*
* Affiche le kernel d'une couche de convolution
*/
void print_kernel_cnn(Kernel_cnn* k, int depth_input, int dim_input, int depth_output, int dim_output);

/*
* Affiche une couche de pooling
*/
void print_pooling(int size);

/*
* Affiche le kernel d'une couche de fully connected
*/
void print_kernel_nn(Kernel_nn* k, int size_input, int size_output);

/*
* Affiche une couche d'input
*/
void print_input(float*** input, int depth, int dim);

/*
* Affiche un cnn. Plus précisément:
* input, kernel_cnn, kernel_nn, pooling
*/
void print_cnn(Network* network);

#endif