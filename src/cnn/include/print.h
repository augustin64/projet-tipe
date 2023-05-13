#ifndef DEF_PRINT_H
#define DEF_PRINT_H

#include "struct.h"

/*
* Affiche le kernel d'une couche de convolution
*/
void print_kernel_cnn(Kernel_cnn* k, int input_depth, int input_width, int output_depth, int output_width);

/*
* Affiche une couche de pooling
*/
void print_pooling(int size, int pooling);

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