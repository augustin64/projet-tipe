#include "struct.h"

#ifndef DEF_MAKE_H
#define DEF_MAKE_H

/*
* Effectue une convolution sans stride sur le processeur
*/
void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim);

/*
* Effectue la convolution sur le CPU ou GPU
*/
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim);
/*
* Effectue un average pooling avec stride=size
*/
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim);

/*
* Effectue une full connection
*/
void make_dense(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output);

/*
* Effectue une full connection qui passe d'une matrice à un vecteur
*/
void make_dense_linearised(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output);

#endif