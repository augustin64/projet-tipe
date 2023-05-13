#include "struct.h"

#ifndef DEF_MAKE_H
#define DEF_MAKE_H

#ifdef __CUDACC__
__host__ __device__
#endif
/*
* On renvoie true si et seulement si _ et _:
* lower_bound <= y < upper_bound
* lower_bound <= x < upper_bound
*/
int pooling_not_outside(int x, int y, int lower_bound, int upper_bound);

/*
* Effectue la propagation d'une convolution avec stride et padding choisis sur le processeur
*/
void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding);

/*
* Effectue la propagation d'une convolution avec stride et padding choisis sur le CPU ou GPU
*/
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Effectue propagation d'average pooling avec stride et padding choisis
*/
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Effectue propagation de max pooling avec stride et padding choisis
*/
void make_max_pooling(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Effectue la propagation d'une couche dense
*/
void make_dense(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Effectue la propagation d'une couche dense qui passe d'une matrice à un vecteur
*/
void make_dense_linearized(Kernel_nn* kernel, float*** input, float* output, int input_depth, int input_width, int size_output);

#endif