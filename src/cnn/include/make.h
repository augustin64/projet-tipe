#include "struct.h"

#ifndef DEF_MAKE_H
#define DEF_MAKE_H

/*
* Effectue une convolution sans stride
*/
void make_convolution(float*** input, Kernel_cnn* kernel, float*** output, int output_dim);

/*
* Effecute un average pooling avec stride=size
*/
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim);

/*
* Effecute une full connection
*/
void make_fully_connected(float* input, Kernel_nn* kernel, float* output, int size_input, int size_output);

#endif