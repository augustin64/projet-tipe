#include "function.h"
#include "struct.h"

#ifndef DEF_BACKPROPAGATION_H
#define DEF_BACKPROPAGATION_H

/*
* Renvoie la valeur minimale entre a et b
*/
int min(int a, int b);

/*
* Renvoie la valeur maximale entre a et b
*/
int max(int a, int b);

/*
* Transfert les informations d'erreur de la sortie voulue à la sortie réelle
*/
void softmax_backward_mse(float* input, float* output, int size);

/*
* Transfert les informations d'erreur de la sortie voulue à la sortie réelle
* en considérant MSE (Mean Squared Error) comme fonction d'erreur
*/
void softmax_backward_cross_entropy(float* input, float* output, int size);

/*
* Transfert les informations d'erreur à travers une couche d'average pooling
* en considérant cross_entropy comme fonction d'erreur
*/
void backward_2d_pooling(float*** input, float*** output, int input_width, int output_width, int depth);

/*
* Transfert les informations d'erreur à travers une couche fully connected
*/
void backward_dense(Kernel_nn* ker, float* input, float* input_z, float* output, int size_input, int size_output, ptr d_function, int is_first);

/*
* Transfert les informations d'erreur à travers une couche de linéarisation
*/
void backward_linearisation(Kernel_nn* ker, float*** input, float*** input_z, float* output, int depth_input, int dim_input, int size_output, ptr d_function);

/*
* Transfert les informations d'erreur à travers un couche de convolution
*/
void backward_convolution(Kernel_cnn* ker, float*** input, float*** input_z, float*** output, int depth_input, int dim_input, int depth_output, int dim_output, ptr d_function, int is_first);

#endif
