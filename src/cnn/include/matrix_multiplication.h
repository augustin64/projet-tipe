#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#ifndef DEF_MATRIX_MULTIPLICATION_H
#define DEF_MATRIX_MULTIPLICATION_H


#ifdef __CUDACC__
/*
* Partie entière supérieure de a/b
*/
int i_div_up(int a, int b);

/*
* Fonction exécutée par chaque thread lancé dans `matrix_multiplication_device`
*/
__global__ void matrix_mul_kernel(float* Md, float* Nd, float* Pd, int n, int p, int q, size_t pitch_m, size_t pitch_n, size_t pitch_p);

/*
* Multiplication de deux matrices sur le GPU
*/
void matrix_multiplication_device(float** m1, float** m2, float** result, int n, int p, int q);
#endif

/*
* Vérification de la compatibilité CUDA
*/
bool check_cuda_compatibility();

/*
* Multiplication naïve de matrices sur le CPU (1 seul coeur)
*/
void matrix_multiplication_host(float** m1, float** m2, float** result, int n, int p, int q);

/*
* Multiplication de matrices (décide si il faut la faire sur CPU ou GPU)
*/
void matrix_multiplication(float** m1, float** m2, float** result, int n, int p, int q, bool use_cuda);
#endif