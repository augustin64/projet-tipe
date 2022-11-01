#include "struct.h"

/*
* Effectue la convolution sur le processeur
*/
void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim);

#ifdef __CUDACC__
/*
* Partie entière supérieure de a/b
*/
int i_div_up(int a, int b);

/*
* Kernel de la convolution sur carte graphique
*/
__global__ void make_convolution_kernel(int k_size, int columns, int rows, float*** bias, size_t pitch_bias, float**** w, size_t pitch_w, float*** input, size_t pitch_input, float*** output, size_t pitch_output, int output_dim);

/*
* Effectue la convolution sur la carte graphique
*/
void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_dim);
#endif

/*
* Détermine si la convolution peut-être faite sur la carte graphique au moment de la compilation
*/
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim);