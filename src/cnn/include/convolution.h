#include "struct.h"

/*
On renvoie true si et seulement si _ et _:
lower_bound <= x < upper_bound
lower_bound <= y < upper_bound
*/
int convolution_not_outside(int x, int y, int lower_bound, int upper_bound);

/*
* Effectue la convolution naïvement sur le processeur
*/
void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding);

#ifdef __CUDACC__
/*
* Kernel de la convolution sur carte graphique
*/
__global__ void make_convolution_kernel(int k_size, int columns, int rows, float* bias, size_t pitch_bias, float**** weights, size_t pitch_weights, float*** input, size_t pitch_input, float*** output, size_t pitch_output, int output_dim, int stride, int padding);

/*
* Effectue la convolution naïvement sur la carte graphique
*/
void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding);
#endif

/*
* Détermine si la convolution peut-être faite sur la carte graphique au moment de la compilation
*/
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding);