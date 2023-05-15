#include "struct.h"


/*
* Effectue la convolution naïvement sur le processeur
*/
void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding);

#ifdef __CUDACC__
/*
* Kernel de la convolution sur carte graphique
*/
__global__ void make_convolution_kernel(float**** weights, float*** bias, int k_size, int rows, int columns, float*** input, float*** output, int output_width, int stride, int padding);

/*
* Effectue la convolution naïvement sur la carte graphique
*/
void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding);
#endif

#ifdef __CUDACC__
extern "C"
#endif
/*
* Détermine si la convolution peut-être faite sur la carte graphique au moment de la compilation
*/
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding);