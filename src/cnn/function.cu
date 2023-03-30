#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../include/colors.h"
#include "../include/utils.h"

#include "include/function.h"

#include "include/config.h"

//* Identity
#ifdef __CUDACC__
__device__ float device_identity(float x) {
    return x;
}

__device__ float device_identity_derivative(float x) {
    (void)x;
    return 1;
}
#endif

float identity(float x) {
    return x;
}

float identity_derivative(float x) {
    (void)x;
    return 1;
}


//* Sigmoid
#ifdef __CUDACC__
__device__ float device_sigmoid(float x) {
    return 1/(1 + exp(-x));
}

__device__ float device_sigmoid_derivative(float x) {
    float tmp = exp(-x);
    return tmp/((1+tmp)*(1+tmp));
}
#endif

float sigmoid(float x) {
    return 1/(1 + exp(-x));
}

float sigmoid_derivative(float x) {
    float tmp = exp(-x);
    return tmp/((1+tmp)*(1+tmp));
}


//* RELU
#ifdef __CUDACC__
__device__ float device_relu(float x) {
    return fmaxf(0, fminf(x, RELU_CLIP_VALUE));
}

__device__ float device_relu_derivative(float x) {
    if (x > 0)
        return 1;
    return 0;
}
#endif

float relu(float x) {
    return fmaxf(0, fminf(x, RELU_CLIP_VALUE));
}

float relu_derivative(float x) {
    if (x > 0)
        return 1;
    return 0;
}


//* Leaky RELU
#ifdef __CUDACC__
__device__ float device_leaky_relu(float x) {
    if (x>0)
        return fminf(x, RELU_CLIP_VALUE);
    return x*LEAKER;
}

__device__ float device_leaky_relu_derivative(float x) {
    if (x > 0)
        return 1;
    return LEAKER;
}
#endif

float leaky_relu(float x) {
    if (x>0)
        return fminf(x, RELU_CLIP_VALUE);
    return x*LEAKER;
}

float leaky_relu_derivative(float x) {
    if (x > 0)
        return 1;
    return LEAKER;
}


//* Tanh
#ifdef __CUDACC__
__device__ float device_tanh_(float x) {
    return tanh(x);
}

__device__ float device_tanh_derivative(float x) {
    float a = tanh(x);
    return 1 - a*a;
}
#endif

float tanh_(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    float a = tanh(x);
    return 1 - a*a;
}




#ifdef __CUDACC__
/*
 * Définition des pointeurs de fonctions pour CUDA
 * voir https://stackoverflow.com/a/15646771
*/
__device__ funcPtr ptr_sigmoid = device_sigmoid;
__device__ funcPtr ptr_relu = device_relu;
__device__ funcPtr ptr_leaky_relu = device_leaky_relu;
__device__ funcPtr ptr_tanh = device_tanh_;
__device__ funcPtr ptr_identity = device_identity;

__device__ funcPtr ptr_identity_derivative = device_identity_derivative;
__device__ funcPtr ptr_sigmoid_derivative = device_sigmoid_derivative;
__device__ funcPtr ptr_relu_derivative = device_relu_derivative;
__device__ funcPtr ptr_leaky_relu_derivative = device_leaky_relu_derivative;
__device__ funcPtr ptr_tanh_derivative = device_tanh_derivative;
#endif



void apply_softmax_input(float ***input, int depth, int rows, int columns) {
    float m = -FLT_MAX;
    float sum=0;
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                m = fmaxf(m, input[i][j][k]);
            }
        }
    }
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = exp(m-input[i][j][k]);
                sum += input[i][j][k];
            }
        }
    }
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = input[i][j][k]/sum;
            }
        }
    }
}


/* 
* Apply function on input
*/
#ifdef __CUDACC__
__global__ void apply_function_input_kernel(funcPtr f, float*** input, int depth, int rows, int columns) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < rows
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < columns

    if (idx >= depth || idy >= rows || idz >= columns) {
        return;
    }

    input[idx][idy][idz] = (*f)(input[idx][idy][idz]);
}


void apply_function_input_device(int activation, float*** input, int depth, int rows, int columns) {
    // Make computation
    dim3 gridSize(i_div_up(depth, BLOCKSIZE_x), i_div_up(rows, BLOCKSIZE_y), i_div_up(columns, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    funcPtr activation_function = get_activation_function_cuda(activation);

    apply_function_input_kernel<<<gridSize, blockSize>>>(activation_function, input, depth, rows, columns);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void apply_function_input_cpu(int activation, float*** input, int depth, int rows, int columns) {
    funcPtr f = get_activation_function(activation);

    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = (*f)(input[i][j][k]);
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void apply_function_input(int activation, float*** input, int depth, int rows, int columns) {
    #ifndef __CUDACC__
    apply_function_input_cpu(activation, input, depth, rows, columns);
    #else
    apply_function_input_device(activation, input, depth, rows, columns);
    #endif
}

void apply_function_to_matrix(int activation, float*** input, int depth, int dim) {
    if (activation == SOFTMAX) {
        return apply_softmax_input(input, depth, dim, dim);
    }
    if (activation >= 1) { // Exclude negative values (derivative)
        return apply_function_input(activation, input, depth, dim, dim);
    }
    printf_error((char*)"fonction d'activation inconnue (apply_function_to_matrix): ");
    printf("%d\n", activation);
}


void apply_function_to_vector(int activation, float*** input, int dim) {
    if (activation == SOFTMAX) {
        return apply_softmax_input(input, 1, 1, dim);
    }
    if (activation >= 1) { // Exclude negative values (derivative)
        return apply_function_input(activation, input, 1, 1, dim);
    }
    printf_error((char*)"fonction d'activation inconnue (apply_function_to_vector): ");
    printf("%d\n", activation);
}


funcPtr get_activation_function(int activation) {
    switch (activation) {
        case RELU:
            return &relu;
        case -RELU:
            return &relu_derivative;

        case IDENTITY:
            return &identity;
        case -IDENTITY:
            return &identity_derivative;

        case SIGMOID:
            return &sigmoid;
        case -SIGMOID:
            return &sigmoid_derivative;
        
        case LEAKY_RELU:
            return &leaky_relu;
        case -LEAKY_RELU:
            return &leaky_relu_derivative;

        case TANH:
            return &tanh_;
        case -TANH:
            return &tanh_derivative;

        case SOFTMAX:
            printf_error((char*)"impossible de renvoyer la fonction softmax\n");
            return NULL;
        case -SOFTMAX:
            printf_error((char*)"impossible de renvoyer la dérivée de la fonction softmax\n");
            return NULL;

        default:
            printf_error((char*)"fonction d'activation inconnue (get_activation_function_cuda): ");
            printf("%d\n", activation);
            return NULL;
    }
}


#ifdef __CUDACC__
extern "C"
funcPtr get_activation_function_cuda(int activation) {
    funcPtr host_function;
    
    switch (activation) {
        case RELU:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_relu, sizeof(funcPtr)));
            break;
        case -RELU:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_relu_derivative, sizeof(funcPtr)));
            break;

        case IDENTITY:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_identity, sizeof(funcPtr)));
            break;
        case -IDENTITY:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_identity_derivative, sizeof(funcPtr)));
            break;

        case SIGMOID:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_sigmoid, sizeof(funcPtr)));
            break;
        case -SIGMOID:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_sigmoid_derivative, sizeof(funcPtr)));
            break;
        
        case LEAKY_RELU:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_leaky_relu, sizeof(funcPtr)));
            break;
        case -LEAKY_RELU:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_leaky_relu_derivative, sizeof(funcPtr)));
            break;

        case TANH:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_tanh, sizeof(funcPtr)));
            break;
        case -TANH:
            gpuErrchk( cudaMemcpyFromSymbol(&host_function, ptr_tanh_derivative, sizeof(funcPtr)));
            break;

        case SOFTMAX:
            printf_error((char*)"impossible de renvoyer la fonction softmax\n");
            return NULL;
        case -SOFTMAX:
            printf_error((char*)"impossible de renvoyer la dérivée de la fonction softmax\n");
            return NULL;

        default:
            printf_error((char*)"fonction d'activation inconnue (get_activation_function_cuda): ");
            printf("%d\n", activation);
            return NULL;
    }
    return host_function;
}
#endif