#include <stdio.h>
#include <float.h>

#include "include/convolution.h"
#include "../include/colors.h"
#include "../include/utils.h"

#include "include/make.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 8
#define BLOCKSIZE_z 8

float max_flt(float a, float b) {
    // Return the max between the two floats
    if (a > b) {
        return a;
    }
    return b;
}





/* 
* Average Pooling
*/
#ifdef __CUDACC__
__global__ void make_average_pooling_kernel(float*** input, float*** output, int size, int output_depth, int output_width) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < output_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width
    int n = size*size;

    if (idx >= output_depth || idy >= output_width || idz >= output_width) {
        return;
    }

    float sum = 0;

    for (int a=0; a < size; a++) {
        for (int b=0; b < size; b++) {
            sum += input[idx][size*idy +a][size*idz +b];
        }
    }
    output[idx][idy][idz] = sum/(float)n;
}

void make_average_pooling_device(float*** input, float*** output, int size, int output_depth, int output_width) {
    // Make computation
    dim3 gridSize(i_div_up(output_depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_average_pooling_kernel<<<gridSize, blockSize>>>(input, output, size, output_depth, output_width);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_average_pooling_cpu(float*** input, float*** output, int size, int output_depth, int output_width) {
    // input[output_depth][output_width+size-1][output_width+size-1]
    // output[output_depth][output_width][output_width]
    float sum;
    int n = size*size;

    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                sum = 0;
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        sum += input[i][size*j +a][size*k +b];
                    }
                }
                output[i][j][k] = sum/(float)n;
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_width) {
    #ifndef __CUDACC__
    make_average_pooling_cpu(input, output, size, output_depth, output_width);
    #else
    make_average_pooling_device(input, output, size, output_depth, output_width);
    #endif
}





/* 
* Max Pooling
*/
#ifdef __CUDACC__
__global__ void make_max_pooling_kernel(float*** input, float*** output, int size, int output_depth, int output_width) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < output_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width

    if (idx >= output_depth || idy >= output_width || idz >= output_width) {
        return;
    }

    float m = FLT_MIN;
    float temp;

    for (int a=0; a < size; a++) {
        for (int b=0; b < size; b++) {
            temp = input[idx][size*idy +a][size*idz +b];
            m = m > temp ? m : temp; // max(m, temp)
        }
    }
    output[idx][idy][idz] = m;
}

void make_max_pooling_device(float*** input, float*** output, int size, int output_depth, int output_width) {
    // Make computation
    dim3 gridSize(i_div_up(output_depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_max_pooling_kernel<<<gridSize, blockSize>>>(input, output, size, output_depth, output_width);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_max_pooling_cpu(float*** input, float*** output, int size, int output_depth, int output_width) {
    // input[output_depth][output_width+size-1][output_width+size-1]
    // output[output_depth][output_width][output_width]
    float m;
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                m = FLT_MIN;
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        m = max_flt(m, input[i][size*j +a][size*k +b]);
                    }
                }
                output[i][j][k] = m;
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_max_pooling(float*** input, float*** output, int size, int output_depth, int output_width) {
    #ifndef __CUDACC__
    make_max_pooling_cpu(input, output, size, output_depth, output_width);
    #else
    make_max_pooling_device(input, output, size, output_depth, output_width);
    #endif
}





/*
* Dense
*/
#ifdef __CUDACC__
__global__ void make_dense_kernel(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < size_output

    if (idx >= size_output) {
        return;
    }
    float f = kernel->bias[idx];

    for (int j=0; j < size_input; j++) {
        f += kernel->weights[j][idx]*input[j];
    }
    output[idx] = f;
}

void make_dense_device(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    // Make computation
    dim3 gridSize(i_div_up(size_output, BLOCKSIZE_x*BLOCKSIZE_y), 1, 1);
    dim3 blockSize(BLOCKSIZE_x*BLOCKSIZE_y, 1, BLOCKSIZE_z);

    make_dense_kernel<<<gridSize, blockSize>>>(kernel, input, output, size_input, size_output);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

#ifdef __CUDACC__
extern "C"
#endif
void make_dense_cpu(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    // input[size_input]
    // output[size_output]
    float f;

    for (int i=0; i < size_output; i++) {
        f = kernel->bias[i];
        for (int j=0; j < size_input; j++) {
            f += kernel->weights[j][i]*input[j];
        }
        output[i] = f;
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_dense(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    #ifndef __CUDACC__
    make_dense_cpu(kernel, input, output, size_input, size_output);
    #else
    make_dense_device(kernel, input, output, size_input, size_output);
    #endif
}





/*
* Dense linearized
*/
#ifdef __CUDACC__
__global__ void make_dense_linearized_kernel(float** weights, float* bias, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < size_output

    if (idx >= size_output) {
        return;
    }
    float f = bias[idx];

    for (int i=0; i < depth_input; i++) {
        for (int j=0; j < dim_input; j++) {
            for (int k=0; k < dim_input; k++) {
                f += input[i][j][k]*weights[k + j*dim_input + i*depth_input][idx];
            }
        }
    }
    output[idx] = f;
}

void make_dense_linearized_device(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    // Make computation
    dim3 gridSize(i_div_up(size_output, BLOCKSIZE_x*BLOCKSIZE_y), 1, 1);
    dim3 blockSize(BLOCKSIZE_x*BLOCKSIZE_y, 1, BLOCKSIZE_z);

    make_dense_linearized_kernel<<<gridSize, blockSize>>>(kernel->weights, kernel->bias, input, output, depth_input, dim_input, size_output);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_dense_linearized_cpu(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    // input[depth_input][dim_input][dim_input]
    // output[size_output]
    float f;

    for (int l=0; l < size_output; l++) {
        f = kernel->bias[l];
        for (int i=0; i < depth_input; i++) {
            for (int j=0; j < dim_input; j++) {
                for (int k=0; k < dim_input; k++) {
                    f += input[i][j][k]*kernel->weights[k + j*dim_input + i*depth_input][l];
                }
            }
        }
        output[l] = f;
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_dense_linearized(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    #ifndef __CUDACC__
    make_dense_linearized_cpu(kernel, input, output, depth_input, dim_input, size_output);
    #else
    make_dense_linearized_device(kernel, input, output, depth_input, dim_input, size_output);
    #endif
}
