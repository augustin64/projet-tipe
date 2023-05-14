#include <stdio.h>
#include <float.h>
#include <math.h>

#include "../common/include/colors.h"
#include "../common/include/utils.h"
#include "include/convolution.h"

#include "include/make.h"

#include "include/config.h"

#ifdef __CUDACC__
__host__ __device__
#endif

/* 
* Average Pooling
*/
#ifdef __CUDACC__
__global__ void make_average_pooling_kernel(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < output_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width
    int max_move = size - padding;
    int input_width = output_width*stride - 2*padding + size - stride;

    if (idx >= output_depth || idy >= output_width || idz >= output_width) {
        return;
    }

    int nb_elements = 0;
    float sum = 0;

    for (int a=-padding; a < max_move; a++) {
        for (int b=-padding; b < max_move; b++) {
            int idy_2 = stride*idy +a;
            int idz_2 = stride*idz +b;
            if (not_outside(idy_2, idz_2, 0, input_width)) {
                sum += input[idx][idy_2][idz_2];
                nb_elements++;
            }
        }
    }
    output[idx][idy][idz] = sum/(float)nb_elements;
}

void make_average_pooling_device(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(output_depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_average_pooling_kernel<<<gridSize, blockSize>>>(input, output, size, output_depth, output_width, stride, padding);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_average_pooling_cpu(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // input[output_depth][output_width+size-1][output_width+size-1]
    // output[output_depth][output_width][output_width]
    int max_move = size - padding;
    int input_width = output_width*stride - 2*padding + size - stride;

    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                float sum = 0.;
                int nb_elements = 0;
                for (int a=-padding; a < max_move; a++) {
                    for (int b=-padding; b < max_move; b++) {
                        int j_2 = stride*j +a;
                        int k_2 = stride*k +b;
                        if (not_outside(j_2, k_2, 0, input_width)) {
                            sum += input[i][j_2][k_2];
                            nb_elements++;
                        }
                    }
                }
                output[i][j][k] = sum/(float)nb_elements;
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    #ifndef __CUDACC__
    make_average_pooling_cpu(input, output, size, output_depth, output_width, stride, padding);
    #else
    make_average_pooling_device(input, output, size, output_depth, output_width, stride, padding);
    #endif
}





/* 
* Max Pooling
*/
#ifdef __CUDACC__
__global__ void make_max_pooling_kernel(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < output_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width
    int input_width = output_width*stride - 2*padding + size - stride;

    if (idx >= output_depth || idy >= output_width || idz >= output_width) {
        return;
    }

    int max_move = size - padding;
    float m = -FLT_MAX;
    float temp;

    for (int a=-padding; a < max_move; a++) {
        for (int b=-padding; b < max_move; b++) {
            int idy_2 = stride*idy +a;
            int idz_2 = stride*idz +b;
            if (not_outside(idy_2, idz_2, 0, input_width)) {
                temp = input[idx][idy_2][idz_2];
                m = m > temp ? m : temp; // max(m, temp)
            }
        }
    }
    output[idx][idy][idz] = m;
}

void make_max_pooling_device(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(output_depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_max_pooling_kernel<<<gridSize, blockSize>>>(input, output, size, output_depth, output_width, stride, padding);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_max_pooling_cpu(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    // input[output_depth][output_width+size-1][output_width+size-1]
    // output[output_depth][output_width][output_width]
    int max_move = size - padding;
    int input_width = output_width*stride - 2*padding + size - stride;
    float m;
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                m = -FLT_MAX;
                for (int a=-padding; a < max_move; a++) {
                    for (int b=-padding; b < max_move; b++) {
                        int j_2 = stride*j +a;
                        int k_2 = stride*k +b;
                        if (not_outside(j_2, k_2, 0, input_width)) {
                            m = fmaxf(m, input[i][j_2][k_2]);
                        }
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
void make_max_pooling(float*** input, float*** output, int size, int output_depth, int output_width, int stride, int padding) {
    #ifndef __CUDACC__
    make_max_pooling_cpu(input, output, size, output_depth, output_width, stride, padding);
    #else
    make_max_pooling_device(input, output, size, output_depth, output_width, stride, padding);
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
__global__ void make_dense_linearized_kernel(float** weights, float* bias, float*** input, float* output, int input_depth, int input_width, int size_output) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < size_output

    if (idx >= size_output) {
        return;
    }
    float f = bias[idx];

    for (int i=0; i < input_depth; i++) {
        for (int j=0; j < input_width; j++) {
            for (int k=0; k < input_width; k++) {
                f += input[i][j][k]*weights[k + j*input_width + i*input_depth][idx];
            }
        }
    }
    output[idx] = f;
}

void make_dense_linearized_device(Kernel_nn* kernel, float*** input, float* output, int input_depth, int input_width, int size_output) {
    // Make computation
    dim3 gridSize(i_div_up(size_output, BLOCKSIZE_x*BLOCKSIZE_y), 1, 1);
    dim3 blockSize(BLOCKSIZE_x*BLOCKSIZE_y, 1, BLOCKSIZE_z);

    make_dense_linearized_kernel<<<gridSize, blockSize>>>(kernel->weights, kernel->bias, input, output, input_depth, input_width, size_output);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_dense_linearized_cpu(Kernel_nn* kernel, float*** input, float* output, int input_depth, int input_width, int size_output) {
    // input[input_depth][input_width][input_width]
    // output[size_output]
    float f;

    for (int l=0; l < size_output; l++) {
        f = kernel->bias[l];
        for (int i=0; i < input_depth; i++) {
            for (int j=0; j < input_width; j++) {
                for (int k=0; k < input_width; k++) {
                    f += input[i][j][k]*kernel->weights[k + j*input_width + i*input_depth][l];
                }
            }
        }
        output[l] = f;
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void make_dense_linearized(Kernel_nn* kernel, float*** input, float* output, int input_depth, int input_width, int size_output) {
    #ifndef __CUDACC__
    make_dense_linearized_cpu(kernel, input, output, input_depth, input_width, size_output);
    #else
    make_dense_linearized_device(kernel, input, output, input_depth, input_width, size_output);
    #endif
}
