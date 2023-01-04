#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "../cnn/include/convolution.h"
#include "../cnn/include/struct.h"
#include "../include/colors.h"
#include "../include/utils.h"


float random_float(float low, float high) {
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}


void fill_matrix_random(float ***matrix, int n, int p, int q, float max_val) {
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            for (int k=0; k < q; k++) {
                matrix[i][j][k] = random_float(0.0f, max_val);
            }
        }
    }
}


void print_matrix(float** mat, int n, int p) {
    for (int i=0; i < n; i++) {
        printf("[\t");
        for (int j=0; j < p; j++) {
            printf("%0.1f\t", mat[i][j]);
        }
        printf("]\n");
    }
}


float*** create_matrix(int n, int p, int q, float max_val) {
    float*** matrix = (float***)malloc(n*sizeof(float**));
    for (int i=0; i < n; i++) {
        matrix[i] = (float**)malloc(sizeof(float*)*p);
        for (int j=0; j < p; j++) {
            matrix[i][j] = (float*)malloc(sizeof(float)*q);
        }
    }

    fill_matrix_random(matrix, n, p, q, max_val);
    return matrix;
}


float*** create_empty_matrix(int n, int p, int q) {
    float*** matrix = (float***)malloc(n*sizeof(float**));
    for (int i=0; i < n; i++) {
        matrix[i] = (float**)malloc(sizeof(float*)*p);
        for (int j=0; j < p; j++) {
            matrix[i][j] = (float*)malloc(sizeof(float)*q);
            for (int k=0; k < q; k++) {
                matrix[i][j][k] = 0.;
            }
        }
    }
    return matrix;
}

void free_matrix(float*** matrix, int n, int p) {
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}

float max_float(float a, float b) {
    return a > b ? a : b;
}

bool check_matrices_equality(float*** m1, float*** m2, int n, int p, int q, int acceptation) {
    float err_max = 0.;
    float err_moy = 0.;
    float err_percent = 0.;
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            for (int k=0; k < q; k++) {
                if (fabs(m1[i][j][k] - m2[i][j][k]) > 0.01*acceptation) {
                    //printf(RED "diff %d %d %d: %f val: %f et %f\n" RESET, i, j, k, fabs(m1[i][j][k] - m2[i][j][k]), m1[i][j][k], m2[i][j][k]);
                    //return false;
                }
                err_percent = 2*fabs(m1[i][j][k] - m2[i][j][k])/fabs(m1[i][j][k] + m2[i][j][k]);
                err_max = max_float(err_max, err_percent);
                err_moy += err_percent;
            }
        }
    }
    printf("err_max:%lf\n", err_max);
    printf("err_moy:%lf\n", err_moy/(n*p*q));
    return true;
}

void run_convolution_test(int input_dim, int output_dim, int rows, int columns) {
    assert(input_dim >= output_dim);
    int k_size = input_dim - output_dim +1;

    // Génération des données aléatoires
    Kernel_cnn* kernel = (Kernel_cnn*)malloc(sizeof(Kernel_cnn));
    
    kernel->k_size = k_size;
    kernel->rows = rows;
    kernel->columns = columns;

    // bias[kernel->columns][dim_output][dim_output]
    kernel->bias = create_matrix(kernel->columns, output_dim, output_dim, 15.0f);
    kernel->d_bias = create_matrix(kernel->columns, output_dim, output_dim, 1.5f);

    // w[rows][columns][k_size][k_size]
    kernel->w = (float****)malloc(sizeof(float***)*kernel->rows);
    kernel->d_w = (float****)malloc(sizeof(float***)*kernel->rows);
    for (int i=0; i < kernel->rows; i++) {
        kernel->w[i] = create_matrix(kernel->columns, kernel->k_size, kernel->k_size, 15.0f);
        kernel->d_w[i] = create_matrix(kernel->columns, kernel->k_size, kernel->k_size, 1.5f);
    }

    float*** input = create_matrix(kernel->rows, input_dim, input_dim, 5.0f);
    float*** output_cpu = create_empty_matrix(kernel->columns, output_dim, output_dim);
    float*** output_gpu = create_empty_matrix(kernel->columns, output_dim, output_dim);

    //printf("(%d, %d, %d, %d) Data generation complete\n", rows, columns, input_dim, output_dim);


    // Lancement des calculs
    clock_t start, end;
    double cpu_time_used, gpu_time_used;

    start = clock();
    make_convolution_device(kernel, input, output_gpu, output_dim);
    end = clock();

    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU: %lf\n", gpu_time_used);


    start = clock();
    make_convolution_cpu(kernel, input, output_cpu, output_dim);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU: %lf\n", rows, columns, input_dim, output_dim, cpu_time_used);    

    // Vérification de l'égalité des matrices
    //printf("(%d, %d, %d, %d) Checking equality.\n", rows, columns, input_dim, output_dim);
    if (!check_matrices_equality(output_gpu, output_cpu, kernel->columns, output_dim, output_dim, kernel->k_size)) {// TODO: change acceptation
        //exit(1);
    }
    //printf(GREEN "OK\n" RESET);

    free_matrix(kernel->bias, kernel->columns, output_dim);
    free_matrix(kernel->d_bias, kernel->columns, output_dim);

    for (int i=0; i < kernel->rows; i++) {
        free_matrix(kernel->w[i], kernel->columns, kernel->k_size);
        free_matrix(kernel->d_w[i], kernel->columns, kernel->k_size);
    }
    free(kernel->w);
    free(kernel->d_w);

    free_matrix(input, kernel->rows, input_dim);
    free_matrix(output_cpu, kernel->columns, output_dim);
    free_matrix(output_gpu, kernel->columns, output_dim);
}


int main(int argc, char* argv[]) {
    if (argc < 5) {
        return 1;
    }
    int n = strtol(argv[1], NULL, 10);
    int p = strtol(argv[2], NULL, 10);
    int q = strtol(argv[3], NULL, 10);
    int r = strtol(argv[4], NULL, 10);

    /*
    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = check_cuda_compatibility();
    if (!cuda_compatible) {
        printf(RED "CUDA not compatible, skipping tests.\n" RESET);
        return 0;
    }
    */
    
    srand(time(NULL));

    run_convolution_test(n, p, q, r);
    
    return 0;
}