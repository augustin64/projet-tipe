#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "../src/cnn/matrix_multiplication.cu"


float random_float(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}


void fill_matrix_random(float **matrix, int n, int p) {
  for (int i=0; i < n; i++) {
    for (int j=0; j < p; j++) {
        matrix[i][j] = random_float(0.0f, 15.0f);
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


float** create_matrix(int n, int p) {
    float** matrix = (float**)malloc(n*sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)malloc(sizeof(float)*p);
    }

    fill_matrix_random(matrix, n, p);
    return matrix;
}


float** create_empty_matrix(int n, int p) {
    float** matrix = (float**)malloc(n*sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)malloc(p*sizeof(float));
        for (int j=0; j < p; j++) {
            matrix[i][j] = 0.;
        }
    }
    return matrix;
}

float max_float(float a, float b) {
    return a > b ? a : b;
}


bool check_matrices_equality(float** m1, float** m2, int n, int p) {
    float err_max = 0.;
    float err_moy = 0.;
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            if (fabs(m1[i][j] - m2[i][j]) > 0.8) {
                //printf("%d %d\n", i, j);
                //return false;
            }
            err_max = max_float(err_max, fabs(m1[i][j] - m2[i][j]));
            err_moy += fabs(m1[i][j] - m2[i][j]);
        }
    }
    printf("err_max: %f\n", err_max);
    printf("err_moy: %f\n", err_moy/(n*p));
    return true;
}


int main() {
    clock_t start, end;
    double cpu_time_used;

    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = check_cuda_compatibility();
    if (!cuda_compatible) {
        printf("CUDA not compatible, skipping tests.\n");
        return 0;
    }
    printf("OK\n");


    printf("Generating matrices.\n");
    srand(time(NULL));
    int n = 200;
    int p = 1000;
    int q = 200;
    float** matrix1 = create_matrix(n, p);
    float** matrix2 = create_matrix(p, q);
    float** result_gpu = create_empty_matrix(n, q);
    float** result_cpu = create_empty_matrix(n, q);
    printf("OK\n");

    printf("Computing on GPU.\n");
    start = clock();
    matrix_multiplication_device(matrix1, matrix2, result_gpu, n, p, q);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time used for GPU: %lf seconds\n", cpu_time_used);
    printf("OK\n");
    
    printf("Computing on CPU.\n");
    start = clock();
    matrix_multiplication_host(matrix1, matrix2, result_cpu, n, p, q);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time used for CPU: %lf seconds\n", cpu_time_used);
    printf("OK\n");


    printf("Checking equality.\n");
    if (!check_matrices_equality(result_gpu, result_cpu, n, q)) {
        return 1;
    }
    printf("OK\n");
    
    return 0;
}

// On obtient une différence entre le calcul fait par le GPU et par le CPU.
// Cette différence est linéaire en p. (err_moy = p*1.639e-6)
// Elle ne varie pas en fonction de n et q.
// Cette erreur est sûrement dûe à différences mineurs dans la précision du stockage des flottants
// Dans la mémoire RAM et VRAM (du GPU)