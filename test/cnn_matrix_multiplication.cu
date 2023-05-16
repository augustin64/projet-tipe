#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../src/cnn/include/matrix_multiplication.h"
#include "../src/common/include/memory_management.h"
#include "../src/common/include/colors.h"
#include "../src/common/include/utils.h"


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
    float** matrix = (float**)nalloc(n, sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)nalloc(p, sizeof(float));
    }

    fill_matrix_random(matrix, n, p);
    return matrix;
}


float** create_empty_matrix(int n, int p) {
    float** matrix = (float**)nalloc(n, sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)nalloc(p, sizeof(float));
        for (int j=0; j < p; j++) {
            matrix[i][j] = 0.;
        }
    }
    return matrix;
}


bool check_matrices_equality(float** m1, float** m2, int n, int p, int acceptation) {
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            if (fabs(m1[i][j] - m2[i][j]) > 0.01*acceptation) {
                return false;
            }
        }
    }
    return true;
}

void run_matrices_test(int n, int p, int q) {
    double start_time, end_time;
    double cpu_time_used, gpu_time_used;

    float** matrix1 = create_matrix(n, p);
    float** matrix2 = create_matrix(p, q);
    float** result_gpu = create_empty_matrix(n, q);
    float** result_cpu = create_empty_matrix(n, q);

    printf("(%d,%d)x(%d,%d) Data generation complete.\n", n, p, p, q);
    start_time = omp_get_wtime();
    matrix_multiplication_device(matrix1, matrix2, result_gpu, n, p, q);
    end_time = omp_get_wtime();

    cpu_time_used = end_time - start_time;
    printf("(%d,%d)x(%d,%d) Time used for GPU: %lf seconds\n", n, p, p, q, cpu_time_used);
    
    start_time = omp_get_wtime();
    matrix_multiplication_host(matrix1, matrix2, result_cpu, n, p, q);
    end_time = omp_get_wtime();

    gpu_time_used = end_time - start_time;
    printf("(%d,%d)x(%d,%d) Time used for CPU: %lf seconds\n", n, p, p, q, gpu_time_used);

    // Vérification de l'égalité des matrices
    printf("(%d,%d)x(%d,%d) Checking equality.\n", n, p, p, q);
    if (!check_matrices_equality(result_gpu, result_cpu, n, q, p)) {
        exit(1);
    }
    printf(GREEN "OK\n" RESET);

    // On libère l'espace mémoire alloué
    for (int i=0; i < n; i++) {
        gree(matrix1[i], false);
    }
    gree(matrix1, false);

    for (int i=0; i < p; i++) {
        gree(matrix2[i], false);
    }
    gree(matrix2, false);

    for (int i=0; i < n; i++) {
        gree(result_cpu[i], false);
    }
    gree(result_cpu, false);

    for (int i=0; i < n; i++) {
        gree(result_gpu[i], false);
    }
    gree(result_gpu, false);
}


int main() {
    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = cuda_setup(true);
    if (!cuda_compatible) {
        printf(RED "CUDA not compatible, skipping tests.\n" RESET);
        return 0;
    }
    printf(GREEN "OK\n" RESET);

    srand(time(NULL));
    run_matrices_test(200, 1000, 200);
    run_matrices_test(200, 1000, 20);
    run_matrices_test(20, 1000, 200);
    
    return 0;
}

// On obtient une différence entre le calcul fait par le GPU et par le CPU.
// Cette différence est linéaire en p. (err_moy = p*1.639e-6)
// Elle ne varie pas en fonction de n et q.
// Cette erreur est sûrement dûe à différences mineurs dans la précision du stockage des flottants
// dans la mémoire RAM et VRAM (du GPU)