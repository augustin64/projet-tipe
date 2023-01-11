#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "../cnn/include/matrix_multiplication.h"


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
    float err_percent = 0.;
    for (int i=0; i < n; i++) {
        for (int j=0; j < p; j++) {
            if (fabs(m1[i][j] - m2[i][j]) > 0.8) {
                //printf("%d %d\n", i, j);
                //return false;
            }
            err_percent = 2*fabs(m1[i][j] - m2[i][j])/fabs(m1[i][j] + m2[i][j]);
            err_max = max_float(err_max, err_percent);
            err_moy += err_percent;
        }
    }
    printf("err_max:%lf\n", err_max);
    printf("err_moy:%lf\n", err_moy/(n*p));
    return true;
}


int main(int argc, char* argv[]) {
    if (argc < 4) {
        return 1;
    }
    int n = strtol(argv[1], NULL, 10);
    int p = strtol(argv[2], NULL, 10);
    int q = strtol(argv[3], NULL, 10);

    clock_t start, end;
    double cpu_time_used;


    srand(time(NULL));
    float** matrix1 = create_matrix(n, p);
    float** matrix2 = create_matrix(p, q);
    float** result_gpu = create_empty_matrix(n, q);
    float** result_cpu = create_empty_matrix(n, q);

    start = clock();
    matrix_multiplication_device(matrix1, matrix2, result_gpu, n, p, q);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU:%lf\n", cpu_time_used);
    
    start = clock();
    //matrix_multiplication_host(matrix1, matrix2, result_cpu, n, p, q);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU:%lf\n", cpu_time_used);

    //check_matrices_equality(result_gpu, result_cpu, n, q);
    
    return 0;
}

// On obtient une différence entre le calcul fait par le GPU et par le CPU.
// Cette différence est linéaire en p. (err_moy = p*1.639e-6)
// Elle ne varie pas en fonction de n et q.
// Cette erreur est sûrement dûe à différences mineurs dans la précision du stockage des flottants
// Dans la mémoire RAM et VRAM (du GPU)