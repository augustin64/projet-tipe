#include <math.h>

#include "include/backpropagation.h"
#include "include/struct.h"

// The first layer needs to be a convolution or a fully connected one

int min(int a, int b) {
    return a<b?a:b;
}

int max(int a, int b) {
    return a > b ? a : b;
}

// Euh..... tout peut être faux à cause de la source
void rms_backward(float* input, float* input_z, float* output, int size) {
    /* Input et output ont la même taille
    On considère que la dernière couche a utilisée softmax */
    float sum=0;
    for (int i=0; i < size; i++)
        sum += exp(input_z[i]);
    float denom = sum*sum;
    for (int i=0; i < size; i++){
        float e_i = exp(input_z[i]);
        input[i] = 2*(input[i]-output[i])*((e_i*(sum-e_i))/denom); // ∂E/∂out_i * ∂out_i/∂net_i = 𝛿_i
    }
}

void backward_2d_pooling(float*** input, float*** output, int input_width, int output_width, int depth) {
    /* Input et output ont la même profondeur (depth) */
    // Inventé par moi-même (et que moi (vraiment que moi (lvdmm)))

    int size = output_width - input_width + 1; // Taille du pooling
    int n = size*size; // Nombre d'éléments dans le pooling

    for (int a=0; a < depth; a++)
        for (int b=0; b < input_width; b++)
            for (int c=0; c < input_width; c++)
                input[a][b][c] = 0;

    for (int i=0; i < depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        input[i][size*j +a][size*k +b] += output[i][j][k]/n;
                    }
                }
            }
        }
    }
}

void backward_fully_connected(Kernel_nn* ker, float* input, float* input_z, float* output, int size_input, int size_output, ptr d_function, int is_first) {
    // Bias
    for (int j=0; j < size_output; j++) {
        ker->d_bias[j] = output[j];
    }

    // Weights
    for (int i=0; i < size_input; i++) {
        for (int j=0; j < size_output; j++) {
            ker->d_weights[i][j] = input[i]*output[j];
        }
    }

    // Input
    if (is_first==1) {// Pas besoin de backpropager dans l'input
        return;
    }

    for (int i=0; i < size_input; i++) {
        float tmp=0;
        for (int j=0; j < size_output; j++) {
            tmp += output[j]*ker->weights[i][j];
        }
        input[i] = tmp*d_function(input_z[i]);
    }
}

void backward_linearisation(Kernel_nn* ker, float*** input, float*** input_z, float* output, int depth_input, int dim_input, int size_output, ptr d_function) {
    // Bias
    for (int j=0; j < size_output; j++) {
        ker->d_bias[j] += output[j];
    }

    // Weights
    int cpt = 0;
    for (int i=0; i < depth_input; i++) {
        for (int k=0; k < dim_input; k++) {
            for (int l=0; l < dim_input; l++) {
                for (int j=0; j < size_output; j++) {
                    ker->d_weights[cpt][j] += input[i][k][l]*output[j];
                }
                cpt++;
            }
        }
    }

    // Input
    cpt = 0;
    for (int i=0; i < depth_input; i++) {
        for (int k=0; k < dim_input; k++) {
            for (int l=0; l < dim_input; l++) {
                float tmp=0;
                for (int j=0; j < size_output; j++) {
                    tmp += output[j]*ker->weights[cpt][j];
                }
                input[i][k][l] = tmp*d_function(input_z[i][k][l]);
                cpt++;
            }
        }
    }
}

void backward_convolution(Kernel_cnn* ker, float*** input, float*** input_z, float*** output, int depth_input, int dim_input, int depth_output, int dim_output, ptr d_function, int is_first) {
    // Bias
    for (int i=0; i < depth_output; i++) {
        for (int j=0; j < dim_output; j++) {
            for (int k=0; k < dim_output; k++) {
                ker->d_bias[i][j][k] += output[i][j][k];
            }
        }
    }

    // Weights
    int k_size = dim_input - dim_output +1;

    for (int h=0; h < depth_input; h++) {
        for (int i=0; i < depth_output; i++) {
            for (int j=0; j < k_size; j++) {
                for (int k=0; k < k_size; k++) {
                    float tmp = 0;
                    for (int l=0; l < dim_output; l++) {
                        for (int m=0; m < dim_output; m++) {
                            tmp += input[h][l+j][m+k]*output[i][l][m];
                        }
                    }
                    ker->d_w[h][i][j][k] += tmp;
                }
            }
        }
    }

    // Input
    if (is_first==1) // Pas besoin de backpropager dans l'input
        return;
    int min_m, max_m, min_n, max_n;
    for (int i=0; i < depth_input; i++) {
        for (int j=0; j < dim_input; j++) {
            for (int k=0; k < dim_input; k++) {
                float tmp = 0;
                for (int l=0; l < depth_output; l++) {
                    min_m = max(0, k_size-1-j);
                    max_m = min(k_size, dim_input - j);
                    min_n = max(0, k_size-1-k);
                    max_n = min(k_size, dim_input-k);
                    for (int m=min_m; m < max_m; m++) {
                        for (int n=min_n; n < max_n; n++) {
                            tmp += output[l][j-k_size+m+1][k-k_size+n+1]*ker->w[i][l][m][n];
                        }
                    }
                }
                input[i][j][k] = tmp*d_function(input_z[i][j][k]);
            }
        }
    }
}
