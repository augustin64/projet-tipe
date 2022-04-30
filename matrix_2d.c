/* 
Version du module Matrice avec des matrices 2d
Une version 3d doit s'inspirer de celle-ci
*/





#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>


typedef struct Matrix {
    int rows; // Nombre de lignes de la matrice
    int columns; // Nombre de colonnes de la matrice
    float** value; // Tableau 2d comportant les valeurs de matrice

} Matrix;




float exp_float(float a);
float max_float(float a, float b);
float min_float(float a, float b);
Matrix* create_matrix(int nb_rows, int nb_columns);
void uniformity_matrix(Matrix* m, float v);
void print_matrix(Matrix* m);
float number_from_matrix(Matrix* m);
float max_in_matrix(Matrix* m);
void free_matrix(Matrix* m);
void product_of_a_scalar_matrix(Matrix* m, float scalar);
void sum_of_a_scalar_matrix(Matrix* m, float scalar);
Matrix* new_copy_matrix(Matrix* m);
Matrix* apply_function_new_matrix(Matrix* m, float (*f)(float));
void apply_function_matrix(Matrix* m, float (*f)(float));
void transpose_matrix(Matrix* m);
void add_matrix(Matrix* m1, Matrix* m2);
Matrix* product_matrix(Matrix* m1, Matrix* m2);
void max_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void min_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void average_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void valid_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void full_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void softmax_matrix(Matrix* m);
float quadratic_cost_matrix(Matrix* m, int i_number, int j_number);
void rotation_180_matrix(Matrix* m);



float exp_float(float a) {
    /* Renvoie l'exponentiel d'un flotant '*/
    return (float)exp(a);
}


float max_float(float a, float b) {
    /* Renvoie le max entre les deux flotants */
    return a>b?a:b;
}


float min_float(float a, float b) {
    /* Renvoie le min entre les deux flotants */
    return a<b?a:b;
}


Matrix* create_matrix(int nb_rows, int nb_columns) {
    /* Créé une matrice en lui allouant de la mémoire */
    Matrix* m = malloc(sizeof(Matrix));
    m->rows = nb_rows;
    m->columns = nb_columns;
    m->value = malloc(sizeof(float*)*m->rows);
    for (int i=0; i < m->rows; i++)
        m->value[i] = malloc(sizeof(float)*m->columns);
    return m;
}


void uniformity_matrix(Matrix* m, float v) {
    /* Insère la même valeur partout dans la matrice */
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            m->value[i][j] = v;
        }
    }
}


void print_matrix(Matrix* m) {
    /* Affiche la matrice */
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            if (j!=0) 
                printf(",");
            printf("%f ", m->value[i][j]);
        }
        printf("\n");
    }
}


float number_from_matrix(Matrix* m) {
    /* Renvoie la somme des éléments de la matrice */
    float tmp=0;
    for (int i=0; i < m->rows ; i++) {
        for (int j=0; j < m->columns; j++) {
            tmp += m->value[i][j];
        }
    }
    return tmp;
}


float max_in_matrix(Matrix* m) {
    /* Renvoie l'élément maximal de la matrice */
    float max_tmp = FLT_MIN;
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            max_tmp = max_float(max_tmp, m->value[i][j]);
        }
    }
    return max_tmp;
}


void free_matrix(Matrix* m) {
    /* Libère l'espace mémoire alloué à la matrice */
    for (int i=0; i < m->rows; i++)
        free(m->value[i]);
    free(m->value);
}


void product_of_a_scalar_matrix(Matrix* m, float scalar) {
    /* Multiplie la matrice par un scalaire */
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            m->value[i][j] *= scalar;
        }
    }
}


void sum_of_a_scalar_matrix(Matrix* m, float scalar) {
    /* Ajoute un scalaire à la matrice */
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            m->value[i][j] += scalar;
        }
    }
}


Matrix* new_copy_matrix(Matrix* m) {
    /* Renvoie une copie de la matrice */
    Matrix* new_m = create_matrix(m->rows, m->columns);
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            new_m->value[i][j] = m->value[i][j];
        }
    }
    return new_m;
}


void copy_matrix(Matrix* m1, Matrix* m2) {
    /* Copie le contenu de la matrice m1 dans la matrice m2 */
    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("Erreur, copie dans de deux matrices dont les dimensions diffèrent");
        return;
    }
    for (int i=0; i < m1->rows; i++) {
        for (int j=0; j < m2->columns; j++) {
            m2->value[i][j] = m1->value[i][j];
        }
    }
}


Matrix* apply_function_new_matrix(Matrix* m, float (*f)(float)) {
    /* Renvoie une matrice avec une fonction appliquée
    à tous les  éléments de l'ancienne matrice */
    Matrix* new_m  = create_matrix(m->rows, m->columns);
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m ->columns; j++) {
            new_m->value[i][j] = (*f)(m->value[i][j]);
        }
    }
    return new_m;
}


void apply_function_matrix(Matrix* m, float (*f)(float)) {
    /* Applique une fonction à tous les éléments de la matrice */
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m ->columns; j++) {
            m->value[i][j] = (*f)(m->value[i][j]);
        }
    }
}


void transpose_matrix(Matrix* m) {
    /* Transpose la matrice si c'est possible */
    if (m->rows != m->columns) {
        printf("Erreur, matrice non compatible avec la transposition");
        return;
    }
    float cpt;
    for (int i=0; i < m->rows; i++) {
        for (int j=i+1; j < m->columns; j++) {
            cpt = m->value[i][j];
            m->value[i][j] = m->value[j][i];
            m->value[j][i] = cpt;
        }
    }
}


void add_matrix(Matrix* m1, Matrix* m2) {
    /* Ajoute la matrice m1 à la matrice m2 */
    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("Erreur, matrices non compatibles avec la somme");
        return;
    }
    for (int i=0; i < m2->rows; i++) {
        for (int j=0; j < m2->columns; j++) {
            m2->value[i][j] += m1->value[i][j];
        }
    }
}


Matrix* product_matrix(Matrix* m1, Matrix* m2) {
    /* Renvoie une nouvelle matrice produit (classique)
    des deux matrices si les dimensions sont correctes*/
    if (m1->columns != m2->rows) {
        printf("Erreur, matrices non compatibles avec le produit");
        return NULL;
    }
    float cpt;
    Matrix* m = create_matrix(m1->rows, m2->columns);
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            cpt=0;
            for (int k=0; k < m2->rows; k++) {
                cpt += m1->value[i][j]* m2->value[k][j];
            }
            m->value[i][j] = cpt;
        }
    }
    return m;
}


void max_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de max pooling avec un décalage
    de (stride) pixels dans la matrice m_out */
    if (m_in->columns < kernel->columns || m_in->rows < kernel->rows) {
        printf("Erreur, kernel plus grand que la matrice dans max pooling");
        return;
    }
    if (((m_in->columns - kernel->columns)/stride)+1  != m_out->columns || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans max pooling");
        return;
    }
    int i, j, a ,b;
    float tmp;
    for (i=0; i < m_out->rows; i++) {
        for (j=0; j < m_out->columns; j++) {
            tmp = FLT_MIN;
            for (a=0; a < kernel->rows; a++) {
                for (b=0; b < kernel->columns; b++) {
                    tmp = max_float(tmp, m_in->value[i*stride +a][j*stride +b]);
                }
            }
            m_out->value[i][j] =  tmp;
        }
    }
}


void min_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de min pooling avec un décalage
    de (stride) pixels dans la matrice m_out */
    if (m_in->columns < kernel->columns || m_in->rows < kernel->rows) {
        printf("Erreur, kernel plus grand que la matrice dans min pooling");
        return;
    }
    if (((m_in->columns - kernel->columns)/stride)+1  != m_out->columns || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans min pooling");
        return;
    }
    int i, j, a ,b;
    float tmp;
    for (i=0; i < m_out->rows; i++) {
        for (j=0; j < m_out->columns; j++) {
            tmp = FLT_MAX;
            for (a=0; a < kernel->rows; a++) {
                for (b=0; b < kernel->columns; b++) {
                    tmp = min_float(tmp, m_in->value[i*stride +a][j*stride +b]);
                }
            }
            m_out->value[i][j] =  tmp;
        }
    }
}


void average_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de max pooling avec un décalage
    de (stride) pixels dans la matrice m_out */
    if (m_in->columns < kernel->columns || m_in->rows < kernel->rows) {
        printf("Erreur, kernel plus grand que la matrice dans average pooling");
        return;
    }
    if (((m_in->columns - kernel->columns)/stride)+1  != m_out->columns || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans average pooling");
        return;
    }
    int i, j, a, b, nb= kernel->rows*kernel->columns;
    for (i=0; i < m_out->rows; i++) {
        for (j=0; j < m_out->columns; j++) {
            m_out->value[i][j] = 0;
            for (a=0; a < kernel->rows; a++) {
                for (b=0; b < kernel->columns; b++) {
                    m_out->value[i][j] += m_in->value[i*stride +a][j*stride +b];
                }
            }
            m_out->value[i][j] =  m_out->value[i][j]/nb;
        }
    }
}


void valid_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Ajoute, la cross-correlation valide de m_in et
    kernel avec un décalage de stride, dans m_out */
    if (m_in->columns < kernel->columns || m_in->rows < kernel->rows) {
        printf("Erreur, kernel plus grand que la matrice dans valid cross-correlation");
        return;
    }
    if (((m_in->columns - kernel->columns)/stride)+1  != m_out->columns || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans valid cross-correlation");
        return;
    }
    int i, j, a, b;
    for (i=0; i < m_out->rows; i++) {
        for (j=0; j < m_out->columns; j++) {
            for (a=0; a < kernel->rows; a++) {
                for (b=0; b < kernel->columns; b++) {
                    m_out->value[i][j] += m_in->value[i*stride +a][j*stride +b]*kernel->value[a][b];
                }
            }
        }
    }
}


void full_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Ajoute, la cross-correlation entière de m_in et
    kernel avec un décalage de stride, dans m_out */
    int rows_k = kernel->rows-1;
    int columns_k = kernel->columns-1;
    if (m_in->columns < kernel->columns || m_in->rows < kernel->rows) {
        printf("Erreur, kernel plus grand que la matrice dans full cross-correlation");
        return;
    }
    if ((m_in->columns + 2*columns_k)/stride  != m_out->columns || (m_in->rows + 2*rows_k)/stride != m_out->rows) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans full cross-correlation");
        return;
    }
    int i, j, a, b, new_i, new_j;
    for (i=-rows_k; i < (m_out->rows + kernel->rows -1); i++) {
        for (j=-columns_k; j < (m_out->columns + kernel->columns -1); j++) {
            m_out->value[i+rows_k][j+columns_k] = 0;
            for (a=0; a < kernel->rows; a++) {
                for (b=0; b < kernel->columns; b++) {
                    new_i = i*stride +a;
                    new_j = j*stride +b;
                    if (new_i >= 0 || new_i < m_in->rows || new_j >= 0 || new_j < m_in->columns)
                         m_out->value[i+rows_k][j+columns_k] += m_in->value[i*stride +a][j*stride +b]*kernel->value[a][b];
                }
            }
        }
    }
}


void softmax_matrix(Matrix* m) {
    /* Applique la fonction softmax sur la matrice en changeant ses valeurs */
    float max = max_in_matrix(m);
    sum_of_a_scalar_matrix(m, (-1)*max);
    apply_function_matrix(m, exp_float);
    float sum = number_from_matrix(m);
    sum = 1/sum;
    product_of_a_scalar_matrix(m, sum);
}


float quadratic_cost_matrix(Matrix* m, int i_number, int j_number) {
    /* Renvoie l'erreur de la matrice où les valeurs sont des probabailités */
    float loss = 0;
    for (int i=0; i < m->rows; i++) {
        for (int j=0; j < m->columns; j++) {
            if (i==i_number && j==j_number)
                loss += (1-m->value[i][j])*(1-m->value[i][j]);
            else
                loss += m->value[i][j]*m->value[i][j];
        }
    }
    return loss;
}


void rotation_180_matrix(Matrix* m) {
    /* Modifie la matrice en pivotant ses valeurs de 180° */
    if (m->rows != m-> columns) {
        printf("Erreur, une matrice non carrée ne peut pas être retourner");
        return;
    }
    float tmp;
    int half_rows = m->rows/2;
    int max_r = m->rows-1;
    int max_c = m->columns-1;
    for (int i=0; i < m->rows; i++) {
        for (int j=i; j < m->columns; j++) {
            if (i!=j || i>=half_rows) {
                tmp = m->value[i][j];
                m->value[i][j] = m->value[max_r-i][max_c-j];
                m->value[max_r-i][max_c-j] = tmp;
            }
        }
    }
}


void valid__cross_correlation_step_forward(Matrix** layer_input, Matrix*** layer_kernel, Matrix** layer_bias, Matrix** layer_output, int len_layer, int depth_kernel, int stride) {
    /* Effectue une étape de la forward-propagation 
    à l'aide d'une cross-correlation valide */
    for (int i=0; i < depth_kernel; i++) {
        copy_matrix(layer_bias[i], layer_output[i]);

        for (int j=0; j < len_layer; j++) {
            valid_cross_correlation_matrix(layer_input[j], layer_kernel[i][j], stride, layer_output[j]);
        }
    }
}


void max_pooling_step_forward(Matrix** layer_input, Matrix*** layer_kernel, Matrix** layer_bias, Matrix** layer_output, int len_layer, int depth_kernel, int stride) {
    /* Effectue une étape de la forward-propagation 
    à l'aide d'un max_pooling */
    for (int i=0; i < depth_kernel; i++) {
        copy_matrix(layer_bias[i], layer_output[i]);

        for (int j=0; j < len_layer; j++) {
            max_pooling_matrix(layer_input[j], layer_kernel[i][j], stride, layer_output[j]);
        }
    }
}


void average_pooling_step_forward(Matrix** layer_input, Matrix*** layer_kernel, Matrix** layer_bias, Matrix** layer_output, int len_layer, int depth_kernel, int stride) {
    /* Effectue une étape de la forward-propagation 
    à l'aide d'un average_pooling */
    for (int i=0; i < depth_kernel; i++) {
        copy_matrix(layer_bias[i], layer_output[i]);

        for (int j=0; j < len_layer; j++) {
            average_pooling_matrix(layer_input[j], layer_kernel[i][j], stride, layer_output[j]);
        }
    }
}


void reshape_step_forward(Matrix** layer_input, Matrix*** layer_kernel, Matrix** layer_bias, Matrix** layer_output, int len_layer, int depth_kernel, int stride) {
    /* Effectue une étape de la forward-propagation 
    en redimensionnant la matrice */
    for (int i=0; i < depth_kernel; i++) {
        copy_matrix(layer_bias[i], layer_output[i]);

        for (int j=0; j < len_layer; j++) {
            average_pooling_matrix(layer_input[j], layer_kernel[i][j], stride, layer_output[j]);
        }
    }
}


int main() {
    Matrix* m = create_matrix(2, 2);
    uniformity_matrix(m, 1);
    print_matrix(m);
    free_matrix(m);
    return 0;
}
