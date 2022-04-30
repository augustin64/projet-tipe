#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>


typedef struct Matrix {
    int depths; // Nombre de couches de la matrice
    int rows; // Nombre de lignes de la matrice
    int columns; // Nombre de colonnes de la matrice
    float*** value; // Tableau 2d comportant les valeurs de matrice

} Matrix;

float exp_float(float a);
float max_float(float a, float b);
float min_float(float a, float b);
Matrix* create_matrix(int nb_layers, int nb_rows, int nb_columns);
void uniformise_matrix(Matrix* m, float x);
float max_in_matrix(Matrix* m);
void free_matrix(Matrix* m);
float number_from_matrix(Matrix* m);
void product_of_a_scalar_matrix(Matrix* m, float scalar);
void sum_of_a_scalar_matrix(Matrix* m, float scalar);
Matrix* copy_matrix(Matrix* m);
Matrix* apply_function_new_matrix(Matrix* m, float (*f)(float));
void apply_function_matrix(Matrix* m, float (*f)(float));
Matrix* add_matrix(Matrix* m1, Matrix* m2);
Matrix* product_matrix(Matrix* m1, Matrix* m2);
void max_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void min_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void average_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void valid_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void full_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out);
void softmax_matrix(Matrix* m);
float quadratic_cost_matrix(Matrix* m, int i_number, int j_number, int k_number);
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


Matrix* create_matrix(int nb_layers, int nb_rows, int nb_columns) {
    /* Créé une matrice en lui allouant de la mémoire */
    Matrix* m = malloc(sizeof(Matrix));
    m->rows = nb_rows;
    m->columns = nb_columns;
    m->depths = nb_layers;
    m->value = malloc(sizeof(float**)*m->depths);
    for (int i=0; i < m->depths; i++) {
        m->value[i] = malloc(sizeof(float*)*m->rows);
        for (int j=0; j < m->rows; j++) {
            m->value[i][j] = malloc(sizeof(float*)*m->columns);
        }
    }
    return m;
}


void uniformise_matrix(Matrix* m, float x) {
    /* Donne la même valeur x à tous les éléments de la matrice */
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                m->value[i][j][k] = x;
            }
        }
    }
}


void print_matrix(Matrix* m) {
    /* Affiche la matrice */
    for (int i=0; i < m->depths; i++) {
        if (i!=0)
            printf("-----------------\n");
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                if (k!=0) 
                    printf(",");
                printf("%f ", m->value[i][j][k]);
            }
            printf("\n");
        }
    }
}


float max_in_matrix(Matrix* m) {
    /* Renvoie l'élément maximal de la matrice */
    float max_tmp = FLT_MIN;
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                max_tmp = max_float(max_tmp, m->value[i][j][k]);
            }
        }
    }
    return max_tmp;
}


void free_matrix(Matrix* m) {
    /* Libère l'espace mémoire alloué à la matrice */
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            free(m->value[i][j]);
        }
        free(m->value[i]);
    }
    free(m->value);
}


float number_from_matrix(Matrix* m) {
    /* Renvoie la somme des éléments de la matrice */
    float tmp=0;
    for (int i=0; i < m->depths ; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                tmp += m->value[i][j][k];
            }
        }
    }
    return tmp;
}


void product_of_a_scalar_matrix(Matrix* m, float scalar) {
    /* Multiplie la matrice par un scalaire */
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                m->value[i][j][k] *= scalar;
            }
        }
    }
}


void sum_of_a_scalar_matrix(Matrix* m, float scalar) {
    /* Ajoute un scalaire à la matrice */
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                m->value[i][j][k] += scalar;
            }
        }
    }
}


Matrix* copy_matrix(Matrix* m) {
    /* Renvoie une copie de la matrice */
    Matrix* new_m = create_matrix(m->depths, m->rows, m->columns);
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                new_m->value[i][j][k] = m->value[i][j][k];
            }
        }
    }
    return new_m;
}


Matrix* apply_function_new_matrix(Matrix* m, float (*f)(float)) {
    /* Renvoie une matrice avec une fonction appliquée
    à tous les  éléments de l'ancienne matrice */
    Matrix* new_m  = create_matrix(m->depths, m->rows, m->columns);
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m ->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                new_m->value[i][j][k] = (*f)(m->value[i][j][k]);
            }
        }
    }
    return new_m;
}


void apply_function_matrix(Matrix* m, float (*f)(float)) {
    /* Applique une fonction à tous les éléments de la matrice */
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m ->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                m->value[i][j][k] = (*f)(m->value[i][j][k]);
            }
        }
    }
}


Matrix* add_matrix(Matrix* m1, Matrix* m2) {
    /* Renvoie la somme de deux matrices */
    if (m1->depths != m2->depths || m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("Erreur, matrices non compatibles avec la somme");
        return NULL;
    }
    Matrix* m = create_matrix(m1->depths, m1->rows, m2->columns);
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                m->value[i][j][k] = m1->value[i][j][k] + m2->value[i][j][k];
            }
        }
    }
    return m;
}


/*Matrix* product_matrix(Matrix* m1, Matrix* m2) { // TO DO
    Renvoie une nouvelle matrice produit (classique)
    des deux matrices si les dimensions sont correctes
    if (m1->depths != m2->rows || m1->rows != ) {
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
}*/


void max_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de max pooling avec un décalage
    de (stride) éléments dans la matrice m_out */
    if (m_in->depths < kernel->depths || m_in->rows < kernel->rows || m_in->columns < kernel->columns) {
        printf("Erreur, kernel plus grand que la matrice dans max pooling");
        return;
    }
    if (((m_in->depths - kernel->depths)/stride)+1 != m_out->depths || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows || ((m_in->columns - kernel->columns)/stride)+1  != m_out->columns) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans max pooling");
        return;
    }
    int i, j, k, a, b, c;
    float tmp;
    for (i=0; i < m_out->depths; i++) {
        for (j=0; j < m_out->rows; j++) {
            for (k=0; k < m_out->columns; k++) {
                tmp = FLT_MIN;
                for (a=0; a < kernel->depths; a++) {
                    for (b=0; b < kernel->rows; b++) {
                        for (c=0; c < kernel->columns; c++) {
                            tmp = max_float(tmp, m_in->value[i*stride +a][j*stride +b][k*stride +c]);
                        }
                    }
                }
                m_out->value[i][j][k] =  tmp;
            }
        }
    }
}


void min_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de min pooling avec un décalage
    de (stride) éléments dans la matrice m_out */
    if (m_in->depths < kernel->depths || m_in->rows < kernel->rows || m_in->columns < kernel->columns) {
        printf("Erreur, kernel plus grand que la matrice dans min pooling");
        return;
    }
    if (((m_in->depths - kernel->depths)/stride)+1 != m_out->depths || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows || ((m_in->columns - kernel->columns)/stride)+1  != m_out->columns) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans min pooling");
        return;
    }
    int i, j, k, a, b, c;
    float tmp;
    for (i=0; i < m_out->depths; i++) {
        for (j=0; j < m_out->rows; j++) {
            for (k=0; k < m_out->columns; k++) {
                tmp = FLT_MAX;
                for (a=0; a < kernel->depths; a++) {
                    for (b=0; b < kernel->rows; b++) {
                        for (c=0; c < kernel->columns; c++) {
                            tmp = min_float(tmp, m_in->value[i*stride +a][j*stride +b][k*stride +c]);
                        }
                    }
                }
                m_out->value[i][j][k] =  tmp;
            }
        }
    }
}


void average_pooling_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère le résultat de average pooling avec un décalage
    de (stride) éléments dans la matrice m_out */
    if (m_in->depths < kernel->depths || m_in->rows < kernel->rows || m_in->columns < kernel->columns) {
        printf("Erreur, kernel plus grand que la matrice dans average pooling");
        return;
    }
    if (((m_in->depths - kernel->depths)/stride)+1 != m_out->depths || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows || ((m_in->columns - kernel->columns)/stride)+1  != m_out->columns) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans average pooling");
        return;
    }
    int i, j, k, a, b, c, nb=kernel->depths*kernel->rows*kernel->columns;
    float tmp;
    for (i=0; i < m_out->depths; i++) {
        for (j=0; j < m_out->rows; j++) {
            for (k=0; k < m_out->columns; k++) {
                tmp = 0;
                for (a=0; a < kernel->depths; a++) {
                    for (b=0; b < kernel->rows; b++) {
                        for (c=0; c < kernel->columns; c++) {
                            tmp += m_in->value[i*stride +a][j*stride +b][k*stride +c];
                        }
                    }
                }
                m_out->value[i][j][k] = tmp/nb;
            }
        }
    }
}


void valid_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère, la cross-correlation valide de m_in et
    kernel avec un décalage de stride, dans m_out */
    if (m_in->depths < kernel->depths || m_in->rows < kernel->rows || m_in->columns < kernel->columns) {
        printf("Erreur, kernel plus grand que la matrice dans valid cross-correlation");
        return;
    }
    if (((m_in->depths - kernel->depths)/stride)+1  != m_out->depths || ((m_in->rows - kernel->rows)/stride)+1 != m_out->rows || ((m_in->columns - kernel->columns)/stride)+1  != m_out->columns) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans valid cross-correlation");
        return;
    }
    int i, j, k, a, b, c, new_i, new_j, new_k;
    for (i=0; i < m_out->depths; i++) {
        for (j=0; j < m_out->rows; j++) {
            for (k=0; k < m_out->columns; k++) {
                m_out->value[i][j][k] = 0;
                for (a=0; a < kernel->depths; a++) {
                    for (b=0; b < kernel->rows; b++) {
                        for (c=0; c < kernel->columns; c++) {
                            m_out->value[i][j][k] += m_in->value[i*stride +a][j*stride +b][k*stride +c]*kernel->value[a][b][c];
                        }
                    }
                }
            }
        }
    }
}


void full_cross_correlation_matrix(Matrix* m_in, Matrix* kernel, int stride, Matrix* m_out) {
    /* Insère, la cross-correlation entière de m_in et
    kernel avec un décalage de stride, dans m_out */
    int rows_k = kernel->rows-1;
    int columns_k = kernel->columns-1;
    int depths_k = kernel->depths-1;
    if (m_in->depths < kernel->depths || m_in->rows < kernel->rows || m_in->columns < kernel->columns) {
        printf("Erreur, kernel plus grand que la matrice dans full cross-correlation");
        return;
    }
    if ((m_in->depths + 2*depths_k)/stride  != m_out->depths || (m_in->rows + 2*rows_k)/stride != m_out->rows || (m_in->columns + 2*columns_k)/stride  != m_out->columns) {
        printf("Erreur, matrice et kernel non compatibles avec le décalage ou la matrice sortante dans full cross-correlation");
        return;
    }
    int i, j, k, a, b, c, new_i, new_j, new_k;
    for (i=-depths_k; i < (m_out->depths + depths_k); i++) {
        for (j=-rows_k; j < (m_out->rows + rows_k); j++) {
            for (k=--columns_k; k < (m_out->columns + columns_k); k++) {
                m_out->value[i+rows_k][j+columns_k] = 0;
                for (a=0; a < kernel->depths; a++) {
                    for (b=0; b < kernel->rows; b++) {
                        for (c=0; c < kernel->columns; c++) {
                            new_i = i*stride +a;
                            new_j = j*stride +b;
                            new_k = k*stride +c;
                            if (new_k >= 0 || new_k < m_in->columns || new_i >= 0 || new_i < m_in->depths || new_j >= 0 || new_j < m_in->rows)
                                m_out->value[i+depths_k][j+rows_k][k+columns_k] += m_in->value[new_i][new_j][new_k]*kernel->value[a][b][c];
                        }
                    }
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


float quadratic_cost_matrix(Matrix* m, int i_number, int j_number, int k_number) {
    /* Renvoie l'erreur de la matrice où les valeurs sont des probabailités */
    float loss = 0;
    for (int i=0; i < m->depths; i++) {
        for (int j=0; j < m->rows; j++) {
            for (int k=0; k < m->columns; k++) {
                if (i==i_number && j==j_number && k==k_number)
                    loss += (1-m->value[i][j][k])*(1-m->value[i][j][k]);
                else
                    loss += m->value[i][j][k]*m->value[i][j][k];
            }
        }
    }
    return loss;
}


/*void rotation_180_matrix(Matrix* m) { // TO DO
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
}*/




int main() {
    Matrix* m = create_matrix(3, 3, 3);
    m->value[0][1][2]=10;
    softmax_matrix(m);
    print_matrix(m);
    free_matrix(m);
    return 1;
}