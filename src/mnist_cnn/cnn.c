#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "cnn.h"

#define PADING_INPUT 2 // Augmente les dimensions de l'image d'entrée
#define RAND_FLT() ((float)rand())/((float)RAND_MAX) // Génère un flotant entre 0 et 1

// Les dérivées sont l'opposé
#define TANH 1
#define SIGMOID 2
#define RELU 3
#define SOFTMAX 4

#define ZERO 0
#define GLOROT_NORMAL 1
#define GLOROT_UNIFROM 2
#define HE_NORMAL 3
#define HE_UNIFORM 4

// Penser à mettre srand(time(NULL)); (pour les proba de dropout)


float max(float a, float b) {
    return a<b?b:a;
}

float sigmoid(float x) {
    return 1/(1 + exp(-x));
}

float sigmoid_derivative(float x) {
    float tmp = exp(-x);
    return tmp/((1+tmp)*(1+tmp));
}

float relu(float x) {
    return max(0, x);
}

float relu_derivative(float x) {
    if (x > 0)
        return 1;
    return 0;
}

float tanh_(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    float a = tanh(x);
    return 1 - a*a;
}

void apply_softmax_input(float ***input, int depth, int rows, int columns) {
    int i, j, k;
    float m = FLT_MIN;
    float sum=0;
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                m = max(m, input[i][j][k]);
            }
        }
    }
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                input[i][j][k] = exp(m-input[i][j][k]);
                sum += input[i][j][k];
            }
        }
    }
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                input[i][j][k] = input[i][j][k]/sum;
            }
        }
    }
}

void apply_function_input(float (*f)(float), float*** input, int depth, int rows, int columns) {
    int i, j ,k;
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                input[i][j][k] = (*f)(input[i][j][k]);
            }
        }
    }
}

void choose_apply_function_input(int activation, float*** input, int depth, int rows, int columns) {
    if (activation == RELU) {
        apply_function_input(relu, input, depth, rows, columns);
    }
    else if (activation == SIGMOID) {
        apply_function_input(sigmoid, input, depth, rows, columns);
    }
    else if (activation == SOFTMAX) {
        apply_softmax_input(input, depth, rows, columns);
    }
    else if (activation == TANH) {
        apply_function_input(tanh_, input, depth, rows, columns);
    }
    else {
        printf("Erreur, fonction d'activation inconnue");
    }
}

int will_be_drop(int dropout_prob) {
    /* Renvoie si oui ou non le neurone va être abandonné */
    return (rand() % 100)<dropout_prob;
}

Network* create_network(int max_size, int dropout, int initialisation, int input_dim, int input_depth) {
    /* Créé un réseau qui peut contenir max_size couche (dont celle d'input et d'output)  */
    if (dropout<0 || dropout>100) {
        printf("Erreur, la probabilité de dropout n'est pas respecté, elle doit être comprise entre 0 et 100\n");
    }
    Network* network = malloc(sizeof(Network));
    network->max_size = max_size;
    network->dropout = dropout;
    network->initialisation = initialisation;
    network->size = 1;
    network->input = malloc(sizeof(float***)*max_size);
    network->kernel = malloc(sizeof(Kernel)*(max_size-1));
    create_a_cube_input_layer(network, 0, input_depth, input_dim);
    int i, j;
    network->dim = malloc(sizeof(int*)*max_size);
    for (i=0; i<max_size; i++) {
        network->dim[i] = malloc(sizeof(int)*2);
    }
    network->dim[0][0] = input_dim;
    network->dim[0][1] = input_depth;
    return network;
}

Network* create_network_lenet5(int dropout, int activation, int initialisation) {
    /* Renvoie un réseau suivant l'architecture LeNet5 */
    Network* network;
    network = create_network(8, dropout, initialisation, 32, 1);
    add_convolution(network, 6, 5, activation);
    add_average_pooling(network, 2, activation);
    add_convolution(network, 16, 5, activation);
    add_average_pooling_flatten(network, 2, activation);
    add_dense(network, 120, 84, activation);
    add_dense(network, 84, 10, activation);
    add_dense(network, 10, 10, SOFTMAX);
    return network;
}

void create_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    /* Créé et alloue de la mémoire à une couche de type input cube */
    int i, j;
    network->input[pos] = malloc(sizeof(float**)*depth);
    for (i=0; i<depth; i++) {
        network->input[pos][i] = malloc(sizeof(float*)*dim);
        for (j=0; j<dim; j++) {
            network->input[pos][i][j] = malloc(sizeof(float)*dim);
        }
    }
    network->dim[pos][0] = dim;
    network->dim[pos][1] = depth;
}

void create_a_line_input_layer(Network* network, int pos, int dim) {
    /* Créé et alloue de la mémoire à une couche de type ligne */
    int i;
    network->input[pos] = malloc(sizeof(float**));
    network->input[pos][0] = malloc(sizeof(float*));
    network->input[pos][0][0] = malloc(sizeof(float)*dim);
}

void initialisation_1d_matrix(int initialisation, float* matrix, int rows, int n) { //NOT FINISHED
    /* Initialise une matrice 1d rows de float en fonction du type d'initialisation */
    int i;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        matrix[i] = lower_bound + RAND_FLT()*distance;
    }
}
void initialisation_2d_matrix(int initialisation, float** matrix, int rows, int columns, int n) { //NOT FINISHED
    /* Initialise une matrice 2d rows*columns de float en fonction du type d'initialisation */
    int i, j;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            matrix[i][j] = lower_bound + RAND_FLT()*distance;
        }
    }
}

void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int rows, int columns, int n) { //NOT FINISHED
    /* Initialise une matrice 3d depth*dim*columns de float en fonction du type d'initialisation */
    int i, j, k;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                matrix[i][j][k] = lower_bound + RAND_FLT()*distance;
            }
        }
    }
}

void initialisation_4d_matrix(int initialisation, float**** matrix, int rows, int columns, int rows1, int columns1, int n) { //NOT FINISHED
    /* Initialise une matrice 4d rows*columns*rows1*columns1 de float en fonction du type d'initialisation */
    int i, j, k, l;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            for (k=0; k<rows1; k++) {
                for (l=0; l<columns1; l++) {
                    matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance;
                }
            }
        }
    }
}

void add_average_pooling(Network* network, int kernel_size, int activation) {
    /* Ajoute au réseau une couche d'average pooling valide de dimension dim*dim */
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = NULL;
    network->kernel[n].activation = activation + 100*kernel_size;
    create_a_cube_input_layer(network, n, network->dim[n-1][1], network->dim[n-1][0]/2);
    network->size++;
}

void add_average_pooling_flatten(Network* network, int kernel_size, int activation) {
    /* Ajoute au réseau une couche d'average pooling valide de dimension dim*dim qui aplatit */
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche d'average pooling, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = NULL;
    network->kernel[n].activation = activation + 100*kernel_size;
    int dim = (network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1])/(kernel_size*kernel_size);
    create_a_line_input_layer(network, n, dim);
    network->size++;
}

void add_convolution(Network* network, int nb_filter, int kernel_size, int activation) {
    /* Ajoute une couche de convolution dim*dim au réseau et initialise les kernels */
    int n = network->size, i, j, k;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche de convolution, le réseau est déjà plein\n");
        return;
    }
    int r = network->dim[n-1][1];
    int c = nb_filter;
    network->kernel[n].nn = NULL;
    network->kernel[n].cnn = malloc(sizeof(Kernel_cnn));
    network->kernel[n].activation = activation;
    network->kernel[n].cnn->k_size = kernel_size;
    network->kernel[n].cnn->rows = r;
    network->kernel[n].cnn->columns = c;
    network->kernel[n].cnn->w = malloc(sizeof(float***)*r);
    network->kernel[n].cnn->d_w = malloc(sizeof(float***)*r);
    for (i=0; i<r; i++) {
        network->kernel[n].cnn->w[i] = malloc(sizeof(float**)*c);
        network->kernel[n].cnn->d_w[i] = malloc(sizeof(float**)*c);
        for (j=0; j<c; j++) {
            network->kernel[n].cnn->w[i][j] = malloc(sizeof(float*)*kernel_size);
            network->kernel[n].cnn->d_w[i][j] = malloc(sizeof(float*)*kernel_size);
            for (k=0; k<kernel_size; k++) {
                network->kernel[n].cnn->w[i][j][k] = malloc(sizeof(float)*kernel_size);
                network->kernel[n].cnn->d_w[i][j][k] = malloc(sizeof(float)*kernel_size);
            }
        }
    }
    network->kernel[n].cnn->bias = malloc(sizeof(float**)*c);
    network->kernel[n].cnn->d_bias = malloc(sizeof(float**)*c);
    for (i=0; i<c; i++) {
        network->kernel[n].cnn->bias[i] = malloc(sizeof(float*)*kernel_size);
        network->kernel[n].cnn->d_bias[i] = malloc(sizeof(float*)*kernel_size);
        for (j=0; j<kernel_size; j++) {
            network->kernel[n].cnn->bias[i][j] = malloc(sizeof(float)*kernel_size);
            network->kernel[n].cnn->d_bias[i][j] = malloc(sizeof(float)*kernel_size);
        }
    }
    create_a_cube_input_layer(network, n, c, network->dim[n-1][0] - 2*(kernel_size/2));
    int n_int = network->dim[n-1][0]*network->dim[n-1][0]*network->dim[n-1][1];
    int n_out = network->dim[n][0]*network->dim[n][0]*network->dim[n][1];
    initialisation_3d_matrix(network->initialisation, network->kernel[n].cnn->bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_3d_matrix(ZERO, network->kernel[n].cnn->d_bias, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(network->initialisation, network->kernel[n].cnn->w, r, c, kernel_size, kernel_size, n_int+n_out);
    initialisation_4d_matrix(ZERO, network->kernel[n].cnn->d_w, r, c, kernel_size, kernel_size, n_int+n_out);
    network->size++;
}

void add_dense(Network* network, int input_units, int output_units, int activation) {
    /* Ajoute une couche dense au réseau et initialise les poids et les biais*/
    int n = network->size;
    if (network->max_size == n) {
        printf("Impossible de rajouter une couche dense, le réseau est déjà plein\n");
        return;
    }
    network->kernel[n].cnn = NULL;
    network->kernel[n].nn = malloc(sizeof(Kernel_nn));
    network->kernel[n].activation = activation;
    network->kernel[n].nn->input_units = input_units;
    network->kernel[n].nn->output_units = output_units;
    network->kernel[n].nn->bias = malloc(sizeof(float)*output_units);
    network->kernel[n].nn->d_bias = malloc(sizeof(float)*output_units);
    network->kernel[n].nn->weights = malloc(sizeof(float*)*input_units);
    network->kernel[n].nn->d_weights = malloc(sizeof(float*)*input_units);
    for (int i=0; i<input_units; i++) {
        network->kernel[n].nn->weights[i] = malloc(sizeof(float)*output_units);
        network->kernel[n].nn->d_weights[i] = malloc(sizeof(float)*output_units);
    }
    initialisation_1d_matrix(network->initialisation, network->kernel[n].nn->bias, output_units, output_units+input_units);
    initialisation_1d_matrix(ZERO, network->kernel[n].nn->d_bias, output_units, output_units+input_units);
    initialisation_2d_matrix(network->initialisation, network->kernel[n].nn->weights, input_units, output_units, output_units+input_units);
    initialisation_2d_matrix(ZERO, network->kernel[n].nn->d_weights, input_units, output_units, output_units+input_units);
    create_a_line_input_layer(network, n, output_units);
    network->size++;
}

void write_image_in_newtork_32(int** image, int height, int width, float** input) {
    /* Ecrit une image 28*28 au centre d'un tableau 32*32 et met à 0 le reste */

    for (int i=0; i < height+2*PADING_INPUT; i++) {
        for (int j=PADING_INPUT; j < width+2*PADING_INPUT; j++) {
            if (i<PADING_INPUT || i>height+PADING_INPUT || j<PADING_INPUT || j>width+PADING_INPUT) {
                input[i][j] = 0.;
            }
            else {
                input[i][j] = (float)image[i][j] / 255.0f;
            }
        }
    }
}

void make_convolution(float*** input, Kernel_cnn* kernel, float*** output, int output_dim) {
    /* Effectue une convolution sans stride */
    //NOT FINISHED, MISS CONDITIONS ON THE CONVOLUTION
    float f;
    int i, j, k, a, b, c, n=kernel->k_size;
    for (i=0; i<kernel->columns; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                f = kernel->bias[i][j][k];
                for (a=0; a<kernel->rows; a++) {
                    for (b=0; b<n; b++) {
                        for (c=0; c<n; c++) {
                            f += kernel->w[a][i][b][c]*input[a][j+a][k+b];
                        }
                    }
                }
                output[i][j][k] = f;
            }
        }
    }
}

void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim) {
    /* Effecute un average pooling avec stride=size */
    //NOT FINISHED, MISS CONDITIONS ON THE POOLING
    float average;
    int i, j, k, a, b, n=size*size;
    for (i=0; i<output_depth; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                average = 0.;
                for (a=0; a<size; a++) {
                    for (b=0; b<size; b++) {
                        average += input[i][2*j +a][2*k +b];
                    }
                }
                output[i][j][k] = average;
            }
        }
    }
}

void make_average_pooling_flattened(float*** input, float* output, int size, int input_depth, int input_dim) {
    /* Effectue un average pooling avec stride=size et aplatissement */
    if ((input_depth*input_dim*input_dim) % (size*size) != 0) {
        printf("Erreur, deux layers non compatibles avec un average pooling flattened");
        return;
    }
    float average;
    int i, j, k, a, b, n=size*size, cpt=0;
    int output_dim = input_dim - 2*(size/2);
    for (i=0; i<input_depth; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                average = 0.;
                for (a=0; a<size; a++) {
                    for (b=0; b<size; b++) {
                        average += input[i][2*j +a][2*k +b];
                    }
                }
                output[cpt] = average;
                cpt++;
            }
        }
    }
}

void make_fully_connected(float* input, Kernel_nn* kernel, float* output, int size_input, int size_output) {
    /* Effecute une full connection */
    int i, j, k;
    float f;
    for (i=0; i<size_output; i++) {
        f = kernel->bias[i];
        for (j=0; j<size_input; j++) {
            f += kernel->weights[i][j]*input[j];
        }
        output[i] = f;
    }
}

void free_a_cube_input_layer(Network* network, int pos, int depth, int dim) {
    /* Libère la mémoire allouée à une couche de type input cube */
    int i, j, k;
    for (i=0; i<depth; i++) {
        for (j=0; j<dim; j++) {
            free(network->input[pos][i][j]);
        }
        free(network->input[pos][i]);
    }
    free(network->input[pos]);
}

void free_a_line_input_layer(Network* network, int pos) {
    /* Libère la mémoire allouée à une couche de type input line */
    free(network->input[pos][0][0]);
    free(network->input[pos][0]);
    free(network->input[pos]);
}

void free_average_pooling(Network* network, int pos) {
    /* Libère l'espace mémoie et supprime une couche d'average pooling classique */
    free_a_cube_input_layer(network, pos, network->dim[pos-1][1], network->dim[pos-1][0]/2);
}

void free_average_pooling_flatten(Network* network, int pos) {
    /* Libère l'espace mémoie et supprime une couche d'average pooling flatten */
    free_a_line_input_layer(network, pos);
}

void free_convolution(Network* network, int pos) {
    /* Libère l'espace mémoire et supprime une couche de convolution */
    int i, j, k, c = network->kernel[pos].cnn->columns;
    int k_size = network->kernel[pos].cnn->k_size;
    int r = network->kernel[pos].cnn->rows;
    free_a_cube_input_layer(network, pos, c, network->dim[pos-1][0] - 2*(k_size/2));
    for (i=0; i<c; i++) {
        for (j=0; j<k_size; j++) {
            free(network->kernel[pos].cnn->bias[i][j]);
            free(network->kernel[pos].cnn->d_bias[i][j]);
        }
        free(network->kernel[pos].cnn->bias[i]);
        free(network->kernel[pos].cnn->d_bias[i]);
    }
    free(network->kernel[pos].cnn->bias);
    free(network->kernel[pos].cnn->d_bias);

    for (i=0; i<r; i++) {
        for (j=0; j<c; j++) {
            for (k=0; k<k_size; k++) {
                free(network->kernel[pos].cnn->w[i][j][k]);
                free(network->kernel[pos].cnn->d_w[i][j][k]);
            }
            free(network->kernel[pos].cnn->w[i][j]);
            free(network->kernel[pos].cnn->d_w[i][j]);
        }
        free(network->kernel[pos].cnn->w[i]);
        free(network->kernel[pos].cnn->d_w[i]);
    }
    free(network->kernel[pos].cnn->w);
    free(network->kernel[pos].cnn->d_w);

    free(network->kernel[pos].cnn);
}

void free_dense(Network* network, int pos) {
    /* Libère l'espace mémoire et supprime une couche dense */
    free_a_line_input_layer(network, pos);
    int i, dim = network->kernel[pos].nn->output_units;
    for (int i=0; i<dim; i++) {
        free(network->kernel[pos].nn->weights[i]);
        free(network->kernel[pos].nn->d_weights[i]);
    }
    free(network->kernel[pos].nn->weights);
    free(network->kernel[pos].nn->d_weights);

    free(network->kernel[pos].nn->bias);
    free(network->kernel[pos].nn->d_bias);

    free(network->kernel[pos].nn);
}

void free_network_creation(Network* network) {
    /* Libère l'espace alloué dans la fonction 'create_network' */
    free_a_cube_input_layer(network, 0, network->dim[0][1], network->dim[0][0]);

    for (int i=0; i<network->max_size; i++) {
        free(network->dim[i]);
    }
    free(network->dim);

    free(network->kernel);
    free(network->input);

    free(network);
}

void free_network_lenet5(Network* network) {
    /* Libère l'espace alloué dans la fonction 'create_network_lenet5' */
    free_dense(network, 6);
    free_dense(network, 5);
    free_dense(network, 4);
    free_average_pooling_flatten(network, 3);
    free_convolution(network, 2);
    free_average_pooling(network, 1);
    free_convolution(network, 0);
    free_network_creation(network);
    if (network->size != network->max_size) {
        printf("Attention, le réseau LeNet5 n'est pas complet");
    }
}

void forward_propagation(Network* network) {
    /* Propage en avant le cnn */
    for (int i=0; i < network->size-1; i++) {
        if (network->kernel[i].nn==NULL && network->kernel[i].cnn!=NULL) {
            make_convolution(network->input[i], network->kernel[i].cnn, network->input[i+1], network->dim[i+1][0]);
            choose_apply_function_input(network->kernel[i].activation, network->input[i+1], network->dim[i+1][1], network->dim[i+1][0], network->dim[i+1][0]);
        }
        else if (network->kernel[i].nn!=NULL && network->kernel[i].cnn==NULL) {
            make_fully_connected(network->input[i][0][0], network->kernel[i].nn, network->input[i+1][0][0], network->dim[i][0], network->dim[i+1][0]);
            choose_apply_function_input(network->kernel[i].activation, network->input[i+1], 1, 1, network->dim[i+1][0]);
        }
        else {
            if (network->size-2==i) {
                printf("Le réseau ne peut pas finir par une pooling layer");
                return;
            }
            if (network->kernel[i+1].nn!=NULL && network->kernel[i+1].cnn==NULL) {
                make_average_pooling_flattened(network->input[i], network->input[i+1][0][0], network->kernel[i].activation/100, network->dim[i][1], network->dim[i][0]);
                choose_apply_function_input(network->kernel[i].activation%100, network->input[i+1], 1, 1, network->dim[i+1][0]);
            }
            else if (network->kernel[i+1].nn==NULL && network->kernel[i+1].cnn!=NULL) {
                make_average_pooling(network->input[i], network->input[i+1], network->kernel[i].activation/100, network->dim[i+1][1], network->dim[i+1][0]);
                choose_apply_function_input(network->kernel[i].activation%100, network->input[i+1], network->dim[i+1][1], network->dim[i+1][0], network->dim[i+1][0]);
            }
            else {
                printf("Le réseau ne peut pas contenir deux poolings layers collées");
                return;
            }
        }
    }
}

void backward_propagation(Network* network, float wanted_number) {
    /* Propage en arrière le cnn */
    float* wanted_output = generate_wanted_output(wanted_number);
    int n = network->size-1;
    float loss = compute_cross_entropy_loss(network->input[n][0][0], wanted_output, network->dim[n][0]);
    int i, j;
    for (i=n; i>=0; i--) {
        if (i==n) {
            if (network->kernel[i].activation == SOFTMAX) {
                int l2 = network->dim[i][0]; // Taille de la dernière couche
                int l1 = network->dim[i-1][0];
                for (j=0; j<l2; j++) {

                }
            }
            else {
                printf("Erreur, seule la fonction softmax est implémentée pour la dernière couche");
                return;
            }
        }
        else {
            if (network->kernel[i].activation == SIGMOID) {

            }
            else if (network->kernel[i].activation == TANH) {

            }
            else if (network->kernel[i].activation == RELU) {
                
            }
        }
    }
    free(wanted_output);
}

float compute_cross_entropy_loss(float* output, float* wanted_output, int len) {
    /* Renvoie l'erreur du réseau neuronal pour une sortie */
    float loss=0.;
    for (int i=0; i<len ; i++) {
        if (wanted_output[i]==1) {
            if (output[i]==0.) {
                loss -= log(FLT_EPSILON);
            }
            else {
                loss -= log(output[i]);
            }
        }
    }
    return loss;
}
                
float* generate_wanted_output(float wanted_number) {
    /* On considère que la sortie voulue comporte 10 éléments */
    float* wanted_output = malloc(sizeof(float)*10);
    for (int i=0; i<10; i++) {
        if (i==wanted_number) {
            wanted_output[i]=1;
        }
        else {
            wanted_output[i]=0;
        }
    }
    return wanted_output;
}
