#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifndef DEF_CNN_H
#define DEF_CNN_H


typedef struct Kernel_cnn {
    int k_size;
    int rows;
    int columns;
    int b;
    float*** bias; // De dimension columns*k_size*k_size
    float*** d_bias; // De dimension columns*k_size*k_size
    float**** w; // De dimension rows*columns*k_size*k_size
    float**** d_w; // De dimension rows*columns*k_size*k_size
} Kernel_cnn;

typedef struct Kernel_nn {
    int input_units;
    int output_units;
    float* bias; // De dimension output_units
    float* d_bias; // De dimension output_units
    float** weights; // De dimension input_units*output_units
    float** d_weights; // De dimension input_units*output_units
} Kernel_nn;

typedef struct Kernel {
    Kernel_cnn* cnn;
    Kernel_nn* nn;
    int activation; // Vaut l'activation sauf pour un pooling où il: vaut kernel_size*100 + activation
} Kernel;

typedef struct Layer {

} Layer;

typedef struct Network{
    int dropout; // Contient la probabilité d'abandon entre 0 et 100 (inclus)
    int initialisation; // Contient le type d'initialisation
    int max_size; // Taille maximale du réseau après initialisation
    int size; // Taille actuelle du réseau
    int** dim; // Contient les dimensions de l'input (width*depth)
    Kernel* kernel;
    float**** input;
} Network;

float max(float a, float b);
float sigmoid(float x);
float sigmoid_derivative(float x);
float relu(float x);
float relu_derivative(float x);
float tanh_(float x);
float tanh_derivative(float x);
void apply_softmax_input(float ***input, int depth, int rows, int columns);
void apply_function_input(float (*f)(float), float*** input, int depth, int rows, int columns);
void choose_apply_function_input(int activation, float*** input, int depth, int rows, int columns);
int will_be_drop(int dropout_prob);
Network* create_network(int max_size, int dropout, int initialisation, int input_dim, int input_depth);
Network* create_network_lenet5(int dropout, int activation, int initialisation);
void create_a_cube_input_layer(Network* network, int pos, int depth, int dim);
void create_a_line_input_layer(Network* network, int pos, int dim);
void initialisation_1d_matrix(int initialisation, float* matrix, int rows, int n); //NOT FINISHED (UNIFORM AND VARIATIONS)
void initialisation_2d_matrix(int initialisation, float** matrix, int rows, int columns, int n); //NOT FINISHED
void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int rows, int columns, int n); //NOT FINISHED
void initialisation_4d_matrix(int initialisation, float**** matrix, int rows, int columns, int rows1, int columns1, int n); //NOT FINISHED
void add_average_pooling(Network* network, int kernel_size, int activation);
void add_average_pooling_flatten(Network* network, int kernel_size, int activation);
void add_convolution(Network* network, int nb_filter, int kernel_size, int activation);
void add_dense(Network* network, int input_units, int output_units, int activation);
void write_image_in_newtork_32(int** image, int height, int width, float** input);
void make_convolution(float*** input, Kernel_cnn* kernel, float*** output, int output_dim);
void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim);
void make_average_pooling_flattened(float*** input, float* output, int size, int input_depth, int input_dim);
void make_fully_connected(float* input, Kernel_nn* kernel, float* output, int size_input, int size_output);
void free_a_cube_input_layer(Network* network, int pos, int depth, int dim);
void free_a_line_input_layer(Network* network, int pos);
void free_average_pooling(Network* network, int pos);
void free_average_pooling_flatten(Network* network, int pos);
void free_convolution(Network* network, int pos);
void free_dense(Network* network, int pos);
void free_network_creation(Network* network);
void free_network_lenet5(Network* network);
float compute_cross_entropy_loss(float* output, float* wanted_output, int len);
void forward_propagation(Network* network);
void backward_propagation(Network* network, float wanted_number); //NOT FINISHED
float compute_cross_entropy_loss(float* output, float* wanted_output, int len);
float* generate_wanted_output(float wanted_number);


#endif