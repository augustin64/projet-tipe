#ifndef DEF_FUNCTION_H
#define DEF_FUNCTION_H


// Les dérivées sont l'opposé
#define IDENTITY 1
#define TANH 2
#define SIGMOID 3
#define RELU 4
#define SOFTMAX 5
#define LEAKY_RELU 6

#define LEAKER 0.2

// RELU and Leaky RELU max value
#define RELU_CLIP_VALUE 15


typedef float (*ptr)(float);
typedef ptr (*pm)();

float identity(float x);

float identity_derivative(float x);

float sigmoid(float x);

float sigmoid_derivative(float x);

float relu(float x);

float relu_derivative(float x);

float leaky_relu(float x);

float leaky_relu_derivative(float x);

float tanh_(float x);

float tanh_derivative(float x);

/*
* Applique softmax sur input[depth][rows][columns]
*/
void apply_softmax_input(float ***input, int depth, int rows, int columns);

/*
* Applique la fonction f sur input[depth][rows][columns]
*/
void apply_function_input(float (*f)(float), float*** input, int depth, int rows, int columns);

/*
* Applique une fonction d'activation (repérée par son identifiant) à une matrice
*/
void apply_function_to_matrix(int activation, float*** input, int depth, int dim);

/*
* Applique une fonction d'activation (repérée par son identifiant) à un vecteur
*/
void apply_function_to_vector(int activation, float*** input, int dim);

/*
* Renvoie la fonction d'activation correspondant à son identifiant (activation)
*/
ptr get_activation_function(int activation);

#endif