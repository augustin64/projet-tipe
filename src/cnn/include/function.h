#ifndef DEF_FUNCTION_H
#define DEF_FUNCTION_H


// Les dérivées sont l'opposé
#define IDENTITY 1
#define TANH 2
#define SIGMOID 3
#define RELU 4
#define SOFTMAX 5


typedef float (*ptr)(float);
typedef ptr (*pm)();

/*
* Fonction max pour les floats
*/
float max_float(float a, float b);

float identity(float x);

float identity_derivative(float x);

float sigmoid(float x);

float sigmoid_derivative(float x);

float relu(float x);

float relu_derivative(float x);

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
* Redirige vers la fonction à appliquer sur une matrice
*/
void choose_apply_function_matrix(int activation, float*** input, int depth, int dim);

/*
* Redirige vers la fonction à appliquer sur un vecteur
*/
void choose_apply_function_vector(int activation, float*** input, int dim);

/*
* Renvoie la fonction d'activation correspondant à son identifiant (activation)
*/
ptr get_function_activation(int activation);

#endif