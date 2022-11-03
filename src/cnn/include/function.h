#ifndef DEF_FUNCTION_H
#define DEF_FUNCTION_H


// Les dérivées sont l'opposé
#define TANH 1
#define SIGMOID 2
#define RELU 3
#define SOFTMAX 4

typedef float (*ptr)(float);
typedef ptr (*pm)();

/*
* Fonction max pour les floats
*/
float max(float a, float b);

float sigmoid(float x);

float sigmoid_derivative(float x);

float relu(float x);

float relu_derivative(float x);

float tanh_(float x);

float tanh_derivative(float x);

/*
* Applique softmax sur ????
*/
void apply_softmax_input(float ***input, int depth, int rows, int columns);

/*
* Applique la fonction f sur ????
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