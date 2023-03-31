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


typedef float (*funcPtr)(float);

//* Identité
#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float identity(float x);

#ifdef __CUDACC__
extern "C"
__host__ __device__ 
#endif
float identity_derivative(float x);

//* Sigmoid
#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float sigmoid(float x);

#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float sigmoid_derivative(float x);

//* RELU
#ifdef __CUDACC__
extern "C"
__host__ __device__ 
#endif
float relu(float x);

#ifdef __CUDACC__
extern "C"
__host__ __device__ 
#endif
float relu_derivative(float x);

//* Leaky RELU
#ifdef __CUDACC__
extern "C"
__host__ __device__ 
#endif
float leaky_relu(float x);

#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float leaky_relu_derivative(float x);

//* Tanh
#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float tanh_(float x);

#ifdef __CUDACC__
extern "C"
__host__ __device__
#endif
float tanh_derivative(float x);


#ifdef __CUDACC__
extern "C"
#endif
/*
* Applique softmax sur input[depth][rows][columns]
*/
void apply_softmax_input(float ***input, int depth, int rows, int columns);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Applique la fonction f sur input[depth][rows][columns]
*/
void apply_function_input(int activation, float*** input, int depth, int rows, int columns);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Applique une fonction d'activation (repérée par son identifiant) à une matrice
*/
void apply_function_to_matrix(int activation, float*** input, int depth, int dim);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Applique une fonction d'activation (repérée par son identifiant) à un vecteur
*/
void apply_function_to_vector(int activation, float*** input, int dim);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Renvoie la fonction d'activation correspondant à son identifiant (activation)
*/
funcPtr get_activation_function(int activation);

/*
* Récupère un pointeur sur le device vers la fonction d'activation demandée puis le transforme en pointeur sur l'host
*/
#ifdef __CUDACC__
extern "C"
funcPtr get_activation_function_cuda(int activation);
#endif

#endif