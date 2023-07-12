#include <stdlib.h>
#include <stdio.h>

#include "struct.h"

#ifndef DEF_MODELS_H
#define DEF_MODELS_H
/*
* Renvoie un réseau suivant l'architecture LeNet5
*/
Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth, int finetuning);

/*
* Renvoie un réseau suivant l'architecture AlexNet
* C'est à dire en entrée 3x227x227 et une sortie de taille 'size_output'
*/
Network* create_network_alexnet(float learning_rate, int dropout, int activation, int initialisation, int size_output, int finetuning);

/*
* Renvoie un réseau suivant l'architecture VGG16 modifiée pour prendre en entrée 3x256x256
* et une sortie de taille 'size_output'
*/
Network* create_network_VGG16(float learning_rate, int dropout, int activation, int initialisation, int size_output, int finetuning);


/*
* Renvoie un réseau suivant l'architecture VGG16 originel pour prendre en entrée 3x227x227
* et une sortie de taille 1 000
*/
Network* create_network_VGG16_227(float learning_rate, int dropout, int activation, int initialisation, int finetuning);

/*
* Renvoie un réseau sans convolution, similaire à celui utilisé dans src/dense
*/
Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth, int finetuning);
#endif