#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "struct/neuron.h"

#ifndef DEF_NEURAL_NETWORK_H
#define DEF_NEURAL_NETWORK_H

float max(float a, float b);
float sigmoid(float x);
float sigmoid_derivee(float x);
float leaky_ReLU(float x);
float leaky_ReLU_derivee(float x);
void creation_du_reseau_neuronal(Reseau* reseau_neuronal, int* neurones_par_couche, int nb_couches);
void suppression_du_reseau_neuronal(Reseau* reseau_neuronal);
void forward_propagation(Reseau* reseau_neuronal);
int* creation_de_la_sortie_voulue(Reseau* reseau_neuronal, int pos_nombre_voulu);
void backward_propagation(Reseau* reseau_neuronal, int* sortie_voulue);
void modification_du_reseau_neuronal(Reseau* reseau_neuronal, uint32_t nb_modifs);
void initialisation_du_reseau_neuronal(Reseau* reseau_neuronal);
float erreur_sortie(Reseau* reseau, int numero_voulu);

#endif
