#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "struct/neuron.h"

#ifndef DEF_NEURAL_NETWORK_H
#define DEF_NEURAL_NETWORK_H

void creation_du_reseau_neuronal(Reseau* reseau_neuronal, int* neurones_par_couche, int nb_couches);
void suppression_du_reseau_neuronal(Reseau* reseau_neuronal);
void forward_propagation(Reseau* reseau_neuronal);
int* creation_de_la_sortie_voulue(Reseau* reseau_neuronal, int pos_nombre_voulu);
void backward_propagation(Reseau* reseau_neuronal, int* sortie_voulue);
void modification_du_reseau_neuronal(Reseau* reseau_neuronal);
void initialisation_du_reseau_neuronal(Reseau* reseau_neuronal);


#endif
