#include "struct.h"

#ifndef DEF_UPDATE_H
#define DEF_UPDATE_H

/*
* Met à jours les poids à partir de données obtenus après plusieurs backpropagations
* Puis met à 0 tous les d_weights
*/
void update_weights(Network* network, Network* d_network, int nb_images);

/*
* Met à jours les biais à partir de données obtenus après plusieurs backpropagations
* Puis met à 0 tous les d_bias
*/
void update_bias(Network* network, Network* d_network, int nb_images);

/*
* Met à 0 toutes les données de backpropagation de poids
*/
void reset_d_weights(Network* network);

/*
* Met à 0 toutes les données de backpropagation de biais
*/
void reset_d_bias(Network* network);

#endif