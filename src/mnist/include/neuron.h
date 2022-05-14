#ifndef DEF_NEURON_H
#define DEF_NEURON_H

typedef struct Neuron {
    float* weights; // Liste de tous les poids des arêtes sortants du neurone
    float bias; // Caractérise le bias du neurone
    float z; // Sauvegarde des calculs faits sur le neurone (programmation dynamique)

    float *back_weights; // Changement des poids sortants lors de la backpropagation
    float *last_back_weights; // Dernier changement de d_poid_sortants
    float back_bias; // Changement du bias lors de la backpropagation
    float last_back_bias; // Dernier changement de back_bias
} Neuron;


typedef struct Layer {
    int nb_neurons; // Nombre de neurones dans la couche (longueur du tableau ci-dessous)
    Neuron** neurons; // Tableau des neurones dans la couche
} Layer;

typedef struct Network {
    int nb_layers; // Nombre de couches dans le réseau neuronal (longueur du tableau ci-dessous)
    Layer** layers; // Tableau des couches dans le réseau neuronal
} Network;

#endif