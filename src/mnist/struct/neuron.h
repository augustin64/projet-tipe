#ifndef DEF_NEURON_H
#define DEF_NEURON_H

typedef struct Neurone{
    float* poids_sortants; // Liste de tous les poids des arêtes sortants du neurone
    float biais; // Caractérise le biais du neurone
    float z; // Sauvegarde des calculs faits sur le neurone (programmation dynamique)

    float *d_poids_sortants; // Changement des poids sortants lors de la backpropagation
    float *last_d_poids_sortants; // Dernier changement de d_poid_sortants
    float d_biais; // Changement du biais lors de la backpropagation
    float last_d_biais; // Dernier changement de d_biais
} Neurone;


typedef struct Couche{
    int nb_neurones; // Nombre de neurones dans la couche (longueur du tableau ci-dessous)
    Neurone** neurones; // Tableau des neurones dans la couche
} Couche;

typedef struct Reseau{
    int nb_couches; // Nombre de couches dans le réseau neuronal (longueur du tableau ci-dessous)
    Couche** couches; // Tableau des couches dans le réseau neuronal
} Reseau;

#endif