#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "../src/mnist/neuron_io.c"


Neurone* creer_neurone(int nb_sortants) {
    Neurone* neurone = malloc(2*sizeof(float*)+6*sizeof(float));
    neurone->poids_sortants = malloc(sizeof(float)*nb_sortants);
    neurone->dw = malloc(sizeof(float)*nb_sortants);

    for (int i=0; i < nb_sortants; i++) {
        neurone->poids_sortants[i] = 0.5;
        neurone->dw[i] = 0.0;
    }
    neurone->activation = 0.0;
    neurone->biais = 0.0;
    neurone->z = 0.0;
    neurone->dactivation = 0.0;
    neurone->dbiais = 0.0;
    neurone->dz = 0.0;

    return neurone;
}


Couche* creer_couche(int nb_neurones, int nb_sortants) {
    Couche* couche = malloc(sizeof(int)+sizeof(Neurone**));
    Neurone** tab = malloc(sizeof(Neurone*)*nb_neurones);

    couche->nb_neurones = nb_neurones;
    couche->neurones = tab;

    for (int i=0; i<nb_neurones; i++) {
        tab[i] = creer_neurone(nb_sortants);
    }
    return couche;
};


Reseau* creer_reseau(int nb_couches, int nb_max_neurones, int nb_min_neurones) {
    Reseau* reseau = malloc(sizeof(int)+sizeof(Couche**));
    reseau->couches = malloc(sizeof(Couche*)*nb_couches);
    int nb_neurones[nb_couches+1];

    reseau->nb_couches = nb_couches;

    for (int i=0; i < nb_couches; i++) {
        nb_neurones[i] = i*(nb_min_neurones-nb_max_neurones)/(nb_couches-1) + nb_max_neurones;
    }
    nb_neurones[nb_couches] = 0;

    for (int i=0; i < nb_couches; i++) {
        reseau->couches[i] = creer_couche(nb_neurones[i], nb_neurones[i+1]);
    }
    return reseau;
}

int main() {
    Reseau* reseau = creer_reseau(5, 300, 10);
    ecrire_reseau(".test-cache/neuron_io.bin", reseau);
    return 1;
}