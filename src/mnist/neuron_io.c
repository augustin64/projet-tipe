#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "struct/neuron.c"


#define MAGIC_NUMBER 2023


Neurone* lire_neurone(uint32_t nb_poids_sortants, FILE *ptr) {
    Neurone* neurone = malloc(sizeof(float)*(6+2*nb_poids_sortants));
    float activation;
    float biais;
    float poids_sortants[nb_poids_sortants];

    fread(&activation, sizeof(float), 1, ptr);
    fread(&biais, sizeof(float), 1, ptr);
    fread(&poids_sortants, sizeof(float)*nb_poids_sortants, 1, ptr);

    neurone->activation = activation;
    neurone->biais = biais;

    neurone->z = 0.0;
    neurone->dactivation = 0.0;
    neurone->dbiais = 0.0;
    neurone->dz = 0.0;

    for (int i=0; i < nb_poids_sortants; i++) {
        neurone->poids_sortants[i] = poids_sortants[i];
        neurone->dw[i] = 0.0;
    }

    return neurone;
}


// Lit une couche de neurones
Neurone** lire_neurones(uint32_t nb_neurones, uint32_t nb_poids_sortants, FILE *ptr) {
    Neurone** neurones = malloc(sizeof(Neurone*)*nb_neurones);

    for (int i=0; i < nb_neurones; i++) {
        neurones[i] = lire_neurone(nb_poids_sortants, ptr);
    }
    return neurones;
}

// Charge l'entièreté du réseau neuronal depuis un fichier binaire
Reseau* lire_reseau(char* filename) {
    FILE *ptr;
    Reseau* reseau = malloc(sizeof(int)+sizeof(Couche**));
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t nb_couches;
    reseau->nb_couches = nb_couches;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    if (magic_number != MAGIC_NUMBER) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&nb_couches, sizeof(uint32_t), 1, ptr);


    Couche** couches = malloc(sizeof(Couche*)*nb_couches);
    uint32_t nb_neurones_couche[nb_couches+1];

    reseau->couches  = couches;

    fread(&nb_neurones_couche, sizeof(nb_neurones_couche)-sizeof(uint32_t), 1, ptr);
    nb_neurones_couche[nb_couches] = 0;

    for (int i=0; i < nb_couches; i++) {
        couches[i]->nb_neurones = nb_neurones_couche[i];
    }

    for (int i=0; i < nb_couches; i++) {
        couches[i]->neurones = lire_neurones(couches[i]->nb_neurones, couches[i+1]->nb_neurones, ptr);
    }

    fclose(ptr);
    return reseau;
}



// Écrit un neurone dans le fichier pointé par *ptr
void ecrire_neurone(Neurone* neurone, int poids_sortants, FILE *ptr) {
    float buffer[poids_sortants+2];

    buffer[0] = neurone->activation;
    buffer[1] = neurone->biais;
    for (int i=0; i < poids_sortants; i++) {
        buffer[i+2] = neurone->poids_sortants[i];
    }

    fwrite(buffer, sizeof(buffer), 1, ptr);
}


// Stocke l'entièreté du réseau neuronal dans un fichier binaire
int ecrire_reseau(char* filename, Reseau* reseau) {
    FILE *ptr;
    int nb_couches = reseau->nb_couches;
    int nb_neurones[nb_couches+1];

    ptr = fopen(filename, "wb");

    uint32_t buffer[nb_couches+2];

    buffer[0] = MAGIC_NUMBER;
    buffer[1] = nb_couches;
    for (int i=0; i < nb_couches; i++) {
        buffer[i+2] = reseau->couches[i]->nb_neurones;
        nb_neurones[i] = reseau->couches[i]->nb_neurones;
    }
    nb_neurones[nb_couches] = 0;

    fwrite(buffer, sizeof(buffer), 1, ptr);

    for (int i=0; i < nb_couches; i++) {
        for (int j=0; j < nb_neurones[i]; j++) {
            ecrire_neurone(reseau->couches[i]->neurones[j], nb_neurones[i+1], ptr);
        }
    }

    fclose(ptr);
    return 1;
}