#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "struct/neuron.c"


#define MAGIC_NUMBER 2023

/*
Neurone* lire_neurones(uint32_t nb_neurones, uint32_t nb_poids_sortants) {
    // le malloc dépend en fait du nombre de neurones de la couche suivante,
    // il faut donc rentrer plus en détails sur la taille de l'allocation
    Neurone* tab = malloc(sizeof(Neurone)*nb_neurones);
    return tab;
}

// Charge l'entièreté du réseau neuronal depuis un fichier binaire
Couche* lire_reseau(char* filename) {
    FILE *ptr;
    Reseau* reseau = malloc(sizeof(int)+sizeof(Couche*));
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t nb_couches;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    if (magic_number != MAGIC_NUMBER) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&nb_couches, sizeof(uint32_t), 1, ptr);


    Couche* couches = malloc((sizeof(int)+sizeof(Neurone*)*nb_couches));
    uint32_t nb_neurones_couche[nb_couches];

    reseau->couche  = couches;

    fread(&nb_neurones_couche, sizeof(nb_neurones_couche), 1, ptr);

    for (int i=0; i < nb_couches+1; i++) {
        couches[i].nb_neurone = nb_neurones_couche[i];
        couches[i].neurone = lire_neurones(couches[i].nb_neurone, nb_neurones_couche[i+1]);
    }

    fclose(ptr);
    return couches;
}
*/


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