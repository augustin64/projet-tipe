#include "struct.h"

#ifndef DEF_TRAIN_H
#define DEF_TRAIN_H

#define EPOCHS 10
#define BATCHES 100
#define USE_MULTITHREADING


/*
* Structure donnée en argument à la fonction 'train_thread'
*/
typedef struct TrainParameters {
    Network* network;
    int*** images;
    unsigned int* labels;
    int width;
    int height;
    int dataset_type;
    char* data_dir;
    int start;
    int nb_images;
    float accuracy;
} TrainParameters;


/*
* Renvoie l'indice maximal d'un tableau tab de taille n
*/
int indice_max(float* tab, int n);

/*
 * Fonction auxiliaire d'entraînement destinée à être exécutée sur plusieurs threads à la fois
*/
void* train_thread(void* parameters);

/*
 * Fonction principale d'entraînement du réseau neuronal convolutif
*/
void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out);

#endif