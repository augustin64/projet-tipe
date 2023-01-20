#include "struct.h"
#include "jpeg.h"

#ifndef DEF_TRAIN_H
#define DEF_TRAIN_H

#define EPOCHS 10
#define BATCHES 500
#define USE_MULTITHREADING
#define LEARNING_RATE 0.01


/*
* Structure donnée en argument à la fonction 'train_thread'
*/
typedef struct TrainParameters {
    Network* network; // Réseau
    jpegDataset* dataset; // Dataset si de type JPEG
    int* index; // Sert à réordonner les images
    int*** images; // Images si de type MNIST
    unsigned int* labels; // Labels si de type MNIST
    int width; // Largeur des images
    int height; // Hauteur des images
    int dataset_type; // Type de dataset
    int start; // Début des images
    int nb_images; // Nombre d'images àn traiter
    float accuracy; // Accuracy (à renvoyer)
    float loss; // Loss (à renvoyer)
} TrainParameters;


/*
 * Fonction auxiliaire d'entraînement destinée à être exécutée sur plusieurs threads à la fois
*/
void* train_thread(void* parameters);

/*
 * Fonction principale d'entraînement du réseau neuronal convolutif
*/
void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out, char* recover);

#endif