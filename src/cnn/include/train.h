#include "struct.h"
#include "jpeg.h"

#ifndef DEF_TRAIN_H
#define DEF_TRAIN_H

#include "config.h"


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
    int nb_images; // Nombre d'images à traiter
    float accuracy; // Accuracy (à renvoyer)
    float loss; // Loss (à renvoyer)

    bool offset; // Décalage aléatoire de l'image
} TrainParameters;

/*
 * Structure donnée en argument à la fonction 'load_image'
*/
typedef struct LoadImageParameters {
    jpegDataset* dataset; // Dataset si de type JPEG
    int index; // Numéro de l'image à charger
} LoadImageParameters;

/*
 * Partie entière supérieure de a/b
*/
int div_up(int a, int b);

/*
 * Fonction auxiliaire pour charger (ouvrir et décompresser) les images de manière asynchrone
 * économise environ 20ms par image pour des images de taille 256*256*3
*/
void* load_image(void* parameters);

/*
 * Fonction auxiliaire d'entraînement destinée à être exécutée sur plusieurs threads à la fois
*/
void* train_thread(void* parameters);

/*
 * Fonction principale d'entraînement du réseau neuronal convolutif
*/
void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out, char* recover, bool with_offset);

#endif