#ifndef DEF_CONFIG_H
#define DEF_CONFIG_H


//* Paramètres d'entraînement
#define EPOCHS 10 // Nombre d'époques par défaut (itérations sur toutes les images)
#define BATCHES 32 // Nombre d'images à voir avant de mettre le réseau à jour
#define LEARNING_RATE 3e-4 // Taux d'apprentissage
#define USE_MULTITHREADING // Commenter pour utiliser un seul coeur durant l'apprentissage (meilleur pour des tailles de batchs traités rapidement)

//* Paramètres d'ADAM optimizer
#define ALPHA 3e-4
#define BETA_1 0.9
#define BETA_2 0.999
#define Epsilon 1e-7

//* Options d'ADAM optimizer
//* Activer ou désactiver Adam sur les couches dense
//#define ADAM_DENSE_WEIGHTS
//#define ADAM_DENSE_BIAS
//* Activer ou désactiver Adam sur les couches convolutives
//#define ADAM_CNN_WEIGHTS
//#define ADAM_CNN_BIAS


//* Paramètre d'optimisation pour un dataset Jpeg
// keep images in ram e.g re-read and decompress each time
// Enabling this will lead to a large amount of ram used while economizing not that
// much computing power
// Note: 50States10K dataset is 90Go once decompressed, use with caution
//#define STORE_IMAGES_TO_RAM


//* Limite du réseau
// Des valeurs trop grandes dans le réseau risqueraient de provoquer des overflows notamment.
// On utilise donc la méthode gradient_clipping,
// qui consiste à majorer tous les biais et poids par un hyper-paramètre choisi précédemment.
// https://arxiv.org/pdf/1905.11881.pdf
#define NETWORK_CLIP_VALUE 300

//* Paramètres CUDA
#define BLOCKSIZE_x 10
#define BLOCKSIZE_y 10
#define BLOCKSIZE_z 10

#endif