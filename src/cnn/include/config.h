#ifndef DEF_CONFIG_H
#define DEF_CONFIG_H

//* Paramètres d'entraînement
#define EPOCHS 10 // Nombre d'époques par défaut (itérations sur toutes les images)
#define BATCHES 32 // Nombre d'images à voir avant de mettre le réseau à jour
#define LEARNING_RATE 3e-4 // Taux d'apprentissage
#define USE_MULTITHREADING // Commenter pour utiliser un seul coeur durant l'apprentissage (meilleur pour des tailles de batchs traités rapidement)


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

#endif