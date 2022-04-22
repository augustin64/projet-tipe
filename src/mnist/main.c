#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "neural_network.c"
#include "neuron_io.c"
#include "mnist.c"


int indice_max(float* tab, int n) {
    int indice = -1;
    float maxi = FLT_MIN;
    
    for (int i=0; i < n; i++) {
        if (tab[i] > maxi) {
            maxi = tab[i];
            indice = i;
        }
    }
    return indice;
}

void help(char* call) {
    printf("Usage: %s ( train | recognize ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\ttrain:\n");
    printf("\t\t--batches | -b [int]\tNombre de batches.\n");
    printf("\t\t--couches | -c [int]\tNombres de couches.\n");
    printf("\t\t--neurons | -n [int]\tNombre de neurones sur la première couche.\n");
    printf("\t\t--recover | -r [FILENAME]\tRécupérer depuis un modèle existant.\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
    printf("\trecognize:\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\t\t--in      | -i [FILENAME]\tFichier contenant les images à reconnaître.\n");
    printf("\t\t--out     | -o (text|json)\tFormat de sortie.\n");
    printf("\ttest:\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurones.\n");
}


void ecrire_image_dans_reseau(int** image, Reseau* reseau, int height, int width) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            reseau->couches[0]->neurones[i*height+j]->z = (float)image[i][j] / 255.0f;
        }
    }
}


void train(int batches, int couches, int neurons, char* recovery, char* image_file, char* label_file, char* out) {
    // Entraînement du réseau sur le set de données MNIST
    Reseau* reseau;

    //int* repartition = malloc(sizeof(int)*couches);
    int nb_neurones_der = 10;
    int repartition[4] = {784, 16, 16, nb_neurones_der};

    float* sortie = malloc(sizeof(float)*nb_neurones_der);
    int* sortie_voulue;
    float accuracy;
    //generer_repartition(couches, repartition);

    /*
    * On repart d'un réseau déjà créée stocké dans un fichier
    * ou on repart de zéro si aucune backup n'est fournie
    * */
    if (! recovery) {
        reseau = malloc(sizeof(Reseau));
        creation_du_reseau_neuronal(reseau, repartition, couches);
        initialisation_du_reseau_neuronal(reseau);
    } else {
        reseau = lire_reseau(recovery);
        printf("Backup restaurée.\n");
    }

    Couche* der_couche = reseau->couches[reseau->nb_couches-1];

    // Chargement des images du set de données MNIST
    int* parameters = read_mnist_images_parameters(image_file);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    int*** images = read_mnist_images(image_file);
    unsigned int* labels = read_mnist_labels(label_file);

    for (int i=0; i < batches; i++) {
        printf("Batch [%d/%d]", i, batches);
        accuracy = 0.;

        for (int j=0; j < nb_images; j++) {
            printf("\rBatch [%d/%d]\tImage [%d/%d]",i, batches, j, nb_images);

            ecrire_image_dans_reseau(images[j], reseau, height, width);
            sortie_voulue = creation_de_la_sortie_voulue(reseau, labels[j]);
            forward_propagation(reseau);

            for (int k=0; k < nb_neurones_der; k++) {
                sortie[k] = der_couche->neurones[k]->z;
            }
            if (indice_max(sortie, nb_neurones_der) == labels[j]) {
                accuracy += 1. / (float)nb_images;
            }

            backward_propagation(reseau, sortie_voulue);

        }
        printf("\rBatch [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%\n",i, batches, nb_images, nb_images, accuracy*100);

        modification_du_reseau_neuronal(reseau, nb_images);
        ecrire_reseau(out, reseau);
    }
    suppression_du_reseau_neuronal(reseau);
}

float** recognize(char* modele, char* entree) {
    Reseau* reseau = lire_reseau(modele);
    Couche* derniere_couche = reseau->couches[reseau->nb_couches-1];

    int* parameters = read_mnist_images_parameters(entree);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    int*** images = read_mnist_images(entree);
    float** results = malloc(sizeof(float*)*nb_images);

    for (int i=0; i < nb_images; i++) {
        results[i] = malloc(sizeof(float)*derniere_couche->nb_neurones);

        ecrire_image_dans_reseau(images[i], reseau, height, width);
        forward_propagation(reseau);

        for (int j=0; j < derniere_couche->nb_neurones; j++) {
            results[i][j] = derniere_couche->neurones[j]->z;
        }
    }
    suppression_du_reseau_neuronal(reseau);

    return results;
}

void print_recognize(char* modele, char* entree, char* sortie) {
    Reseau* reseau = lire_reseau(modele);
    int nb_der_couche = reseau->couches[reseau->nb_couches-1]->nb_neurones;

    suppression_du_reseau_neuronal(reseau);

    int* parameters = read_mnist_images_parameters(entree);
    int nb_images = parameters[0];

    float** resultats = recognize(modele, entree);

    if (! strcmp(sortie, "json")) {
        printf("{\n");
    }
    for (int i=0; i < nb_images; i++) {
        if (! strcmp(sortie, "text"))
            printf("Image %d\n", i);
        else
            printf("\"%d\" : [", i);

        for (int j=0; j < nb_der_couche; j++) {
            if (! strcmp(sortie, "json")) {
                printf("%f", resultats[i][j]);

                if (j+1 < nb_der_couche) {
                    printf(", ");
                }
            } else
                printf("Probabilité %d: %f\n", j, resultats[i][j]);
        }
        if (! strcmp(sortie, "json")) {
            if (i+1 < nb_images) {
                printf("],\n");
            } else {
                printf("]\n");
            }
        }
    }
    if (! strcmp(sortie, "json"))
        printf("}\n");

}

void test(char* modele, char* fichier_images, char* fichier_labels) {
    Reseau* reseau = lire_reseau(modele);
    int nb_der_couche = reseau->couches[reseau->nb_couches-1]->nb_neurones;

    suppression_du_reseau_neuronal(reseau);

    int* parameters = read_mnist_images_parameters(fichier_images);
    int nb_images = parameters[0];

    float** resultats = recognize(modele, fichier_images);
    unsigned int* labels = read_mnist_labels(fichier_labels);
    float accuracy;

    for (int i=0; i < nb_images; i++) {
        if (indice_max(resultats[i], nb_der_couche) == labels[i]) {
                accuracy += 1. / (float)nb_images;
        }
    }
    printf("%d Images\tAccuracy: %0.1f%%\n", nb_images, accuracy*100);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Pas d'action spécifiée\n");
        help(argv[0]);
        exit(1);
    }
    if (! strcmp(argv[1], "train")) {
        int batches = 5;
        int couches = 4;
        int neurons = 784;
        char* images = NULL;
        char* labels = NULL;
        char* recovery = NULL;
        char* out = NULL;
        int i = 2;
        while (i < argc) {
            // Utiliser un switch serait sans doute plus élégant
            if ((! strcmp(argv[i], "--batches"))||(! strcmp(argv[i], "-b"))) {
                batches = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else
                if ((! strcmp(argv[i], "--couches"))||(! strcmp(argv[i], "-c"))) {
                couches = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--neurons"))||(! strcmp(argv[i], "-n"))) {
                neurons = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                images = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--labels"))||(! strcmp(argv[i], "-l"))) {
                labels = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--recover"))||(! strcmp(argv[i], "-r"))) {
                recovery = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! images) {
            printf("Pas de fichier d'images spécifié\n");
            exit(1);
        }
        if (! labels) {
            printf("Pas de fichier de labels spécifié\n");
            exit(1);
        }
        if (! out) {
            printf("Pas de fichier de sortie spécifié, default: out.bin\n");
            out = "out.bin";
        }
        // Entraînement en sourçant neural_network.c
        train(batches, couches, neurons, recovery, images, labels, out);
        exit(0);
    }
    if (! strcmp(argv[1], "recognize")) {
        char* in = NULL;
        char* modele = NULL;
        char* out = NULL;
        int i = 2;
        while(i < argc) {
            if ((! strcmp(argv[i], "--in"))||(! strcmp(argv[i], "-i"))) {
                in = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! in) {
            printf("Pas d'entrée spécifiée\n");
            exit(1);
        }
        if (! modele) {
            printf("Pas de modèle spécifié\n");
            exit(1);
        }
        if (! out) {
            out = "text";
        }
        print_recognize(modele, in, out);
        // Reconnaissance puis affichage des données sous le format spécifié
        exit(0);
    }
    if (! strcmp(argv[1], "test")) {
        char* modele = NULL;
        char* images = NULL;
        char* labels = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                images = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--labels"))||(! strcmp(argv[i], "-l"))) {
                labels = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            }
        }
        test(modele, images, labels);
        exit(0);
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}