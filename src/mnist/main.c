#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "neural_network.c"
#include "neuron_io.c"
#include "mnist.c"

void print_image(unsigned int width, unsigned int height, int** image, float* previsions) {
    char tab[] = {' ', '.', ':', '%', '#', '\0'};

    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%c", tab[image[i][j]/52]);
        }
        if (i < 10) {
            printf("\t%d : %f", i, previsions[i]);
        }
        printf("\n");
    }
}

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
    printf("\t\t--layers  | -c [int]\tNombres de layers.\n");
    printf("\t\t--neurons | -n [int]\tNombre de neurons sur la première layer.\n");
    printf("\t\t--recover | -r [FILENAME]\tRécupérer depuis un modèle existant.\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurons.\n");
    printf("\trecognize:\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurons.\n");
    printf("\t\t--in      | -i [FILENAME]\tFichier contenant les images à reconnaître.\n");
    printf("\t\t--out     | -o (text|json)\tFormat de sortie.\n");
    printf("\ttest:\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurons.\n");
    printf("\t\t--preview-fails | -p\tAfficher les images ayant échoué.\n");
}


void write_image_in_network(int** image, Network* network, int height, int width) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            network->layers[0]->neurons[i*height+j]->z = (float)image[i][j] / 255.0f;
        }
    }
}


void train(int batches, int layers, int neurons, char* recovery, char* image_file, char* label_file, char* out) {
    // Entraînement du réseau sur le set de données MNIST
    Network* network;

    //int* repartition = malloc(sizeof(int)*layers);
    int nb_neurons_der = 10;
    int repartition[2] = {784, nb_neurons_der};

    float* sortie = malloc(sizeof(float)*nb_neurons_der);
    int* desired_output;
    float accuracy;
    float loss;
    //generer_repartition(layers, repartition);

    /*
    * On repart d'un réseau déjà créée stocké dans un fichier
    * ou on repart de zéro si aucune backup n'est fournie
    * */
    if (! recovery) {
        network = malloc(sizeof(Network));
        network_creation(network, repartition, layers);
        network_initialisation(network);
    } else {
        network = read_network(recovery);
        printf("Backup restaurée.\n");
    }

    Layer* der_layer = network->layers[network->nb_layers-1];

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
        loss = 0.;

        for (int j=0; j < nb_images; j++) {
            printf("\rBatch [%d/%d]\tImage [%d/%d]",i, batches, j, nb_images);

            write_image_in_network(images[j], network, height, width);
            desired_output = desired_output_creation(network, labels[j]);
            forward_propagation(network);
            backward_propagation(network, desired_output);

            for (int k=0; k < nb_neurons_der; k++) {
                sortie[k] = der_layer->neurons[k]->z;
            }
            if (indice_max(sortie, nb_neurons_der) == labels[j]) {
                accuracy += 1. / (float)nb_images;
            }
            loss += loss_computing(network, labels[j]) / (float)nb_images;
            free(desired_output);
        }
        network_modification(network, nb_images);
        printf("\rBatch [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%\tLoss: %f\n",i, batches, nb_images, nb_images, accuracy*100, loss);
        write_network(out, network);
    }
    deletion_of_network(network);
}

float** recognize(char* modele, char* entree) {
    Network* network = read_network(modele);
    Layer* derniere_layer = network->layers[network->nb_layers-1];

    int* parameters = read_mnist_images_parameters(entree);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    int*** images = read_mnist_images(entree);
    float** results = malloc(sizeof(float*)*nb_images);

    for (int i=0; i < nb_images; i++) {
        results[i] = malloc(sizeof(float)*derniere_layer->nb_neurons);

        write_image_in_network(images[i], network, height, width);
        forward_propagation(network);

        for (int j=0; j < derniere_layer->nb_neurons; j++) {
            results[i][j] = derniere_layer->neurons[j]->z;
        }
    }
    deletion_of_network(network);

    return results;
}

void print_recognize(char* modele, char* entree, char* sortie) {
    Network* network = read_network(modele);
    int nb_der_layer = network->layers[network->nb_layers-1]->nb_neurons;

    deletion_of_network(network);

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

        for (int j=0; j < nb_der_layer; j++) {
            if (! strcmp(sortie, "json")) {
                printf("%f", resultats[i][j]);

                if (j+1 < nb_der_layer) {
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

void test(char* modele, char* fichier_images, char* fichier_labels, bool preview_fails) {
    Network* network = read_network(modele);
    int nb_der_layer = network->layers[network->nb_layers-1]->nb_neurons;

    deletion_of_network(network);

    int* parameters = read_mnist_images_parameters(fichier_images);
    int nb_images = parameters[0];
    int width = parameters[1];
    int height = parameters[2];
    int*** images = read_mnist_images(fichier_images);

    float** resultats = recognize(modele, fichier_images);
    unsigned int* labels = read_mnist_labels(fichier_labels);
    float accuracy;

    for (int i=0; i < nb_images; i++) {
        if (indice_max(resultats[i], nb_der_layer) == labels[i]) {
            accuracy += 1. / (float)nb_images;
        } else if (preview_fails) {
            printf("--- Image %d, %d --- Prévision: %d ---\n", i, labels[i], indice_max(resultats[i], nb_der_layer));
            print_image(width, height, images[i], resultats[i]);
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
        int batches = 100;
        int layers = 2;
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
                if ((! strcmp(argv[i], "--layers"))||(! strcmp(argv[i], "-c"))) {
                layers = strtol(argv[i+1], NULL, 10);
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
        train(batches, layers, neurons, recovery, images, labels, out);
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
        bool preview_fails = false;
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
            } else if ((! strcmp(argv[i], "--preview-fails"))||(! strcmp(argv[i], "-p"))) {
                preview_fails = true;
                i++;
            }
        }
        test(modele, images, labels, preview_fails);
        exit(0);
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}