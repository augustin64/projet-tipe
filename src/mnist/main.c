#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <sys/sysinfo.h>

#include "neural_network.c"
#include "neuron_io.c"
#include "mnist.c"

#define EPOCHS 10
#define BATCHES 100


typedef struct TrainParameters {
    Network* network;
    int*** images;
    int* labels;
    int start;
    int nb_images;
    int height;
    int width;
    float accuracy;
} TrainParameters;


void print_image(unsigned int width, unsigned int height, int** image, float* previsions) {
    char tab[] = {' ', '.', ':', '%', '#', '\0'};

    for (int i=0; i < (int)height; i++) {
        for (int j=0; j < (int)width; j++) {
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
    printf("Usage: %s ( train | recognize | test ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\ttrain:\n");
    printf("\t\t--epochs  | -e [int]\tNombre d'époques (itérations sur tout le set de données).\n");
    printf("\t\t--couches  | -c [int]\tNombres de couches.\n");
    printf("\t\t--neurones | -n [int]\tNombre de neurones sur la première couche.\n");
    printf("\t\t--recover | -r [FILENAME]\tRécupérer depuis un modèle existant.\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
    printf("\t\t--delta   | -d [FILENAME]\tFichier où écrire le réseau différentiel.\n");
    printf("\t\t--nb-images | -N [int]\tNombres d'images à traiter.\n");
    printf("\t\t--start   | -s [int]\tPremière image à traiter.\n");
    printf("\trecognize:\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\t\t--in      | -i [FILENAME]\tFichier contenant les images à reconnaître.\n");
    printf("\t\t--out     | -o (text|json)\tFormat de sortie.\n");
    printf("\ttest:\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\t\t--preview-fails | -p\tAfficher les images ayant échoué.\n");
}


void write_image_in_network(int** image, Network* network, int height, int width) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            network->layers[0]->neurons[i*height+j]->z = (float)image[i][j] / 255.0f;
        }
    }
}

void* train_images(void* parameters) {
    TrainParameters* param = (TrainParameters*)parameters;
    Network* network = param->network;
    Layer* last_layer = network->layers[network->nb_layers-1];
    int nb_neurons_last_layer = last_layer->nb_neurons;

    int*** images = param->images;
    int* labels = param->labels;

    int start = param->start;
    int nb_images = param->nb_images;
    int height = param->height;
    int width = param->width;
    float accuracy = 0.;
    float* sortie = (float*)malloc(sizeof(float)*nb_neurons_last_layer);
    int* desired_output;

    for (int i=start; i < start+nb_images; i++) {
        write_image_in_network(images[i], network, height, width);
        desired_output = desired_output_creation(network, labels[i]);
        forward_propagation(network);
        backward_propagation(network, desired_output);

        for (int k=0; k < nb_neurons_last_layer; k++) {
            sortie[k] = last_layer->neurons[k]->z;
        }
        if (indice_max(sortie, nb_neurons_last_layer) == labels[i]) {
            accuracy += 1.;
        }
        free(desired_output);
    }
    free(sortie);
    param->accuracy = accuracy;
}


void train(int epochs, int layers, int neurons, char* recovery, char* image_file, char* label_file, char* out, char* delta, int nb_images_to_process, int start) {
    // Entraînement du réseau sur le set de données MNIST
    Network* network;
    Network* delta_network;

    //int* repartition = malloc(sizeof(int)*layers);
    int nb_neurons_last_layer = 10;
    int repartition[2] = {784, nb_neurons_last_layer};

    float accuracy;

    int nb_threads = get_nprocs();
    pthread_t *tid = (pthread_t *)malloc(nb_threads * sizeof(pthread_t));
    //generer_repartition(layers, repartition);

    /*
    * On repart d'un réseau déjà créée stocké dans un fichier
    * ou on repart de zéro si aucune backup n'est fournie
    * */
    if (! recovery) {
        network = (Network*)malloc(sizeof(Network));
        network_creation(network, repartition, layers);
        network_initialisation(network);
    } else {
        network = read_network(recovery);
        printf("Backup restaurée.\n");
    }

    if (delta != NULL) {
        // On initialise un réseau complet mais la seule partie qui nous intéresse est la partie différentielle
        delta_network = (Network*)malloc(sizeof(Network));

        int* repart = (int*)malloc(sizeof(network->nb_layers));
        for (int i=0; i < network->nb_layers; i++) {
            repart[i] = network->layers[i]->nb_neurons;
        }

        network_creation(delta_network, repart, network->nb_layers);
        network_initialisation(delta_network);
        free(repart);
    }

    // Chargement des images du set de données MNIST
    int* parameters = read_mnist_images_parameters(image_file);
    int nb_images_total = parameters[0];
    int nb_remaining_images = 0; // Nombre d'images restantes dans un batch
    int height = parameters[1];
    int width = parameters[2];

    int*** images = read_mnist_images(image_file);
    unsigned int* labels = read_mnist_labels(label_file);

    if (nb_images_to_process != -1) {
        nb_images_total = nb_images_to_process;
    }

    TrainParameters** train_parameters = (TrainParameters**)malloc(sizeof(TrainParameters*)*nb_threads);
    for (int i=0; i < epochs; i++) {
        accuracy = 0.;
        for (int k=0; k < nb_images_total / BATCHES; k++) {
            nb_remaining_images = BATCHES;

            for (int j=0; j < nb_threads; j++) {
                train_parameters[j] = (TrainParameters*)malloc(sizeof(TrainParameters));
                train_parameters[j]->network = copy_network(network);
                train_parameters[j]->images = (int***)images;
                train_parameters[j]->labels = (int*)labels;
                train_parameters[j]->nb_images = BATCHES / nb_threads;
                train_parameters[j]->start = nb_images_total - BATCHES*(nb_images_total / BATCHES - k -1) - nb_remaining_images + start;
                train_parameters[j]->height = height;
                train_parameters[j]->width = width;

                if (j == nb_threads-1) {
                    train_parameters[j]->nb_images = nb_remaining_images;
                }
                nb_remaining_images -= train_parameters[j]->nb_images;

                pthread_create( &tid[j], NULL, train_images, (void*) train_parameters[j]);
            }
            for(int j=0; j < nb_threads; j++ ) {
                pthread_join( tid[j], NULL );
                accuracy += train_parameters[j]->accuracy / (float) nb_images_total;
                patch_network(network, train_parameters[j]->network, train_parameters[j]->nb_images);
                if (delta != NULL)
                    patch_delta(delta_network, train_parameters[j]->network, train_parameters[j]->nb_images);
                deletion_of_network(train_parameters[j]->network);
                free(train_parameters[j]);
            }
            printf("\rThread [%d/%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%", nb_threads, nb_threads, i, epochs, BATCHES*(k+1), nb_images_total, accuracy*100);
        }
        printf("\rThread [%d/%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%\n", nb_threads, nb_threads, i, epochs, nb_images_total, nb_images_total, accuracy*100);
        write_network(out, network);
        if (delta != NULL)
            write_delta_network(delta, delta_network);
    }
    write_network(out, network);
    if (delta != NULL) {
        deletion_of_network(delta_network);
    }
    deletion_of_network(network);
    free(train_parameters);
    free(tid);
}

float** recognize(char* modele, char* entree) {
    Network* network = read_network(modele);
    Layer* derniere_layer = network->layers[network->nb_layers-1];

    int* parameters = read_mnist_images_parameters(entree);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    int*** images = read_mnist_images(entree);
    float** results = (float**)malloc(sizeof(float*)*nb_images);

    for (int i=0; i < nb_images; i++) {
        results[i] = (float*)malloc(sizeof(float)*derniere_layer->nb_neurons);

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
    int nb_last_layer = network->layers[network->nb_layers-1]->nb_neurons;

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

        for (int j=0; j < nb_last_layer; j++) {
            if (! strcmp(sortie, "json")) {
                printf("%f", resultats[i][j]);

                if (j+1 < nb_last_layer) {
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
    if (! strcmp(sortie, "json")) {
        printf("}\n");
    }

}

void test(char* modele, char* fichier_images, char* fichier_labels, bool preview_fails) {
    Network* network = read_network(modele);
    int nb_last_layer = network->layers[network->nb_layers-1]->nb_neurons;

    deletion_of_network(network);

    int* parameters = read_mnist_images_parameters(fichier_images);
    int nb_images = parameters[0];
    int width = parameters[1];
    int height = parameters[2];
    int*** images = read_mnist_images(fichier_images);

    float** resultats = recognize(modele, fichier_images);
    unsigned int* labels = read_mnist_labels(fichier_labels);
    float accuracy = 0.;

    for (int i=0; i < nb_images; i++) {
        if (indice_max(resultats[i], nb_last_layer) == (int)labels[i]) {
            accuracy += 1. / (float)nb_images;
        } else if (preview_fails) {
            printf("--- Image %d, %d --- Prévision: %d ---\n", i, labels[i], indice_max(resultats[i], nb_last_layer));
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
        int epochs = EPOCHS;
        int layers = 2;
        int neurons = 784;
        int nb_images = -1;
        int start = 0;
        char* images = NULL;
        char* labels = NULL;
        char* recovery = NULL;
        char* out = NULL;
        char* delta = NULL;
        int i = 2;
        while (i < argc) {
            // Utiliser un switch serait sans doute plus élégant
            if ((! strcmp(argv[i], "--epochs"))||(! strcmp(argv[i], "-e"))) {
                epochs = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else
                if ((! strcmp(argv[i], "--couches"))||(! strcmp(argv[i], "-c"))) {
                layers = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--neurones"))||(! strcmp(argv[i], "-n"))) {
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
            } else if ((! strcmp(argv[i], "--delta"))||(! strcmp(argv[i], "-d"))) {
                delta = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--nb-images"))||(! strcmp(argv[i], "-N"))) {
                nb_images = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--start"))||(! strcmp(argv[i], "-s"))) {
                start = strtol(argv[i+1], NULL, 10);
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
        train(epochs, layers, neurons, recovery, images, labels, out, delta, nb_images, start);
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
