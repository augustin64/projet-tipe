#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <pthread.h>
#include <sys/sysinfo.h>

#include "../mnist/mnist.c"
#include "../colors.h"
#include "neuron_io.c"
#include "utils.c"
#include "free.c"
#include "cnn.c"

#include "include/train.h"


void* train_thread(void* parameters) {
    TrainParameters* param = (TrainParameters*)parameters;
    Network* network = param->network;

    int*** images = param->images;
    int* labels = param->labels;

    int width = param->width;
    int height = param->height;
    int dataset_type = param->dataset_type;
    int start = param->start;
    int nb_images = param->nb_images;
    float accuracy = 0.;

    for (int i=start;  i < start+nb_images; i++) {
        if (dataset_type == 0) {
            write_image_in_network_32(images[i], height, width, network->input[0][0]);
            forward_propagation(network);
            //backward_propagation(network, labels[i]);

            // TODO get_indice_max(network last layer)
            // TODO if indice_max == labels[i] then accuracy += 1.
        } else {
            printf_error("Dataset de type JPG non implémenté\n");
            exit(1);
        }
    }

    param->accuracy = accuracy;
    return NULL;
}


void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out) {
    int input_dim = -1;
    int input_depth = -1;
    float accuracy;

    int nb_images_total;
    int nb_remaining_images;

    int*** images;
    unsigned int* labels;

    if (dataset_type == 0) { // Type MNIST
        // Chargement des images du set de données MNIST
        int* parameters = read_mnist_images_parameters(images_file);
        nb_images_total = parameters[0];
        free(parameters);

        images = read_mnist_images(images_file);
        labels = read_mnist_labels(labels_file);

        input_dim = 32;
        input_depth = 1;
    } else { // TODO Type JPG
        input_dim = 256;
        input_depth = 3;

        nb_images_total = 0;
        printf_error("Dataset de type jpg non-implémenté.\n");
        exit(1);
    }

    // Initialisation du réseau
    Network* network = create_network_lenet5(0, 0, TANH, GLOROT_NORMAL, input_dim, input_depth);

    #ifdef USE_MULTITHREADING
    // Récupération du nombre de threads disponibles
    int nb_threads = get_nprocs();
    pthread_t *tid = (pthread_t*)malloc(nb_threads * sizeof(pthread_t));

    // Création des paramètres donnés à chaque thread dans le cas du multi-threading
    TrainParameters** train_parameters = (TrainParameters**)malloc(sizeof(TrainParameters*)*nb_threads);
    TrainParameters* param;

    for (int k=0; k < nb_threads; k++) {
        train_parameters[k] = (TrainParameters*)malloc(sizeof(TrainParameters));
        param = train_parameters[k];
        param->dataset_type = dataset_type;
        if (dataset_type == 0) {
            param->images = images;
            param->labels = labels;
            param->data_dir = NULL;
            param->width = 28;
            param->height = 28;
        } else {
            param->data_dir = data_dir;
            param->images = NULL;
            param->labels = NULL;
        }
        param->nb_images = BATCHES / nb_threads;
    }
    #else
    // Création des paramètres donnés à l'unique
    // thread dans l'hypothèse ou le multi-threading n'est pas utilisé.

    // Cela est utile à des fins de débogage notamment,
    // où l'utilisation de threads rend vite les choses plus compliquées qu'elles ne le sont.
    TrainParameters* train_params = (TrainParameters*)malloc(sizeof(TrainParameters));
    
    train_params->network = network;
    train_params->dataset_type = dataset_type;
    if (dataset_type == 0) {
        train_params->images = images;
        train_params->labels = labels;
        train_params->width = 28;
        train_params->height = 28;
        train_params->data_dir = NULL;
    } else {
        train_params->data_dir = data_dir;
        train_params->images = NULL;
        train_params->width = 0;
        train_params->height = 0;
        train_params->labels = NULL;
    }
    train_params->nb_images = BATCHES;
    #endif

    for (int i=0; i < epochs; i++) {
        // La variable accuracy permet d'avoir une ESTIMATION
        // du taux de réussite et de l'entraînement du réseau,
        // mais n'est en aucun cas une valeur réelle dans le cas
        // du multi-threading car chaque copie du réseau initiale sera légèrement différente
        // et donnera donc des résultats différents sur les mêmes images.
        accuracy = 0.;
        for (int j=0; j < nb_images_total / BATCHES; j++) {
            nb_remaining_images = BATCHES;

            #ifdef USE_MULTITHREADING
            for (int k=0; k < nb_threads; k++) {
                if (k == nb_threads-1) {
                    train_parameters[k]->nb_images = nb_remaining_images;
                    nb_remaining_images = 0;
                } else {
                    nb_remaining_images -= BATCHES / nb_threads;
                }
                train_parameters[k]->network = copy_network(network);
                train_parameters[k]->start = BATCHES*j + (nb_images_total/BATCHES)*k;
                pthread_create( &tid[j], NULL, train_thread, (void*) train_parameters[k]);
            }
            for (int k=0; k < nb_threads; k++) {
                // On attend la terminaison de chaque thread un à un
                pthread_join( tid[j], NULL );
                accuracy += train_parameters[k]->accuracy / (float) nb_images_total;
                // TODO patch_network(network, train_parameters[k]->network, train_parameters[k]->nb_images);
                free_network(train_parameters[k]->network);
            }
            printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%", nb_threads, i, epochs, BATCHES*(j+1), nb_images_total, accuracy*100);
            #else
            train_params->start = j*BATCHES;
            train_thread((void*)train_params);
            accuracy += train_params->accuracy / (float) nb_images_total;
            printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%", i, epochs, BATCHES*(j+1), nb_images_total, accuracy*100);
            #endif
        }
        #ifdef USE_MULTITHREADING
        printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%\n", nb_threads, i, epochs, nb_images_total, nb_images_total, accuracy*100);
        #else
        printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: %0.1f%%\n", i, epochs, nb_images_total, nb_images_total, accuracy*100);
        #endif
        write_network(out, network);
    }
    free_network(network);
    #ifdef USE_MULTITHREADING
    free(tid);
    #else
    free(train_params);
    #endif
}
