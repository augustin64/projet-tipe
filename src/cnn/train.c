#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <time.h>

#include "../mnist/include/mnist.h"
#include "include/initialisation.h"
#include "include/neuron_io.h"
#include "../include/colors.h"
#include "include/function.h"
#include "include/creation.h"
#include "include/update.h"
#include "include/utils.h"
#include "include/free.h"
#include "include/jpeg.h"
#include "include/cnn.h"

#include "include/train.h"

int div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


void* train_thread(void* parameters) {
    TrainParameters* param = (TrainParameters*)parameters;
    Network* network = param->network;
    imgRawImage* image;
    int maxi;

    int*** images = param->images;
    int* labels = (int*)param->labels;
    int* index = param->index;

    int width = param->width;
    int height = param->height;
    int dataset_type = param->dataset_type;
    int start = param->start;
    int nb_images = param->nb_images;
    float accuracy = 0.;
    for (int i=start;  i < start+nb_images; i++) {
        if (dataset_type == 0) {
            write_image_in_network_32(images[index[i]], height, width, network->input[0][0]);
            forward_propagation(network);
            maxi = indice_max(network->input[network->size-1][0][0], 10);
            backward_propagation(network, labels[i]);

            if (maxi == labels[index[i]]) {
                accuracy += 1.;
            }
        } else {
            if (!param->dataset->images[index[i]]) {
                image = loadJpegImageFile(param->dataset->fileNames[index[i]]);
                param->dataset->images[index[i]] = image->lpData;
                free(image);
            }
            write_image_in_network_260(param->dataset->images[index[i]], height, width, network->input[0]);
            forward_propagation(network);
            maxi = indice_max(network->input[network->size-1][0][0], param->dataset->numCategories);
            backward_propagation(network, param->dataset->labels[index[i]]);

            if (maxi == (int)param->dataset->labels[index[i]]) {
                accuracy += 1.;
            }

            free(param->dataset->images[index[i]]);
            param->dataset->images[index[i]] = NULL;
        }
    }

    param->accuracy = accuracy;
    return NULL;
}


void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out) {
    srand(time(NULL));
    int input_dim = -1;
    int input_depth = -1;
    float accuracy;
    float current_accuracy;

    int nb_images_total; // Images au total
    int nb_images_total_remaining; // Images restantes dans un batch
    int batches_epoques; // Batches par époque

    int*** images; // Images sous forme de tableau de tableaux de tableaux de pixels (degré de gris, MNIST)
    unsigned int* labels; // Labels associés aux images du dataset MNIST
    jpegDataset* dataset; // Structure de données décrivant un dataset d'images jpeg
    int* shuffle_index; // shuffle_index[i] contient le nouvel index de l'élément à l'emplacement i avant mélange

    if (dataset_type == 0) { // Type MNIST
        // Chargement des images du set de données MNIST
        int* parameters = read_mnist_images_parameters(images_file);
        nb_images_total = parameters[0];
        free(parameters);

        images = read_mnist_images(images_file);
        labels = read_mnist_labels(labels_file);

        input_dim = 32;
        input_depth = 1;
    } else { // Type JPG
        dataset = loadJpegDataset(data_dir);
        input_dim = dataset->height + 4; // image_size + padding
        input_depth = dataset->numComponents;

        nb_images_total = dataset->numImages;
    }

    // Initialisation du réseau
    Network* network = create_network_lenet5(1, 0, TANH, GLOROT, input_dim, input_depth);

    shuffle_index = (int*)malloc(sizeof(int)*nb_images_total);
    for (int i=0; i < nb_images_total; i++) {
        shuffle_index[i] = i;
    }

    #ifdef USE_MULTITHREADING
    int nb_remaining_images; // Nombre d'images restantes à lancer pour une série de threads
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
            param->dataset = NULL;
            param->width = 28;
            param->height = 28;
        } else {
            param->dataset = dataset;
            param->width = dataset->width;
            param->height = dataset->height;
            param->images = NULL;
            param->labels = NULL;
        }
        param->nb_images = BATCHES / nb_threads;
        param->index = shuffle_index;
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
        train_params->dataset = NULL;
    } else {
        train_params->dataset = dataset;
        train_params->width = dataset->width;
        train_params->height = dataset->height;
        train_params->images = NULL;
        train_params->labels = NULL;
    }
    train_params->nb_images = BATCHES;
    train_params->index = shuffle_index;
    #endif

    for (int i=0; i < epochs; i++) {
        // La variable accuracy permet d'avoir une ESTIMATION
        // du taux de réussite et de l'entraînement du réseau,
        // mais n'est en aucun cas une valeur réelle dans le cas
        // du multi-threading car chaque copie du réseau initiale sera légèrement différente
        // et donnera donc des résultats différents sur les mêmes images.
        accuracy = 0.;
        knuth_shuffle(shuffle_index, nb_images_total);
        batches_epoques = div_up(nb_images_total, BATCHES);
        nb_images_total_remaining = nb_images_total;
        for (int j=0; j < batches_epoques; j++) {
            #ifdef USE_MULTITHREADING
            if (j == batches_epoques-1) {
                nb_remaining_images = nb_images_total_remaining;
                nb_images_total_remaining = 0;
            } else {
                nb_images_total_remaining -= BATCHES;
                nb_remaining_images = BATCHES;
            }

            for (int k=0; k < nb_threads; k++) {
                if (k == nb_threads-1) {
                    train_parameters[k]->nb_images = nb_remaining_images;
                    nb_remaining_images = 0;
                } else {
                    nb_remaining_images -= BATCHES / nb_threads;
                }
                train_parameters[k]->start = BATCHES*j + (BATCHES/nb_threads)*k;
                train_parameters[k]->network = copy_network(network);

                pthread_create( &tid[k], NULL, train_thread, (void*) train_parameters[k]);
            }
            for (int k=0; k < nb_threads; k++) {
                // On attend la terminaison de chaque thread un à un
                pthread_join( tid[k], NULL );
                accuracy += train_parameters[k]->accuracy / (float) nb_images_total;

                update_weights(network, train_parameters[k]->network, train_parameters[k]->nb_images);
                update_bias(network, train_parameters[k]->network, train_parameters[k]->nb_images);
                free_network(train_parameters[k]->network);
            }
            current_accuracy = accuracy * nb_images_total/((j+1)*BATCHES);
            printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "YELLOW"%0.1f%%"RESET" ", nb_threads, i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100);
            fflush(stdout);
            #else
            (void)nb_images_total_remaining; // Juste pour enlever un warning

            train_params->start = j*BATCHES;
            
            train_thread((void*)train_params);
            
            accuracy += train_params->accuracy / (float) nb_images_total;
            current_accuracy = accuracy * nb_images_total/((j+1)*BATCHES);
            
            update_weights(network, network, train_params->nb_images);
            update_bias(network, network, train_params->nb_images);
            
            printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "YELLOW"%0.1f%%"RESET" ", i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100);
            fflush(stdout);
            #endif
        }
        #ifdef USE_MULTITHREADING
        printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "GREEN"%0.1f%%"RESET" \n", nb_threads, i, epochs, nb_images_total, nb_images_total, accuracy*100);
        #else
        printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "GREEN"%0.1f%%"RESET" \n", i, epochs, nb_images_total, nb_images_total, accuracy*100);
        #endif
        write_network(out, network);
    }
    free(shuffle_index);
    free_network(network);
    #ifdef USE_MULTITHREADING
    free(tid);
    #else
    free(train_params);
    #endif
}
