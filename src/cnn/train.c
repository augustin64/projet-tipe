#include <sys/sysinfo.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../mnist/include/mnist.h"
#include "include/initialisation.h"
#include "include/neuron_io.h"
#include "../include/colors.h"
#include "../include/utils.h"
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

    float* wanted_output;
    float accuracy = 0.;
    float loss = 0.;

    for (int i=start;  i < start+nb_images; i++) {
        if (dataset_type == 0) {
            write_image_in_network_32(images[index[i]], height, width, network->input[0][0]);
            forward_propagation(network);
            maxi = indice_max(network->input[network->size-1][0][0], 10);
            if (maxi == -1) {
                printf("\n");
                printf_error("Le réseau sature.\n");
                exit(1);
            }
            
            wanted_output = generate_wanted_output(labels[index[i]], 10);
            loss += compute_mean_squared_error(network->input[network->size-1][0][0], wanted_output, 10);
            free(wanted_output);

            backward_propagation(network, labels[index[i]]);

            if (maxi == labels[index[i]]) {
                accuracy += 1.;
            }
        } else {
            if (!param->dataset->images[index[i]]) {
                image = loadJpegImageFile(param->dataset->fileNames[index[i]]);
                param->dataset->images[index[i]] = image->lpData;
                gree(image);
            }
            write_image_in_network_260(param->dataset->images[index[i]], height, width, network->input[0]);
            forward_propagation(network);
            maxi = indice_max(network->input[network->size-1][0][0], param->dataset->numCategories);
            backward_propagation(network, param->dataset->labels[index[i]]);

            if (maxi == (int)param->dataset->labels[index[i]]) {
                accuracy += 1.;
            }

            gree(param->dataset->images[index[i]]);
            param->dataset->images[index[i]] = NULL;
        }
    }

    param->accuracy = accuracy;
    param->loss = loss;
    return NULL;
}


void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out, char* recover) {
    #ifdef USE_CUDA
    bool compatibility = check_cuda_compatibility();
    if (!compatibility) {
        printf("Exiting.\n");
        exit(1);
    }
    #endif
    srand(time(NULL));
    Network* network;
    int input_dim = -1;
    int input_depth = -1;

    float loss;
    float batch_loss; // May be redundant with loss, but gives more informations
    float accuracy;
    float current_accuracy;

    int nb_images_total; // Images au total
    int nb_images_total_remaining; // Images restantes dans un batch
    int batches_epoques; // Batches par époque

    int*** images; // Images sous forme de tableau de tableaux de tableaux de pixels (degré de gris, MNIST)
    unsigned int* labels; // Labels associés aux images du dataset MNIST
    jpegDataset* dataset; // Structure de données décrivant un dataset d'images jpeg
    int* shuffle_index; // shuffle_index[i] contient le nouvel index de l'élément à l'emplacement i avant mélange

    double start_time, end_time;
    double elapsed_time;

    double algo_start = omp_get_wtime();

    start_time = omp_get_wtime();

    if (dataset_type == 0) { // Type MNIST
        // Chargement des images du set de données MNIST
        int* parameters = read_mnist_images_parameters(images_file);
        nb_images_total = parameters[0];
        gree(parameters);

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
    if (!recover) {
        // Le nouveau TA calculé à partir du loss est majoré par 0.75*TA
        network = create_network_lenet5(LEARNING_RATE*0.75, 0, TANH, GLOROT, input_dim, input_depth);
        //network = create_simple_one(LEARNING_RATE*0.75, 0, RELU, GLOROT, input_dim, input_depth);
    } else {
        network = read_network(recover);
        network->learning_rate = LEARNING_RATE;
    }


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
    bool* thread_used = (bool*)malloc(sizeof(bool)*nb_threads);

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
        param->network = copy_network(network);
    }
    #else
    // Création des paramètres donnés à l'unique
    // thread dans l'hypothèse ou le multi-threading n'est pas utilisé.
    // Cela est utile à des fins de débogage notamment,
    // où l'utilisation de threads rend vite les choses plus compliquées qu'elles ne le sont.
    TrainParameters* train_params = (TrainParameters*)nalloc(sizeof(TrainParameters));

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
    end_time = omp_get_wtime();

    elapsed_time = end_time - start_time;
    printf("Taux d'apprentissage initial: %lf\n", network->learning_rate);
    printf("Initialisation: %0.2lf s\n\n", elapsed_time);

    for (int i=0; i < epochs; i++) {

        start_time = omp_get_wtime();
        // La variable accuracy permet d'avoir une ESTIMATION
        // du taux de réussite et de l'entraînement du réseau,
        // mais n'est en aucun cas une valeur réelle dans le cas
        // du multi-threading car chaque copie du réseau initiale sera légèrement différente
        // et donnera donc des résultats différents sur les mêmes images.
        accuracy = 0.;
        loss = 0.;
        knuth_shuffle(shuffle_index, nb_images_total);
        batches_epoques = div_up(nb_images_total, BATCHES);
        nb_images_total_remaining = nb_images_total;
        #ifndef USE_MULTITHREADING
            train_params->nb_images = BATCHES;
        #endif

        for (int j=0; j < batches_epoques; j++) {
            batch_loss = 0.;
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

                    if (train_parameters[k]->start+train_parameters[k]->nb_images >= nb_images_total) {
                        train_parameters[k]->nb_images = nb_images_total - train_parameters[k]->start -1;
                    }
                    if (train_parameters[k]->nb_images > 0) {
                        thread_used[k] = true;
                        copy_network_parameters(network, train_parameters[k]->network);
                        pthread_create( &tid[k], NULL, train_thread, (void*) train_parameters[k]);
                    } else {
                        thread_used[k] = false;
                    }
                }
                for (int k=0; k < nb_threads; k++) {
                    // On attend la terminaison de chaque thread un à un
                    if (thread_used[k]) {
                        pthread_join( tid[k], NULL );
                        accuracy += train_parameters[k]->accuracy / (float) nb_images_total;
                        loss += train_parameters[k]->loss/nb_images_total;
                        batch_loss += train_parameters[k]->loss/BATCHES;
                    }
                }

                // On attend que tous les fils aient fini avant d'appliquer des modifications au réseau principal
                for (int k=0; k < nb_threads; k++) {
                    if (train_parameters[k]->network) { // Si le fil a été utilisé
                        update_weights(network, train_parameters[k]->network);
                        update_bias(network, train_parameters[k]->network);
                    }
                }
                current_accuracy = accuracy * nb_images_total/((j+1)*BATCHES);
                printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " YELLOW "%0.2f%%" RESET, nb_threads, i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100);
                fflush(stdout);
            #else
                (void)nb_images_total_remaining; // Juste pour enlever un warning

                train_params->start = j*BATCHES;

                // Ne pas dépasser le nombre d'images à cause de la partie entière
                if (j == batches_epoques-1) {
                    train_params->nb_images = nb_images_total - j*BATCHES;
                }

                train_thread((void*)train_params);

                accuracy += train_params->accuracy / (float) nb_images_total;
                current_accuracy = accuracy * nb_images_total/((j+1)*BATCHES);
                loss += train_params->loss/nb_images_total;
                batch_loss += train_params->loss/BATCHES;

                update_weights(network, network);
                update_bias(network, network);

                printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "YELLOW"%0.4f%%"RESET, i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100);
                fflush(stdout);
            #endif
            // Il serait intéressant d'utiliser la perte calculée pour
            // savoir l'avancement dans l'apprentissage et donc comment adapter le taux d'apprentissage
            network->learning_rate = LEARNING_RATE*log(batch_loss+1);
        }
        end_time = omp_get_wtime();
        elapsed_time = end_time - start_time;
        #ifdef USE_MULTITHREADING
        printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " GREEN "%0.4f%%" RESET " \tTemps: %0.2f s\n", nb_threads, i, epochs, nb_images_total, nb_images_total, accuracy*100, elapsed_time);
        #else
        printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: "GREEN"%0.4f%%"RESET" \tTemps: %0.2f s\n", i, epochs, nb_images_total, nb_images_total, accuracy*100, elapsed_time);
        #endif
        write_network(out, network);
    }

    // To generate a new neural and compare performances with scripts/benchmark_binary.py
    if (epochs == 0) {
        write_network(out, network);
    }
    free(shuffle_index);
    free_network(network);

    #ifdef USE_MULTITHREADING
    free(tid);
    for (int i=0; i < nb_threads; i++) {
        free_network(train_parameters[i]->network);
    }
    free(train_parameters);
    #else
    free(train_params);
    #endif

    if (dataset_type == 0) {
        for (int i=0; i < nb_images_total; i++) {
            for (int j=0; j < 28; j++) {
                gree(images[i][j]);
            }
            gree(images[i]);
        }
        gree(images);
        gree(labels);
    } else {
        free_dataset(dataset);
    }

    end_time = omp_get_wtime();
    elapsed_time = end_time - algo_start;
    printf("\nTemps total: %0.1f s\n", elapsed_time);
}
