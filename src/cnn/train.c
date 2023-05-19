#include <sys/sysinfo.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../common/include/memory_management.h"
#include "../common/include/colors.h"
#include "../common/include/utils.h"
#include "../common/include/mnist.h"
#include "include/initialisation.h"
#include "include/test_network.h"
#include "include/neuron_io.h"
#include "include/function.h"
#include "include/update.h"
#include "include/models.h"
#include "include/utils.h"
#include "include/free.h"
#include "include/jpeg.h"
#include "include/cnn.h"

#include "include/train.h"

int div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void* load_image(void* parameters) {
    LoadImageParameters* param = (LoadImageParameters*)parameters;

    if (!param->dataset->images[param->index]) {
        imgRawImage* image = loadJpegImageFile(param->dataset->fileNames[param->index]);
        param->dataset->images[param->index] = image->lpData;
        free(image);
    } else {
        printf_warning((char*)"Image déjà chargée\n"); // Pas possible techniquement, donc on met un warning
    }

    return NULL;
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

    #ifdef DETAILED_TRAIN_TIMINGS
        double start_time;
    #endif

    pthread_t tid;
    LoadImageParameters* load_image_param = (LoadImageParameters*)malloc(sizeof(LoadImageParameters));
    if (dataset_type != 0) {
        load_image_param->dataset = param->dataset;
        load_image_param->index = index[start];

        pthread_create(&tid, NULL, load_image, (void*) load_image_param);
    }

    for (int i=start;  i < start+nb_images; i++) {
        if (dataset_type == 0) {
            write_image_in_network_32(images[index[i]], height, width, network->input[0][0], true);

            #ifdef DETAILED_TRAIN_TIMINGS
                start_time = omp_get_wtime();
            #endif

            forward_propagation(network);

            #ifdef DETAILED_TRAIN_TIMINGS
                printf("Temps de forward: ");
                printf_time(omp_get_wtime() - start_time);
                printf("\n");
                start_time = omp_get_wtime();
            #endif

            maxi = indice_max(network->input[network->size-1][0][0], 10);
            if (maxi == -1) {
                printf("\n");
                printf_error((char*)"Le réseau sature.\n");
                exit(1);
            }
            
            wanted_output = generate_wanted_output(labels[index[i]], 10);
            loss += compute_mean_squared_error(network->input[network->size-1][0][0], wanted_output, 10);
            gree(wanted_output, false);

            backward_propagation(network, labels[index[i]]);

            #ifdef DETAILED_TRAIN_TIMINGS
                printf("Temps de backward: ");
                printf_time(omp_get_wtime() - start_time);
                printf("\n");
                start_time = omp_get_wtime();
            #endif

            if (maxi == labels[index[i]]) {
                accuracy += 1.;
            }
        } else {
            pthread_join(tid, NULL);
            if (!param->dataset->images[index[i]]) {
                image = loadJpegImageFile(param->dataset->fileNames[index[i]]);
                param->dataset->images[index[i]] = image->lpData;
                free(image);
            }

            if (i != start+nb_images-1) {
                load_image_param->index = index[i+1];
                pthread_create(&tid, NULL, load_image, (void*) load_image_param);
            }
            write_256_image_in_network(param->dataset->images[index[i]], width, param->dataset->numComponents, network->width[0], network->input[0]);

            #ifdef DETAILED_TRAIN_TIMINGS
                start_time = omp_get_wtime();
            #endif

            forward_propagation(network);

            #ifdef DETAILED_TRAIN_TIMINGS
                printf("Temps de forward: ");
                printf_time(omp_get_wtime() - start_time);
                printf("\n");
                start_time = omp_get_wtime();
            #endif

            maxi = indice_max(network->input[network->size-1][0][0], param->dataset->numCategories);
            backward_propagation(network, param->dataset->labels[index[i]]);
            
            #ifdef DETAILED_TRAIN_TIMINGS
                printf("Temps de backward: ");
                printf_time(omp_get_wtime() - start_time);
                printf("\n");
                start_time = omp_get_wtime();
            #endif


            if (maxi == (int)param->dataset->labels[index[i]]) {
                accuracy += 1.;
            }

            free(param->dataset->images[index[i]]);
            param->dataset->images[index[i]] = NULL;
        }
    }

    free(load_image_param);

    param->accuracy = accuracy;
    param->loss = loss;
    return NULL;
}


void train(int dataset_type, char* images_file, char* labels_file, char* data_dir, int epochs, char* out, char* recover) {
    #ifdef USE_CUDA
    bool compatibility = cuda_setup(true);
    if (!compatibility) {
        printf("Exiting.\n");
        exit(1);
    }
    #endif
    srand(time(NULL));
    float loss;
    float batch_loss; // May be redundant with loss, but gives more informations
    float test_accuracy = 0.; // Used to decrease Learning rate
    (void)test_accuracy; // To avoid warnings when not used
    float accuracy;
    float batch_accuracy;
    float current_accuracy;


    //* Différents timers pour mesurer les performance en terme de vitesse
    double start_time, end_time;
    double elapsed_time;

    double algo_start = omp_get_wtime();

    start_time = omp_get_wtime();


    //* Chargement du dataset
    int input_width = -1;
    int input_depth = -1;

    int nb_images_total; // Images au total
    int nb_images_total_remaining; // Images restantes dans un batch
    int batches_epoques; // Batches par époque

    int*** images = NULL; // Images sous forme de tableau de tableaux de tableaux de pixels (degré de gris, MNIST)
    unsigned int* labels = NULL; // Labels associés aux images du dataset MNIST
    jpegDataset* dataset = NULL; // Structure de données décrivant un dataset d'images jpeg
    if (dataset_type == 0) { // Type MNIST
        // Chargement des images du set de données MNIST
        int* parameters = read_mnist_images_parameters(images_file);
        nb_images_total = parameters[0];
        free(parameters);

        images = read_mnist_images(images_file);
        labels = read_mnist_labels(labels_file);

        input_width = 32;
        input_depth = 1;
    } else { // Type JPG
        dataset = loadJpegDataset(data_dir);
        input_width = dataset->height + 4; // image_size + padding
        input_depth = dataset->numComponents;

        nb_images_total = dataset->numImages;
    }

    //* Création du réseau
    Network* network;
    if (!recover) {
        if (dataset_type == 0) {
            network = create_network_lenet5(LEARNING_RATE, 0, LEAKY_RELU, HE, input_width, input_depth);
            //network = create_simple_one(LEARNING_RATE, 0, RELU, GLOROT, input_width, input_depth);    
        } else {
            network = create_network_VGG16(LEARNING_RATE, 0, RELU, HE, dataset->numCategories);

            #ifdef USE_MULTITHREADING
                printf_warning("Utilisation de VGG16 avec multithreading. La quantité de RAM utilisée peut devenir excessive\n");
            #endif
        }
    } else {
        network = read_network(recover);
        network->learning_rate = LEARNING_RATE;
    }

    /*
       shuffle_index[i] contient le nouvel index de l'élément à l'emplacement i avant mélange
       Cela permet de réordonner le jeu d'apprentissage pour éviter certains biais
       qui pourraient provenir de l'ordre établi.
    */
    int* shuffle_index = (int*)malloc(sizeof(int)*nb_images_total);
    for (int i=0; i < nb_images_total; i++) {
        shuffle_index[i] = i;
    }


    //* Création des paramètres d'entrée de train_thread
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

    end_time = omp_get_wtime();

    elapsed_time = end_time - start_time;
    printf("Taux d'apprentissage initial: %0.2e\n", network->learning_rate);
    printf("Initialisation: ");
    printf_time(elapsed_time);
    printf("\n\n");

    //* Boucle d'apprentissage
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
            batch_accuracy = 0.;
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
                        batch_accuracy += train_parameters[k]->accuracy / (float) BATCHES; // C'est faux pour le dernier batch mais on ne l'affiche pas pour lui (enfin très rapidement)
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
                printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " YELLOW "%0.2f%%" RESET " \tBatch Accuracy: " YELLOW "%0.2f%%" RESET, nb_threads, i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100, batch_accuracy*100);
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
                batch_accuracy += train_params->accuracy / (float)BATCHES;
                loss += train_params->loss/nb_images_total;
                batch_loss += train_params->loss/BATCHES;

                update_weights(network, network);
                update_bias(network, network);

                printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " YELLOW "%0.4f%%" RESET "\tBatch Accuracy: " YELLOW "%0.2f%%" RESET, i, epochs, BATCHES*(j+1), nb_images_total, current_accuracy*100, batch_accuracy*100);
            #endif
        }
        //* Fin d'une époque: affichage des résultats et sauvegarde du réseau
        end_time = omp_get_wtime();
        elapsed_time = end_time - start_time;
        #ifdef USE_MULTITHREADING
        printf("\rThreads [%d]\tÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " GREEN "%0.4f%%" RESET " \tLoss: %lf\tTemps: ", nb_threads, i, epochs, nb_images_total, nb_images_total, accuracy*100, loss);
        printf_time(elapsed_time);
        printf("\n");
        #else
        printf("\rÉpoque [%d/%d]\tImage [%d/%d]\tAccuracy: " GREEN "%0.4f%%" RESET " \tLoss: %lf\tTemps: ", i, epochs, nb_images_total, nb_images_total, accuracy*100, loss);
        printf_time(elapsed_time);
        printf("\n");
        #endif
        write_network(out, network);
        // If you want to test the network between each epoch, uncomment the following lines:
        /*
        float* test_results = test_network(0, out, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", NULL, false, false, true);
        printf("Tests: Accuracy: %0.2lf%%\tLoss: %lf\n", test_results[0], test_results[1]);
        if (test_results[0] < test_accuracy) {
            network->learning_rate *= 0.1;
            printf("Decreased learning rate to %0.2e\n", network->learning_rate);
        }
        if (test_results[0] == test_accuracy) {
            network->learning_rate *= 2;
            printf("Increased learning rate to %0.2e\n", network->learning_rate);
        }
        test_accuracy = test_results[0];
        free(test_results);

        test_results = test_network(0, out, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", NULL, false, false, false);
        printf("Tests sans offset: Accuracy: %0.2lf%%\tLoss: %lf\n", test_results[0], test_results[1]);
        free(test_results);
        */
    }

    //* Fin de l'algo
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
                free(images[i][j]);
            }
            free(images[i]);
        }
        free(images);
        free(labels);
    } else {
        free_dataset(dataset);
    }

    end_time = omp_get_wtime();
    elapsed_time = end_time - algo_start;
    printf("\nTemps total: ");
    printf_time(elapsed_time);
    printf("\n");
}
