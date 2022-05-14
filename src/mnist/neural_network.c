#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "include/neuron.h"

// Définit le taux d'apprentissage du réseau neuronal, donc la rapidité d'adaptation du modèle (compris entre 0 et 1)
// Cette valeur peut évoluer au fur et à mesure des époques (linéaire c'est mieux)
#define LEARNING_RATE 0.5
//Retourne un nombre aléatoire entre 0 et 1
#define RAND_DOUBLE() ((double)rand())/((double)RAND_MAX)
//Coefficient leaking ReLU
#define COEFF_LEAKY_RELU 0.2
#define MAX_RESEAU 100000
#define INT_MIN -2147483648

#define PRINT_POIDS false
#define PRINT_BIAIS false


float max(float a, float b){
    return a < b ? b : a;
}

float sigmoid(float x){
    return 1/(1 + exp(-x));
}

float sigmoid_derivative(float x){
    float tmp = exp(-x);
    return tmp/((1+tmp)*(1+tmp));
}

float leaky_ReLU(float x){
    if (x > 0)
        return x;
    return COEFF_LEAKY_RELU;
}

float leaky_ReLU_derivative(float x){
    if (x > 0)
        return 1;
    return COEFF_LEAKY_RELU;
}

void network_creation(Network* network, int* neurons_per_layer, int nb_layers) {
    /* Créé et alloue de la mémoire aux différentes variables dans le réseau neuronal*/
    Layer* layer;

    network->nb_layers = nb_layers;
    network->layers = (Layer**)malloc(sizeof(Layer*)*nb_layers);

    for (int i=0; i < nb_layers; i++) {
        network->layers[i] = (Layer*)malloc(sizeof(Layer));
        layer = network->layers[i];
        layer->nb_neurons = neurons_per_layer[i]; // Nombre de neurones pour la couche
        layer->neurons = (Neuron**)malloc(sizeof(Neuron*)*network->layers[i]->nb_neurons); // Création des différents neurones dans la couche

        for (int j=0; j < layer->nb_neurons; j++) {
            layer->neurons[j] = (Neuron*)malloc(sizeof(Neuron));

            if (i != network->nb_layers-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
                layer->neurons[j]->weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);// Création des poids sortants du neurone
                layer->neurons[j]->back_weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
                layer->neurons[j]->last_back_weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
            }
        }
    }
}




void deletion_of_network(Network* network) {
    /* Libère l'espace mémoire alloué aux différentes variables dans la fonction
    'creation_du_network' */
    Layer* layer;
    Neuron* neuron;

    for (int i=0; i<network->nb_layers; i++) {
        layer = network->layers[i];
        if (i!=network->nb_layers-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
            for (int j=0; j<network->layers[i]->nb_neurons; j++) {
                neuron = layer->neurons[j];
                free(neuron->weights);
                free(neuron->back_weights);
            }
        }
        free(layer->neurons); // On libère enfin la liste des neurones de la couche
    }
    free(network->layers);
    free(network); // Pour finir, on libère le réseau neuronal contenant la liste des couches
}




void forward_propagation(Network* network) {
    /* Effectue une propagation en avant du réseau neuronal lorsque les données 
    on été insérées dans la première couche. Le résultat de la propagation se 
    trouve dans la dernière couche */
    Layer* layer; // Couche actuelle
    Layer* pre_layer; // Couche précédente
    Neuron* neuron;
    float sum;
    float max_z;

    for (int i=1; i < network->nb_layers; i++) { // La première couche contient déjà des valeurs
        sum = 0;
        max_z = INT_MIN;
        layer = network->layers[i];
        pre_layer = network->layers[i-1];

        for (int j=0; j < layer->nb_neurons; j++) {
            neuron = layer->neurons[j];
            neuron->z = neuron->bias;

            for (int k=0; k < pre_layer->nb_neurons; k++) {
                neuron->z += pre_layer->neurons[k]->z * pre_layer->neurons[k]->weights[j];
            }

            if (i < network->nb_layers-1) { // Pour toutes les couches sauf la dernière on utilise la fonction leaky_ReLU (a*z si z<0,  z sinon)
                neuron->z = leaky_ReLU(neuron->z);  
            } else { // Pour la dernière couche on utilise la fonction softmax
                max_z = max(max_z, neuron->z);
            }
        }
    }
    layer = network->layers[network->nb_layers-1];
    int size_last_layer = layer->nb_neurons;

    for (int j=0; j < size_last_layer; j++) {
        neuron = layer->neurons[j];
        neuron->z = exp(neuron->z - max_z);
        sum += neuron->z;
    }
    for (int j=0; j < size_last_layer; j++) {
        neuron = layer->neurons[j];
        neuron->z = neuron->z / sum;
    }
}




int* desired_output_creation(Network* network, int wanted_number) {
    /* Renvoie la liste des sorties voulues à partir du nombre
    de couches, de la liste du nombre de neurones par couche et de la
    position du résultat voulue, */
    int nb_neurons = network->layers[network->nb_layers-1]->nb_neurons;

    int* desired_output = (int*)malloc(sizeof(int)*nb_neurons);

    for (int i=0; i < nb_neurons; i++) // On initialise toutes les sorties à 0 par défaut
        desired_output[i] = 0;

    desired_output[wanted_number] = 1; // Seule la sortie voulue vaut 1
    return desired_output;
}



void backward_propagation(Network* network, int* desired_output) {
    /* Effectue une propagation en arrière du réseau neuronal */
    Neuron* neuron;
    Neuron* neuron2;
    float changes;
    float tmp;

    int i = network->nb_layers-2;
    int neurons_nb = network->layers[i+1]->nb_neurons;
    // On commence par parcourir tous les neurones de la couche finale
    for (int j=0; j < network->layers[i+1]->nb_neurons; j++) {
        neuron = network->layers[i+1]->neurons[j];
        tmp = (desired_output[j]==1) ? neuron->z - 1 : neuron->z;
        for (int k=0; k < network->layers[i]->nb_neurons; k++) {
            neuron2 = network->layers[i]->neurons[k];
            neuron2->back_weights[j] += neuron2->z*tmp;
            neuron2->last_back_weights[j] = neuron2->z*tmp;
        }
        neuron->last_back_bias = tmp;
        neuron->back_bias += tmp;
    }
    for (i--; i >= 0; i--) {
        neurons_nb =  network->layers[i+1]->nb_neurons;
        for (int j=0; j < neurons_nb; j++) {
            neuron = network->layers[i+1]->neurons[j];
            changes = 0;
            for (int k=0; k < network->layers[i+2]->nb_neurons; k++) {
                changes += (neuron->weights[k]*neuron->last_back_weights[k])/neurons_nb;
            }
            changes = changes*leaky_ReLU_derivative(neuron->z);
            neuron->back_bias += changes;
            neuron->last_back_bias = changes;
            for (int l=0; l < network->layers[i]->nb_neurons; l++){
                neuron2 = network->layers[i]->neurons[l];
                neuron2->back_weights[j] += neuron2->weights[j]*changes;
                neuron2->last_back_weights[j] = neuron2->weights[j]*changes;
            }
        }
    }
}




void network_modification(Network* network, uint32_t nb_modifs) {
    /* Modifie les poids et le biais des neurones du réseau neuronal à partir
    du nombre de couches et de la liste du nombre de neurone par couche */
    Neuron* neuron;

    for (int i=0; i < network->nb_layers; i++) { // on exclut la dernière couche
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            neuron = network->layers[i]->neurons[j];
            if (neuron->bias != 0 && PRINT_BIAIS)
                printf("C %d\tN %d\tb: %f      \tDb: %f\n", i, j, neuron->bias,  (LEARNING_RATE/nb_modifs) * neuron->back_bias);
            neuron->bias -= (LEARNING_RATE/nb_modifs) * neuron->back_bias; // On modifie le biais du neurone à partir des données de la propagation en arrière
            neuron->back_bias = 0;

            if (neuron->bias > MAX_RESEAU)
                neuron->bias = MAX_RESEAU;
            else if (neuron->bias < -MAX_RESEAU)
                neuron->bias = -MAX_RESEAU;

            if (i != network->nb_layers-1) {
                for (int k=0; k < network->layers[i+1]->nb_neurons; k++) {
                    if (neuron->weights[k] != 0 && PRINT_POIDS)
                        printf("C %d\tN %d -> %d\tp: %f  \tDp: %f\n", i, j, k, neuron->weights[k],  (LEARNING_RATE/nb_modifs) * neuron->back_weights[k]);
                    neuron->weights[k] -= (LEARNING_RATE/nb_modifs) * neuron->back_weights[k]; // On modifie le poids du neurone à partir des données de la propagation en arrière
                    neuron->back_weights[k] = 0;

                    if (neuron->weights[k] > MAX_RESEAU) {
                        neuron->weights[k] = MAX_RESEAU;
                        printf("Erreur, max du réseau atteint");
                    }
                    else if (neuron->weights[k] < -MAX_RESEAU) {
                        neuron->weights[k] = -MAX_RESEAU;
                        printf("Erreur, min du réseau atteint");
                    }
                }
            }
        }
    }
}




void network_initialisation(Network* network) {
    /* Initialise les variables du réseau neuronal (bias, poids, ...)
    en suivant de la méthode de Xavier ...... à partir du nombre de couches et de la liste du nombre de neurone par couches */
    Neuron* neuron;
    double upper_bound;
    double lower_bound;
    double bound_gap;

    int nb_layers_loop = network->nb_layers -1;

    upper_bound = 1/sqrt((double)network->layers[nb_layers_loop]->nb_neurons);
    lower_bound = -upper_bound;
    bound_gap = upper_bound - lower_bound;
    
    srand(time(0));
    for (int i=0; i < nb_layers_loop; i++) { // On exclut la dernière couche
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {

            neuron = network->layers[i]->neurons[j];
            // Initialisation des bornes supérieure et inférieure

            if (i!=nb_layers_loop) {
                for (int k=0; k < network->layers[i+1]->nb_neurons; k++) {
                    neuron->weights[k] = lower_bound + RAND_DOUBLE()*bound_gap;
                    neuron->back_weights[k] = 0;
                    neuron->last_back_weights[k] = 0;
                }
            }
            if (i > 0) {// Pour tous les neurones n'étant pas dans la première couche
                neuron->bias = lower_bound + RAND_DOUBLE()*bound_gap;
                neuron->back_bias = 0;
                neuron->last_back_bias = 0;
            }
        }
    }
}

void patch_network(Network* network, Network* delta, uint32_t nb_modifs) {
    // Les deux réseaux donnés sont supposés de même dimensions
    Neuron* neuron;
    Neuron* dneuron;

    for (int i=0; i < network->nb_layers; i++) {
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            neuron = network->layers[i]->neurons[j];
            dneuron = delta->layers[i]->neurons[j];
            neuron->bias -= (LEARNING_RATE/nb_modifs) * dneuron->back_bias;
            dneuron->back_bias = 0;

            if (i != network->nb_layers-1) {
                for (int k=0; k < network->layers[i+1]->nb_neurons; k++) {
                    neuron->weights[k] -= (LEARNING_RATE/nb_modifs) * dneuron->back_weights[k]; // On modifie le poids du neurone à partir des données de la propagation en arrière
                    dneuron->back_weights[k] = 0;
                }
            }
        }
    }
}

Network* copy_network(Network* network) {
    // Renvoie une copie modifiable d'un réseau de neurones
    Network* network2 = (Network*)malloc(sizeof(Network));
    Layer* layer;
    Neuron* neuron1;
    Neuron* neuron;

    network2->nb_layers = network->nb_layers;
    network2->layers = (Layer**)malloc(sizeof(Layer*)*network->nb_layers);
    for (int i=0; i < network2->nb_layers; i++) {
        layer = (Layer*)malloc(sizeof(Layer));
        layer->nb_neurons = network->layers[i]->nb_neurons;
        layer->neurons = (Neuron**)malloc(sizeof(Neuron*)*layer->nb_neurons);
        for (int j=0; j < layer->nb_neurons; j++) {
            neuron = (Neuron*)malloc(sizeof(Neuron));

            neuron1 = network->layers[i]->neurons[j];
            neuron->bias = neuron1->bias;
            neuron->z = neuron1->z;
            neuron->back_bias = neuron1->back_bias;
            neuron->last_back_bias = neuron1->last_back_bias;
            if (i != network2->nb_layers-1) {
                (void)network2->layers[i+1]->nb_neurons;
                neuron->weights = (float*)malloc(sizeof(float)*network->layers[i+1]->nb_neurons);
                neuron->back_weights = (float*)malloc(sizeof(float)*network->layers[i+1]->nb_neurons);
                neuron->last_back_weights = (float*)malloc(sizeof(float)*network->layers[i+1]->nb_neurons);
                for (int k=0; k < network->layers[i+1]->nb_neurons; k++) {
                    neuron->weights[k] = neuron1->weights[k];
                    neuron->back_weights[k] = neuron1->back_weights[k];
                    neuron->last_back_weights[k] = neuron1->last_back_weights[k];
                }
            }
            layer->neurons[j] = neuron;
        }
    network2->layers[i] = layer;
    }
    return network2;
}


float loss_computing(Network* network, int numero_voulu){
    /* Renvoie l'erreur du réseau neuronal pour une sortie */
    float erreur = 0;
    float neuron_value;

    for (int i=0; i < network->nb_layers-1; i++) {
        neuron_value = network->layers[network->nb_layers-1]->neurons[i]->z;

        if (i == numero_voulu) {
            erreur += (1-neuron_value)*(1-neuron_value);
        }
        else {
            erreur += neuron_value*neuron_value;
        }
    }
    
    return erreur;
}
