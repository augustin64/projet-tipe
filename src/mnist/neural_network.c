#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "struct/neuron.h"

// Définit le taux d'apprentissage du réseau neuronal, donc la rapidité d'adaptation du modèle (compris entre 0 et 1)
//Cette valeur peut évoluer au fur et à mesure des époques (linéaire c'est mieux)
#define TAUX_APPRENTISSAGE 0.15
//Retourne un nombre aléatoire entre 0 et 1
#define RAND_DOUBLE() ((double)rand())/((double)RAND_MAX)
//Coefficient leaking ReLU
#define COEFF_LEAKY_RELU 0.2
#define MAX_RESEAU 10


float max(float a, float b){
    return a<b?b:a;
}

float sigmoid(float x){
    return 1/(1 + exp(x));
}

float sigmoid_derivee(float x){
    float tmp = sigmoid(x);
    return tmp*(1-tmp);
}

float leaky_ReLU(float x){
    if (x>0)
        return x;
    return x*COEFF_LEAKY_RELU;
}

float leaky_ReLU_derivee(float x){
    if (x>0)
        return 1;
    return COEFF_LEAKY_RELU;
}

void creation_du_reseau_neuronal(Reseau* reseau, int* neurones_par_couche, int nb_couches) {
    /* Créé et alloue de la mémoire aux différentes variables dans le réseau neuronal*/
    Couche* couche;

    reseau->nb_couches = nb_couches;
    reseau->couches = (Couche**)malloc(sizeof(Couche*)*nb_couches);

    for (int i=0; i < nb_couches; i++) {
        reseau->couches[i] = (Couche*)malloc(sizeof(Couche));
        couche = reseau->couches[i];
        couche->nb_neurones = neurones_par_couche[i]; // nombre de neurones pour la couche
        couche->neurones = (Neurone**)malloc(sizeof(Neurone*)*reseau->couches[i]->nb_neurones); // Création des différents neurones dans la couche

        for (int j=0; j < couche->nb_neurones; j++) {
            couche->neurones[j] = (Neurone*)malloc(sizeof(Neurone));

            if (i != reseau->nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
                couche->neurones[j]->poids_sortants = (float*)malloc(sizeof(float)*neurones_par_couche[i+1]);// Création des poids sortants du neurone
                couche->neurones[j]->d_poids_sortants = (float*)malloc(sizeof(float)*neurones_par_couche[i+1]);
            }
        }
    }
}




void suppression_du_reseau_neuronal(Reseau* reseau) {
    /* Libère l'espace mémoire alloué aux différentes variables dans la fonction
    'creation_du_reseau' */

    for (int i=0; i<reseau->nb_couches; i++) {
        if (i!=reseau->nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
            for (int j=0; j<reseau->couches[i]->nb_neurones; j++) {
                free(reseau->couches[i]->neurones[j]->poids_sortants);
                free(reseau->couches[i]->neurones[j]->d_poids_sortants);
            }
        }
        free(reseau->couches[i]->neurones); // On libère enfin la liste des neurones de la couche
    }
    free(reseau); // Pour finir, on libère le réseau neronal contenant la liste des couches
}




void forward_propagation(Reseau* reseau) {
    /* Effectue une propagation en avant du réseau neuronal lorsque les données 
    on été insérées dans la première couche. Le résultat de la propagation se 
    trouve dans la dernière couche */
    Couche* couche; // Couche actuelle
    Couche* pre_couche; // Couche précédente

    for (int i=1; i < reseau->nb_couches; i++) { // La première couche contient déjà des valeurs
        couche = reseau->couches[i];
        pre_couche = reseau->couches[i-1];

        for (int j=0; j < couche->nb_neurones; j++) {
            couche->neurones[j]->z = sigmoid(couche->neurones[j]->biais)-0.5;

            for (int k=0; k < pre_couche->nb_neurones; k++) {
                couche->neurones[j]->z += pre_couche->neurones[k]->z * pre_couche->neurones[k]->poids_sortants[j];
            }

            if (i < reseau->nb_couches-1) { // Pour toutes les couches sauf la dernière on utilise la fonction leaky_ReLU (a*z si z<0,  z sinon)
                couche->neurones[j]->z = leaky_ReLU(couche->neurones[j]->z);  
            }
            else { // Pour la dernière couche on utilise la fonction sigmoid permettant d'obtenir un résultat entre 0 et 1 à savoir une probabilité
                couche->neurones[j]->z = sigmoid(couche->neurones[j]->z);
            }
        }
    }
}




int* creation_de_la_sortie_voulue(Reseau* reseau, int pos_nombre_voulu) {
    /* Renvoie la liste des sorties voulues à partir du nombre
    de couches, de la liste du nombre de neurones par couche et de la
    position du résultat voulue, */
    int nb_neurones = reseau->couches[reseau->nb_couches-1]->nb_neurones;

    int* sortie_voulue = (int*)malloc(sizeof(int)*nb_neurones);

    for (int i=0; i < nb_neurones; i++) // On initialise toutes les sorties à 0 par défault
        sortie_voulue[i] = 0;

    sortie_voulue[pos_nombre_voulu] = 1; // Seule la sortie voulue vaut 1
    return sortie_voulue;
}



void backward_propagation(Reseau* reseau, int* sortie_voulue) {
    /* Effectue une propagation en arrière du réseau neuronal */
    Neurone* neurone;
    Neurone* neurone2;
    float changes;

    // On commence par parcourir tous les neurones de la couche finale
    for (int i=reseau->nb_couches-2; i>=0; i--) {
        if (i==reseau->nb_couches-2){
            for (int j=0; j<reseau->couches[i]->nb_neurones; j++) {
                neurone = reseau->couches[reseau->nb_couches-1]->neurones[i];
                changes = sigmoid_derivee(neurone->z)*2*(neurone->z - sortie_voulue[i]);
                //neurone->biais = neurone->biais - TAUX_APPRENTISSAGE*changes;
                for (int k=0; k<reseau->couches[i+1]->nb_neurones; k++) {
                    reseau->couches[i]->neurones[j]->d_poids_sortants[k] += reseau->couches[i-1]->neurones[k]->poids_sortants[j]*changes;
                }
            }
        }
        else {
            for (int j=0; j<reseau->couches[i+1]->nb_neurones; j++) {
                float changes = 0;
                for (int k=0; k<reseau->couches[i+2]->nb_neurones; k++) {
                    changes += reseau->couches[i+1]->neurones[j]->poids_sortants[k]*reseau->couches[i+1]->neurones[j]->d_poids_sortants[k];
                }
                changes = changes*leaky_ReLU_derivee(reseau->couches[i+1]->neurones[j]->z);
                reseau->couches[i+1]->neurones[j]->d_biais += changes;
                for (int k=0; k<reseau->couches[i]->nb_neurones; k++){
                    reseau->couches[i]->neurones[k]->d_poids_sortants[j] += reseau->couches[i]->neurones[k]->poids_sortants[j]*changes;
                }
            }
        }
    }
    //mise_a_jour_parametres(reseau);
}




void modification_du_reseau_neuronal(Reseau* reseau) {
    /* Modifie les poids et le biais des neurones du réseau neuronal à partir
    du nombre de couches et de la liste du nombre de neurone par couche */
    Neurone* neurone;

    for (int i=0; i < reseau->nb_couches-1; i++) { // on exclut la dernière couche
        for (int j=0; j < reseau->couches[i]->nb_neurones; j++) {
            neurone = reseau->couches[i]->neurones[j];
            neurone->biais = neurone->biais + TAUX_APPRENTISSAGE * neurone->d_biais; // On modifie le biais du neurone à partir des données de la propagation en arrière
            neurone->d_biais = 0;
            if (neurone->biais > MAX_RESEAU)
                neurone->biais = MAX_RESEAU;
            else if (neurone->biais < -MAX_RESEAU)
                neurone->biais = -MAX_RESEAU;

            for (int k=0; k < reseau->couches[i+1]->nb_neurones; k++) {
                neurone->poids_sortants[k] = neurone->poids_sortants[k] - (TAUX_APPRENTISSAGE * neurone->d_poids_sortants[k]); // On modifie le poids du neurone à partir des données de la propagation en arrière
                neurone->d_poids_sortants[k] = 0;
                if (neurone->poids_sortants[k] > MAX_RESEAU)
                    neurone->poids_sortants[k] = MAX_RESEAU;
                else if (neurone->poids_sortants[k] < -MAX_RESEAU)
                    neurone->poids_sortants[k] = -MAX_RESEAU;
            }
        }
    }
}




void initialisation_du_reseau_neuronal(Reseau* reseau) {
    /* Initialise les variables du réseau neuronal (activation, biais, poids, ...)
    en suivant de la méthode de Xavier ...... à partir du nombre de couches et de la liste du nombre de neurone par couche */
    Neurone* neurone;
    double borne_superieure;
    double borne_inferieure;
    double ecart_bornes;

    srand(time(0));
    for (int i=0; i < reseau->nb_couches-1; i++) { // On exclut la dernière couche
        for (int j=0; j < reseau->couches[i]->nb_neurones-1; j++) {

            neurone = reseau->couches[i]->neurones[j];
            // Initialisation des bornes supérieure et inférieure
            borne_superieure = 1/sqrt((double)reseau->couches[reseau->nb_couches-1]->nb_neurones);
            borne_inferieure = -borne_superieure;
            ecart_bornes = borne_superieure - borne_inferieure;

            neurone->activation = borne_inferieure + RAND_DOUBLE()*ecart_bornes;

            for (int k=0; k < reseau->couches[i+1]->nb_neurones-1; k++) { // Pour chaque neurone de la couche suivante auquel le neurone est relié
                neurone->poids_sortants[k] = borne_inferieure + RAND_DOUBLE()*ecart_bornes; // Initialisation des poids sortants aléatoirement
            }
            if (i > 0) {// Pour tous les neurones n'étant pas dans la première couche
                neurone->biais = borne_inferieure + RAND_DOUBLE()*ecart_bornes; // On initialise le biais aléatoirement
            }
        }
    }
    borne_superieure = 1/sqrt((double)reseau->couches[reseau->nb_couches-1]->nb_neurones);
    borne_inferieure = -borne_superieure;
    ecart_bornes = borne_superieure - borne_inferieure;

    for (int j=0; j < reseau->couches[reseau->nb_couches-1]->nb_neurones; j++) {// Intialisation de la dernière couche exclue ci-dessus
        neurone = reseau->couches[reseau->nb_couches-1]->neurones[j];
        //Il y a pas de biais et activation variables pour la dernière couche
        neurone->activation = 1;
        neurone->biais = 0;
    }
}




float erreur_sortie(Reseau* reseau, int numero_voulu){
    /* Renvoie l'erreur du réseau neuronal pour une sortie */
    float erreur = 0;
    float neurone_value;

    for (int i=0; i < reseau->nb_couches-1; i++) {
        neurone_value = reseau->couches[reseau->nb_couches-1]->neurones[i]->z;

        if (i==numero_voulu) {
            erreur += (1-neurone_value)*(1-neurone_value);
        }
        else {
            erreur += neurone_value*neurone_value;
        }
    }
    
    return erreur;
}