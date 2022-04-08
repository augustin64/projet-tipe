#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "struct/neuron.h"

#define TAUX_APPRENTISSAGE 0.15 // Définit le taux d'apprentissage du réseau neuronal, donc la rapidité d'adaptation du modèle (compris entre 0 et 1)


float max(float a, float b){
    return a<b?b:a;
}

float sigmoid(float x){
    return 1/(1 + exp(x));
}

float sigmoid_derivee(float x){
    float tmp = exp(-x);
    return tmp/((1 + tmp)*(1+tmp));
}

float ReLU(float x){
    return max(x, 0);
}

void creation_du_reseau_neuronal(Reseau* reseau_neuronal, int* neurones_par_couche, int nb_couches) {
    /* Créé et alloue de la mémoire aux différentes variables dans le réseau neuronal*/

    reseau_neuronal->nb_couches = nb_couches;
    reseau_neuronal->couches = (Couche**)malloc(sizeof(Couche*)*nb_couches); // 

    for (int i=0; i < nb_couches; i++) {

        reseau_neuronal->couches[i] = (Couche*)malloc(sizeof(Couche)); // Utilité ? 
        reseau_neuronal->couches[i]->nb_neurones = neurones_par_couche[i]; // nombre de neurones pour la couche
        reseau_neuronal->couches[i]->neurones = (Neurone**)malloc(sizeof(Neurone*)*reseau_neuronal->couches[i]->nb_neurones); // Création des différents neurones dans la couche
        for (int j=0; j < reseau_neuronal->couches[i]->nb_neurones; j++) {
            reseau_neuronal->couches[i]->neurones[j] = (Neurone*)malloc(sizeof(Neurone));
            if (i!=reseau_neuronal->nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
                reseau_neuronal->couches[i]->neurones[j]->poids_sortants = (float*)malloc(sizeof(float)*neurones_par_couche[i+1]);// Création des poids sortants du neurone
                reseau_neuronal->couches[i]->neurones[j]->d_poids_sortants = (float*)malloc(sizeof(float)*neurones_par_couche[i+1]);
            }
        }
    }
}




void suppression_du_reseau_neuronal(Reseau* reseau_neuronal) {
    /* Libère l'espace mémoire alloué aux différentes variables dans la fonction
    'creation_du_reseau_neuronal' */

    for (int i=0; i<reseau_neuronal->nb_couches; i++) {
        if (i!=reseau_neuronal->nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
            for (int j=0; j<reseau_neuronal->couches[i]->nb_neurones; j++) {
                free(reseau_neuronal->couches[i]->neurones[j]->poids_sortants);
                free(reseau_neuronal->couches[i]->neurones[j]->d_poids_sortants);
            }
        }
        free(reseau_neuronal->couches[i]->neurones); // On libère enfin la liste des neurones de la couche
    }
    free(reseau_neuronal); // Pour finir, on libère le réseau neronal contenant la liste des couches
}




void forward_propagation(Reseau* reseau_neuronal) {
    /* Effectue une propagation en avant du réseau neuronal lorsque les données 
    on été insérées dans la première couche. Le résultat de la propagation se 
    trouve dans la dernière couche*/

    for (int i=1; i<reseau_neuronal->nb_couches; i++) { // La première couche contient déjà des valeurs
        for (int j=0; j<reseau_neuronal->couches[i]->nb_neurones; j++) {
        
            reseau_neuronal->couches[i]->neurones[j]->z = reseau_neuronal->couches[i]->neurones[j]->biais;
            for (int k=0; k<reseau_neuronal->couches[i-1]->nb_neurones; k++) {
                reseau_neuronal->couches[i]->neurones[j]->z += reseau_neuronal->couches[i-1]->neurones[k]->activation * reseau_neuronal->couches[i-1]->neurones[k]->z * reseau_neuronal->couches[i-1]->neurones[k]->poids_sortants[i]; // ???
            }

            if (i<reseau_neuronal->nb_couches-1) { // Pour toutes les couches sauf la dernière on utilise la fonction ReLU (0 si z<0,  z sinon)
                reseau_neuronal->couches[i]->neurones[j]->z=ReLU(reseau_neuronal->couches[i]->neurones[j]->z);
                    
            }
            else{ // Pour la dernière couche on utilise la fonction sigmoid permettant d'obtenir un résultat entre 0 et 1 à savoir une probabilité
                reseau_neuronal->couches[i]->neurones[j]->z = sigmoid(reseau_neuronal->couches[i]->neurones[j]->z);
            }
        }
    }
}




int* creation_de_la_sortie_voulue(Reseau* reseau_neuronal, int pos_nombre_voulu) {
    /* Renvoie la liste des sorties voulues à partir du nombre
    de couches, de la liste du nombre de neurones par couche et de la
    position du résultat voulue, */
    int nb_neurones = reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->nb_neurones;

    int* sortie_voulue = (int*)malloc(sizeof(int)*nb_neurones);

    for (int i=0; i < nb_neurones; i++) // On initialise toutes les sorties à 0 par défault
        sortie_voulue[i] = 0;

    sortie_voulue[pos_nombre_voulu] = 1; // Seule la sortie voulue vaut 1
    return sortie_voulue;
}




void backward_propagation(Reseau* reseau_neuronal, int* sortie_voulue) {
    /* Effectue une propagation en arrière du réseau neuronal */

    // On commence par parcourir tous les neurones de la couche finale
    for (int i=0; i<reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->nb_neurones; i++) {
        // On calcule l'erreur de la sortie 
        reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->d_z = (reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->z - sortie_voulue[i])*(reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->z - sortie_voulue[i]);

        for(int k=0; k<reseau_neuronal->couches[reseau_neuronal->nb_couches-2]->nb_neurones; k++) { // Pour chaque neurone de l'avant dernière couche
            reseau_neuronal->couches[reseau_neuronal->nb_couches-2]->neurones[k]->d_poids_sortants[i] = (reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->d_z * reseau_neuronal->couches[reseau_neuronal->nb_couches-2]->neurones[k]->activation);
            reseau_neuronal->couches[reseau_neuronal->nb_couches-2]->neurones[k]->d_activation = reseau_neuronal->couches[reseau_neuronal->nb_couches-2]->neurones[k]->poids_sortants[i] * reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->d_z;
        }
        // ???
        reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->d_biais = reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[i]->d_z;
    }

    for(int i=reseau_neuronal->nb_couches-2; i>0; i--) { // On remonte les couche de l'avant dernière jusqu'à la première
        for(int j=0; j<reseau_neuronal->couches[i]->nb_neurones; j++) {
            if(reseau_neuronal->couches[i]->neurones[j]->z >= 0) // ??? ...
                reseau_neuronal->couches[i]->neurones[j]->d_z = reseau_neuronal->couches[i]->neurones[j]->d_activation;
            else // ??? ...
                reseau_neuronal->couches[i]->neurones[j]->d_z = 0;

            for(int k=0; k<reseau_neuronal->couches[i-1]->nb_neurones; k++) {
                reseau_neuronal->couches[i-1]->neurones[k]->d_poids_sortants[j] = reseau_neuronal->couches[i]->neurones[j]->d_z * reseau_neuronal->couches[i-1]->neurones[k]->activation;
                if(i>1) // ??? ...
                    reseau_neuronal->couches[i-1]->neurones[k]->d_activation = reseau_neuronal->couches[i-1]->neurones[k]->poids_sortants[j] * reseau_neuronal->couches[i]->neurones[j]->d_z;
            }
            reseau_neuronal->couches[i]->neurones[j]->d_biais = reseau_neuronal->couches[i]->neurones[j]->d_z; // ??? ...
        }
    }

}




void modification_du_reseau_neuronal(Reseau* reseau_neuronal) {
    /* Modifie les poids et le biais des neurones du réseau neuronal à partir
    du nombre de couches et de la liste du nombre de neurone par couche */

    for (int i=0; i<reseau_neuronal->nb_couches-1; i++) { // on exclut la dernière couche
        for (int j=0; i<reseau_neuronal->couches[i]->nb_neurones; j++) {
            reseau_neuronal->couches[i]->neurones[j]->biais = reseau_neuronal->couches[i]->neurones[j]->biais - (TAUX_APPRENTISSAGE * reseau_neuronal->couches[i]->neurones[j]->d_biais); // On modifie le biais du neurone à partir des données de la propagation en arrière
            for (int k=0; k<reseau_neuronal->couches[i+1]->nb_neurones; k++) {
                reseau_neuronal->couches[i]->neurones[j]->poids_sortants[k] = reseau_neuronal->couches[i]->neurones[j]->poids_sortants[k] - (TAUX_APPRENTISSAGE * reseau_neuronal->couches[i]->neurones[j]->d_poids_sortants[k]); // On modifie le poids du neurone à partir des données de la propagation en arrière
            }
        }
    }
}




void initialisation_du_reseau_neuronal(Reseau* reseau_neuronal) {
    /* Initialise les variables du réseau neuronal (activation, biais, poids, ...)
    en suivant de la méthode de Xavier ...... à partir du nombre de couches et de la liste du nombre de neurone par couche */
    srand(time(0));
    for (int i=0; i<reseau_neuronal->nb_couches-1; i++) { // On exclut la dernière couche
        for (int j=0; j<reseau_neuronal->couches[i]->nb_neurones-1; j++) {
            // Initialisation des bornes supérieure et inférieure
            double borne_superieure = 1/sqrt(reseau_neuronal->couches[i]->nb_neurones);
            double borne_inferieure = - borne_superieure;
            reseau_neuronal->couches[i]->neurones[j]->activation = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure);
            for (int k=0; k<reseau_neuronal->couches[i+1]->nb_neurones-1; k++) { // Pour chaque neurone de la couche suivante auquel le neurone est relié
                reseau_neuronal->couches[i]->neurones[j]->poids_sortants[k] = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // Initialisation des poids sortants aléatoirement
                reseau_neuronal->couches[i]->neurones[j]->d_poids_sortants[k] = 0.0; // ... ???
            }
            if (i > 0) {// Pour tous les neurones n'étant pas dans la première couche
                reseau_neuronal->couches[i]->neurones[j]->biais = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // On initialise le biais aléatoirement
            }
        }
    }
    double borne_superieure = 1/sqrt(reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->nb_neurones);
    double borne_inferieure = - borne_superieure;
    for (int j=0; j < reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->nb_neurones; j++) {// Intialisation de la dernière couche exclut ci-dessus
        reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[j]->activation = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure);
        reseau_neuronal->couches[reseau_neuronal->nb_couches-1]->neurones[j]->biais = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // On initialise le biais aléatoirement
    }
}
