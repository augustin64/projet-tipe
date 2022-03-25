#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define TAUX_APPRENTISSAGE 0.15 // Définit le taux d'apprentissage du réseau neuronal, donc la rapidité d'adaptation du modèle (compris entre 0 et 1)

//<> Le nombre de couches doit être supérieur à 2

/*---------------------------------------------------
----------------------Structures---------------------
---------------------------------------------------*/
typedef struct neurone_struct{
    float activation; // Caractérise l'activation du neurone
    float* poids_sortants; // Liste de tous les poids des arêtes sortants du neurone
    float biais; // Caractérise le biais du neurone
    float z; // Sauvegarde des calculs faits sur le neurone (programmation dynamique)

    float dactivation;
    float *dw;
    float dbiais;
    float dz;
} neurone_struct;


typedef struct couche_struct{
    int nb_neurone; // Nombre de neurones dans la couche (longueur de la liste ci-dessous)
    neurone_struct* neurone; // Liste des neurones dans la couche
} couche_struct;


/*---------------------------------------------------
----------------------Fonctions----------------------
---------------------------------------------------*/

couche_struct* reseau_neuronal;

void creation_du_reseau_neuronal(int nb_couches, int* neurones_par_couche);
void suppression_du_reseau_neuronal(int nb_couches, int* neurones_par_couche);
void forward_propagation(int nb_couches, int* neurones_par_couche);
int* creation_de_la_sortie_voulue(int nb_couches, int* neurones_par_couche, int pos_nombre_voulu);
void backward_propagation(int nb_couches, int* neurones_par_couche, int* sortie_voulue);
void modification_du_reseau_neuronal(int nb_couches, int* neurones_par_couche);
void initialisation_du_reseau_neuronal(int nb_couches, int* neurones_par_couche);



void creation_du_reseau_neuronal(int nb_couches, int* neurones_par_couche) {
    /* Créé les différentes variables dans la variable du réseau neuronal à
    partir du nombre de couches et de la liste du nombre de neurones par couche */

    reseau_neuronal = (couche_struct*)malloc(sizeof(couche_struct)*nb_couches); // Création des différentes couches
    for (int i=0; i<nb_couches; i++) {

        reseau_neuronal[i].nb_neurone = neurones_par_couche[i]; // nombre de neurones pour la couche
        reseau_neuronal[i].neurone = (neurone_struct*)malloc(sizeof(neurone_struct)*neurones_par_couche[i]); // Création des différents neurones dans la couche

        if (i!=nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
            for (int j=0; j<neurones_par_couche[i]; j++) {
                reseau_neuronal[i].neurone[j].poids_sortants = (float*)malloc(sizeof(float)*neurones_par_couche[i+1]) ;// Création des poids sortants du neurone
            }
        }
    }
}




void suppression_du_reseau_neuronal(int nb_couches, int* neurones_par_couche) {
    /* Libère l'espace mémoire alloué aux différentes variables dans la fonction
    'creation_du_reseau_neuronal' à partir du nombre de couche et de la liste du 
    nombre de neurone par couche */

    for (int i=0; i<nb_couches; i++) {
        if (i!=nb_couches-1) { // On exclut la dernière couche dont les neurones ne contiennent pas de poids sortants
            for (int j=0; j<neurones_par_couche[i]; j++) {
                free(reseau_neuronal[i].neurone[j].poids_sortants); // On libère la variables des poids sortants
            }
        }
        free(reseau_neuronal[i].neurone); // On libère enfin la liste des neurones de la couche
    }
    free(reseau_neuronal); // Pour finir, on libère le réseau neronal contenant la liste des couches
}




void forward_propagation(int nb_couches, int* neurones_par_couche) {
    /* Effectue une propagation en avant du réseau neuronal à partir du nombre
    de couches et de la liste du nombre de neurones par couche */

    for (int i=1; i<nb_couches; i++) { // La première couche contient déjà des valeurs
        for (int j=0; j<neurones_par_couche[i];j++) { // Pour chaque neurone de la couche
        
            reseau_neuronal[i].neurone[j].z = reseau_neuronal[i].neurone[j].biais; // On réinitialise l'utilisation actuelle du neurone à son biais
            for (int k=0; k<neurones_par_couche[i-1]; k++) {
                reseau_neuronal[i].neurone[j].z += reseau_neuronal[i-1].neurone[k].activation * reseau_neuronal[i-1].neurone[k].z * reseau_neuronal[i-1].neurone[k].poids_sortants[i]; // ???
            }

            if (i<nb_couches-1) { // Pour toutes les couches sauf la dernière on utilise la fonction relu
                if (reseau_neuronal[i].neurone[j].z < 0)
                    reseau_neuronal[i].neurone[j].activation = 0;
                else
                    reseau_neuronal[i].neurone[j].activation = reseau_neuronal[i].neurone[j].z;
            }
            else{ // Pour la dernière couche on utilise la fonction sigmoid permettant d'obtenir un résultat entre 0 et 1 étant une probabilité
                reseau_neuronal[i].neurone[j].activation = 1/(1 + exp(reseau_neuronal[i].neurone[j].activation));
            }
        }
    }
}




int* creation_de_la_sortie_voulue(int nb_couches, int* neurones_par_couche, int pos_nombre_voulu) {
    /* Renvoie la liste des sorties voulues à partir du nombre
    de couches, de la liste du nombre de neurones par couche et de la
    position du résultat voulue, */

    int* sortie_voulue = (int*)malloc(sizeof(int));
    for (int i=0; i<neurones_par_couche[nb_couches-1]; i++) // On initialise toutes les sorties à 0 par défault
        sortie_voulue[i]=0;
    sortie_voulue[pos_nombre_voulu]=1; // Seule la sortie voulue vaut 1
    return sortie_voulue;
}




void backward_propagation(int nb_couches, int* neurones_par_couche, int* sortie_voulue) {
    /* Effectue une propagation en arrière du réseau neuronal à partir du
    nombre de couches, de la liste du nombre de neurone par couche  et de 
    la liste des sorties voulues*/

    // On commence par parcourir tous les neurones de la couche finale
    for (int i=0; i<neurones_par_couche[nb_couches-1]; i++) {
        // On applique la formule de propagation en arrière 
        reseau_neuronal[nb_couches-1].neurone[i].dz = (reseau_neuronal[nb_couches-1].neurone[i].activation - sortie_voulue[i]) * (reseau_neuronal[nb_couches-1].neurone[i].activation) * (1- reseau_neuronal[nb_couches-1].neurone[i].activation);

        for(int k=0; k<neurones_par_couche[nb_couches-2]; k++) { // Pour chaque neurone de l'avant dernière couche
            reseau_neuronal[nb_couches-2].neurone[k].dw[i] = (reseau_neuronal[nb_couches-1].neurone[i].dz * reseau_neuronal[nb_couches-2].neurone[k].activation);
            reseau_neuronal[nb_couches-2].neurone[k].dactivation = reseau_neuronal[nb_couches-2].neurone[k].poids_sortants[i] * reseau_neuronal[nb_couches-1].neurone[i].dz;
        }
        // ???
        reseau_neuronal[nb_couches-1].neurone[i].dbiais = reseau_neuronal[nb_couches-1].neurone[i].dz;
    }

    for(int i=nb_couches-2; i>0; i--) { // On remonte les couche de l'avant dernière jusqu'à la première
        for(int j=0; j<neurones_par_couche[i]; j++) {
            if(reseau_neuronal[i].neurone[j].z >= 0) // ??? ...
                reseau_neuronal[i].neurone[j].dz = reseau_neuronal[i].neurone[j].dactivation;
            else // ??? ...
                reseau_neuronal[i].neurone[j].dz = 0;

            for(int k=0; k<neurones_par_couche[i-1]; k++) {
                reseau_neuronal[i-1].neurone[k].dw[j] = reseau_neuronal[i].neurone[j].dz * reseau_neuronal[i-1].neurone[k].activation;
                if(i>1) // ??? ...
                    reseau_neuronal[i-1].neurone[k].dactivation = reseau_neuronal[i-1].neurone[k].poids_sortants[j] * reseau_neuronal[i].neurone[j].dz;
            }
            reseau_neuronal[i].neurone[j].dbiais = reseau_neuronal[i].neurone[j].dz; // ??? ...
        }
    }

}




void modification_du_reseau_neuronal(int nb_couches, int* neurones_par_couche) {
    /* Modifie les poids et le biais des neurones du réseau neuronal à partir
    du nombre de couches et de la liste du nombre de neurone par couche */

    for (int i=0; i<nb_couches-1; i++) { // on exclut la dernière couche
        for (int j=0; i<neurones_par_couche[i]; j++) {
            reseau_neuronal[i].neurone[j].biais = reseau_neuronal[i].neurone[j].biais - (TAUX_APPRENTISSAGE * reseau_neuronal[i].neurone[j].dbiais); // On modifie le biais du neurone à partir des données de la propagation en arrière
            for (int k=0; k<neurones_par_couche[i+1]; k++) {
                reseau_neuronal[i].neurone[j].poids_sortants[k] = reseau_neuronal[i].neurone[j].poids_sortants[k] - (TAUX_APPRENTISSAGE * reseau_neuronal[i].neurone[j].dw[k]); // On modifie le poids du neurone à partir des données de la propagation en arrière
            }
        }
    }
}




void initialisation_du_reseau_neuronal(int nb_couches, int* neurones_par_couche) {
    /* Initialise les variables du réseau neuronal (activation, biais, poids, ...)
    en suivant de la méthode de Xavier ...... à partir du nombre de couches et de la liste du nombre de neurone par couche */
    srand(time(0));
    for (int i=0; i<nb_couches-1; i++) { // on exclut la dernière couche
        for (int j=0; j<neurones_par_couche[i]-1; j++) {
            // Initialisation des bornes supérieure et inférieure
            double borne_superieure = 1/sqrt(neurones_par_couche[i]);
            double borne_inferieure = - borne_superieure;
            for (int k=0; k<neurones_par_couche[i+1]-1; k++) { // Pour chaque neurone de la couche suivante auquel le neurone est relié
                reseau_neuronal[i].neurone[j].poids_sortants[k] = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // Initialisation des poids sortants aléatoirement
                reseau_neuronal[i].neurone[j].dw[k] = 0.0; // ... ???
            }
            if(i>0) // Pour tous les neurones n'étant pas dans la première couche
                reseau_neuronal[i].neurone[j].biais = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // On initialise le biais aléatoirement
        }
    }
    double borne_superieure = 1/sqrt(neurones_par_couche[nb_couches-1]);
    double borne_inferieure = - borne_superieure;
    for (int j=0; j<neurones_par_couche[nb_couches-1]; j++) // Pour chaque neurone de la dernière couche
        reseau_neuronal[nb_couches-1].neurone[j].biais = borne_inferieure + ((double)rand())/((double)RAND_MAX)*(borne_superieure - borne_inferieure); // On initialise le biais aléatoirement
}
