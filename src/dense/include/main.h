#include "neural_network.h"

#ifndef DEF_MAIN_H
#define DEF_MAIN_H

/*
* Affiche une image ainsi que les prévisions faites par le réseau de neurones sur sa valeur
* width, height: dimensions de l'image
* image: tableau 2*2 contenant l'image
* previsions: prévisions faites par le réseau neuronal
*/
void print_image(unsigned int width, unsigned int height, int** image, float* previsions);

/*
* Renvoie l'indice de l'élément maximum d'un tableau de taille n
*/
int indice_max(float* tab, int n);

/*
* Affiche un message d'aide
* call: chaîne de caractères utilisées pour appeler le programme
*/
void help(char* call);

/*
* Écrit l'image les valeurs d'un tableau dans la première couche d'un réseau neuronal
* image: tableau contenant l'image
* network: réseau neuronal
* height, width: dimensions de l'image
*/
void write_image_in_network(int** image, Network* network, int height, int width, bool random_offset);

/*
* Sous fonction de 'train' assignée à un thread
* parameters: voir la structure 'TrainParameters'
*/
void* train_thread(void* parameters);

/*
* Fonction d'entraînement du réseau
* epochs: nombre d'époques
* layers: nombre de couches
* neurons: nombre de neurones sur la première couche
* recovery: nom du fichier contenant un réseau sur lequel continuer l'entraînement (Null si non utilisé)
* image_file: nom du fichier contenant les images
* label_file: nom du fichier contenant les labels associés
* out: nom du fichier dans lequel écrire le réseau entraîné
* delta: nom du fichier où écrire le réseau différentiel (utilisation en parallèle avec d'autres clients) (Null si non utilisé)
* nb_images_to_process: nombre d'images sur lesquelles entraîner le réseau  (-1 si non utilisé)
* start: index auquel démarrer si nb_images_to_process est utilisé (0 si non utilisé)
*/
void train(int epochs, char* recovery, char* image_file, char* label_file, char* out, char* delta, int nb_images_to_process, int start, bool random_offset);

/*
* Échange deux éléments d'un tableau
*/
void swap(int* tab, int i, int j);

/*
* Mélange un tableau avec le mélange de Knuth
*/
void knuth_shuffle(int* tab, int n);

/*
* Reconnaissance d'un set d'images, renvoie un tableau de float contentant les prédictions
* modele: nom du fichier contenant le réseau neuronal
* entree: nom du fichier contenant les images à reconnaître
*/
float** recognize(char* modele, char* entree, bool random_offset);

/*
* Renvoie les prédictions d'images sur stdout
* modele: nom du fichier contenant le réseau neuronal
* entree: fichier contenant les images
* sortie: vaut 'text' ou 'json', spécifie le format auquel afficher les prédictions
*/
void print_recognize(char* modele, char* entree, char* sortie, bool random_offset);

/*
* Teste un réseau neuronal avec un fichier d'images ainsi que leurs propriétés
* modele: nom du fichier contenant le réseau neuronal
* fichier_images: nom du fichier contenant les images
* fichier_labels: nom du fichier contenant les labels
* preview_fails: faut-il afficher les images qui ne sont pas correctement reconnues ?
*/
void test(char* modele, char* fichier_images, char* fichier_labels, bool preview_fails, bool random_offset);

int main(int argc, char* argv[]);

#endif