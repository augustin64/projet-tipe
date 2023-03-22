#include "struct.h"

#ifndef DEF_TEST_NETWORK_H
#define DEF_TEST_NETWORK_H

/*
* Renvoie le taux de réussite d'un réseau sur des données de test
*/
float* test_network(int dataset_type, char* modele, char* images_file, char* labels_file, char* data_dir, bool preview_fails, bool to_stdout, bool with_offset);

/*
* Classifie un fichier d'images sous le format MNIST à partir d'un réseau préalablement entraîné
*/
void recognize_mnist(Network* network, char* input_file, char* out);

/*
* Classifie une image jpg à partir d'un réseau préalablement entraîné
*/
void recognize_jpg(Network* network, char* input_file,char* out);

/*
* Classifie une image à partir d'un réseau préalablement entraîné
*/
void recognize(int dataset_type, char* modele, char* input_file, char* out);
#endif