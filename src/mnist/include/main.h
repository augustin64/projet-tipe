#ifndef DEF_MAIN_H
#define DEF_MAIN_H

typedef struct TrainParameters {
    Network* network;
    int*** images;
    int* labels;
    int start;
    int nb_images;
    int height;
    int width;
    float accuracy;
} TrainParameters;


void print_image(unsigned int width, unsigned int height, int** image, float* previsions);
int indice_max(float* tab, int n);
void help(char* call);
void write_image_in_network(int** image, Network* network, int height, int width);
void* train_images(void* parameters);
void train(int epochs, int layers, int neurons, char* recovery, char* image_file, char* label_file, char* out, char* delta, int nb_images_to_process, int start);
float** recognize(char* modele, char* entree);
void print_recognize(char* modele, char* entree, char* sortie);
void test(char* modele, char* fichier_images, char* fichier_labels, bool preview_fails);
int main(int argc, char* argv[]);

#endif