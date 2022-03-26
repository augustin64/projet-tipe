typedef struct Neurone{
    float activation; // Caractérise l'activation du neurone
    float* poids_sortants; // Liste de tous les poids des arêtes sortants du neurone
    float biais; // Caractérise le biais du neurone
    float z; // Sauvegarde des calculs faits sur le neurone (programmation dynamique)

    float dactivation;
    float *dw;
    float dbiais;
    float dz;
} Neurone;


typedef struct Couche{
    int nb_neurone; // Nombre de neurones dans la couche (longueur de la liste ci-dessous)
    Neurone* neurone; // Liste des neurones dans la couche
} Couche;

typedef struct Reseau{
    int nb_couche;
    Couche* couche;
} Reseau;