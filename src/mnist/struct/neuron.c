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