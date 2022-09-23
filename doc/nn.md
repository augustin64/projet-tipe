# Réseau de neurones simple [lien](/src/mnist)

## Lecture/ Écriture du réseau de neurone:
Le fichier est au format IDX (format binaire)
Les informations sont stockées de la manière suivante:

### Header
type | nom de la variable | commentaire
:---:|:---:|:---:
uint32_t|magic_number|Variable servant à vérifier que le fichier n'est pas corrompu, vaut 2023
uint32_t|network->nb_layers|Nombre de couches du réseau
uint32_t|network->layers[0]->nb_neurons|Nombre de neurones de la première couche
uint32_t|network->layers[1]->nb_neurons|Nombre de neurones de la deuxième couche
uint32_t|...|
uint32_t|network->layers[n-1]->nb_neurons|Nombre de neurones de la n-ième couche
uint32_t|network->layers[1]->nb_neurons|Nombre de neurones de la deuxième couche


### Corps
Et ensuite, pour chaque couche, chaque neurone:
type | nom de la variable | commentaire
:---:|:---:|:---:
float|activation|importance du neurone dans le réseau
float|biais|biais du neurone
float|weights[0]|poids vers le premier neurone de la couche suivante
float|...|
float|weights[n-1]|poids vers le dernier neurone de la couche suivante