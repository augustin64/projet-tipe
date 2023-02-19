# Neuron IO

## Lecture/ Écriture du réseau de neurone:
Le fichier est au format IDX (format binaire)
Les informations sont stockées de la manière suivante:

### Header
type | nom de la variable | commentaire
:---:|:---:|:---:
uint32_t|magic_number|Variable servant à vérifier que le fichier n'est pas corrompu, vaut 1012
uint32_t|size|Nombre de couches du réseau
uint32_t|initialisation|Fonction d'initialisation du réseau
uint32_t|dropout|Probabilité d'abandon
uint32_t|input_width[0]|
uint32_t|input_depth[0]|
uint32_t|...|
uint32_t|...|
uint32_t|input_width[size-1]|
uint32_t|input_depth[size-1]|
uint32_t|type_couche[0]|
uint32_t|...|
uint32_t|type_couche[size-1]|

> type_couche:  
> | 0 -> cnn  
> | 1 -> nn  
> | 2 -> pooling

### Pré-corps:

On stocke pour chaque couche des informations supplémentaires en fonction de son type:

#### Si la couche est un cnn:
type | nom de la variable | commentaire
:---:|:---:|:---:
uint32_t|activation|
uint32_t|linearisation|
uint32_t|k_size|
uint32_t|rows|
uint32_t|columns|

#### Si la couche est un nn:
type | nom de la variable | commentaire
:---:|:---:|:---:
uint32_t|activation|
uint32_t|linearisation|
uint32_t|size_input|
uint32_t|output_units|

#### Si la couche est de type pooling:
type | nom de la variable | commentaire
:---:|:---:|:---:
uint32_t|linearisation|
uint32_t|pooling|


### Corps
On constitue ensuite le corps du fichier à partir des données contenues dans chaque couche de la manière suivante:

- Si la couche est de type pooling, on ne rajoute rien.

- Si la couche est de type cnn, on ajoute les biais et poids de manière croissante sur leurs indices:

type | nom de la variable | commentaire
:---:|:---:|:---:
float|bias[0][0][0]|biais
float|...|
float|bias[cnn->columns-1][cnn->k_size-1][cnn->k_size-1]|
float|w[0][0][0][0]|poids
float|...|
float|w[cnn->rows][cnn->columns-1][cnn->k_size-1][cnn->k_size-1]|

- Si la couche est de type nn, on ajoute les poids de manière croissante sur leurs indices:

type | nom de la variable | commentaire
:---:|:---:|:---:
float|bias[0]|biais
float|...|
float|bias[nn->output_units-1]|biais
float|weights[0][0]|poids
float|...|
float|weights[nn->size_input-1][nn->output_units-1]|
