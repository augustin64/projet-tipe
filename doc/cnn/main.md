# Main

## Compilation


#### Deux options sont disponibles à la compilation

Une première fonctionnant sous toutes les machines:
```bash
make cnn-main
```
Une seconde utilisant CUDA pour la convolution, qui sera plus rapide, mais ne fonctionnera que sur les machines équipées d'une carte graphique Nvidia (nécessite `nvcc` disponible sous le paquet `cuda` ou `cuda-tools` pour la compilation):
```bash
make cnn-main-cuda
```

## Options à la compilation

- La taille de la couche d'entrée ainsi que la fonction d'activation utilisée sont définissable dans la [création du réseau](/src/cnn/train.c#L116) (L'architecture utilisée se définit ici également, LeNet5 étant adaptée uniquement au jeu de données MNIST)
- La définition du nombre d'époques par défaut se fait dans la définition [`EPOCHS`](/src/cnn/include/train.h#L7)
- La définition de la taille des batches se fait dans la définition [`BATCHES`](/src/cnn/include/train.h#L8)
- Le multi-threading est activé par défaut, se désactive en enlevant la définition [`USE_MULTITHREADING`](/src/cnn/include/train.h#L9) (le multithreading ne fonctionne pas pour le moment)
- Il y a une option pour conserver l'ensemble du jeu de données JPEG dans la mémoire RAM [détails](/doc/cnn/jpeg.md#STORE_TO_RAM)

### Options spécifiques à CUDA
- La définition de la taille des blocs peut-être trop élevée pour certaines carte graphiques, il faudra alors réduire l'une des définitions de [`BLOCKSIZE`](/src/cnn/convolution.cu#L37)

## Utilisation
```
Usage: build/cnn-main ( train | recognize | test ) [OPTIONS]

OPTIONS:
	train:
		--dataset | -d (mnist|jpg)	Format du set de données.
	(mnist)	--images  | -i [FILENAME]	Fichier contenant les images.
	(mnist)	--labels  | -l [FILENAME]	Fichier contenant les labels.
	 (jpg) 	--datadir | -dd [FOLDER]	Dossier contenant les images.
		--recover | -r [FILENAME]	Récupérer depuis un modèle existant.
		--epochs  | -e [int]		Nombre d'époques.
		--out     | -o [FILENAME]	Fichier où écrire le réseau de neurones.
	recognize:
		--dataset | -d (mnist|jpg)	Format de l'image à reconnaître.
		--modele  | -m [FILENAME]	Fichier contenant le réseau entraîné.
		--input   | -i [FILENAME]	Image jpeg ou fichier binaire à reconnaître.
	test:
		--modele  | -m [FILENAME]	Fichier contenant le réseau entraîné.
		--dataset | -d (mnist|jpg)	Format du set de données.
	(mnist)	--images  | -i [FILENAME]	Fichier contenant les images.
	(mnist)	--labels  | -l [FILENAME]	Fichier contenant les labels.
	 (jpg) 	--datadir | -dd [FOLDER]	Dossier contenant les images.
		--preview-fails | -p		Afficher les images ayant échoué.
```

## Entraînement

Entraînement du réseau de neurones

Exemple (MNIST):
```bash
build/cnn-main train                            \
    --dataset mnist                             \
    --epochs 15                                 \
    --images data/mnist/train-images-idx3-ubyte \
    --labels data/mnist/train-labels-idx1-ubyte \
    --out reseau.bin
```

Exemple (JPG):
```bash
build/cnn-main train                            \
    --dataset jpg                               \
    --epochs 15                                 \
    --datadir data/50States10K/train            \
    --out reseau.bin
```

Le réseau de neurones entraîné est sauvegardé dans le fichier de sortie à la fin de chaque époque.  

## Reconnaissance d'images

### MNIST
La reconnaissance d'images se fait avec un fichier formaté de la même manière que le jeu de données MNIST.  
Le plus simple pour dessiner à la main est d'utiliser le [serveur web](/doc/webserver) prévu à cet effet  

Note:
Le serveur web ne prend pour le moment qu'une option pour dessiner et faire reconnaître par le réseau de neurones simple.  
Cependant, l'image dessinée est stockée dans le fichier `.cache/image-idx3-ubyte`, la faire reconnaître par le réseau convolutif est donc possible avec la commande suivante:
```bash
build/cnn-main recognize                \
    --dataset jpg                       \
    --modele reseau.bin                 \
    --input .cache/image-idx3-ubyte     \
    --output json
```

### JPEG

L'image d'entrée doit conserver la même taille que les images ayant servi à entraîner le réseau (256x256 pixels)

Exemple:
```bash
build/cnn-main recognize                \
    --dataset jpg                       \
    --modele reseau.bin                 \
    --input image.jpeg                  \
    --output json
```


## Test sur le jeu prévu à cet effet

Exemple (MNIST):
```bash
build/cnn-main test                         \
    --dataset mnist                         \
    -i data/mnist/t10k-images-idx3-ubyte    \
    -l data/mnist/t10k-labels-idx1-ubyte    \
    -m reseau.bin
```

Exemple (JPG):
```bash
build/cnn-main test                         \
    --dataset jpg                           \
    --datadir data/50States10K/test         \
    -m reseau.bin
```