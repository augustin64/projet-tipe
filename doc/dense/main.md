# Main

## Compilation

```bash
make dense-main
```

## Options à la compilation

- La définition de la taille des batches se fait dans l'un des [`#define`](/src/dense/main.c#L15)
- Le multi-threading est activé par défaut, réduisible à un seul thread actif en remplaçant [`get_nprocs()`](/src/dense/main.c#L144) par 1
- L'ajustement du nombre de couches, bien qu'étant une option en ligne de commande, ne définit pas les valeurs pour chaque couche. On préférera donc modifier directement ces valeurs dans le code [ici](/src/dense/main.c#L140) et [ici](/src/dense/main.c#L378)

## Utilisation
```bash
Usage: build/dense-main ( train | recognize | test ) [OPTIONS]

OPTIONS:
	train:
		--epochs  | -e [int]	Nombre d'époques (itérations sur tout le set de données).
		--recover | -r [FILENAME]	Récupérer depuis un modèle existant.
		--images  | -i [FILENAME]	Fichier contenant les images.
		--labels  | -l [FILENAME]	Fichier contenant les labels.
		--out     | -o [FILENAME]	Fichier où écrire le réseau de neurones.
		--delta   | -d [FILENAME]	Fichier où écrire le réseau différentiel.
		--nb-images | -N [int]	Nombres d'images à traiter.
		--start   | -s [int]	Première image à traiter.
	recognize:
		--modele  | -m [FILENAME]	Fichier contenant le réseau de neurones.
		--in      | -i [FILENAME]	Fichier contenant les images à reconnaître.
		--out     | -o (text|json)	Format de sortie.
	test:
		--images  | -i [FILENAME]	Fichier contenant les images.
		--labels  | -l [FILENAME]	Fichier contenant les labels.
		--modele  | -m [FILENAME]	Fichier contenant le réseau de neurones.
		--preview-fails | -p	Afficher les images ayant échoué.
```

## Entraînement

Entraînement du réseau de neurones

Exemple:
```bash
build/dense-main train \
    -e 15 \
    -i data/mnist/train-images-idx3-ubyte \
    -l data/mnist/train-labels-idx1-ubyte \
    -o reseau.bin
```

Le réseau de neurones est sauvegardé dans le fichier de sortie à la fin de chaque époque.  

## Reconnaissance d'images

La reconnaissance d'images se fait avec un fichier formaté de la même manière que le jeu de données MNIST.  
Le plus simple pour dessiner à la main est d'utiliser le [serveur web](/doc/webserver) prévu à cet effet  

Exemple:
```bash
build/dense-main recognize \
    -m reseau.bin \
    -i .cache/image-idx3-ubyte \
    -o json
```

## Test sur le jeu prévu à cet effet

Exemple:
```bash
build/dense-main test \
    -i data/mnist/t10k-images-idx3-ubyte \
    -l data/mnist/t10k-labels-idx1-ubyte \
    -m reseau.bin
```