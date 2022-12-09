# Utils

## Compilation

```bash
make mnist-utils
```

## Utilisation

```bash
Usage: build/mnist-utils ( print-poids | print-biais | creer-reseau | patch-network ) [OPTIONS]

OPTIONS:
	print-poids:
		--reseau | -r [FILENAME]	Fichier contenant le réseau de neurones.
	print-biais:
		--reseau | -r [FILENAME]	Fichier contenant le réseau de neurones.
	count-labels:
		--labels | -l [FILENAME]	Fichier contenant les labels.
	creer-reseau:
		--out    | -o [FILENAME]	Fichier où écrire le réseau de neurones.
		--number | -n [int]	Numéro à privilégier.
	patch-network:
		--network | -n [FILENAME]	Fichier contenant le réseau de neurones.
		--delta   | -d [FILENAME]	Fichier de patch à utiliser.
	print-images:
		--images  | -i [FILENAME]	Fichier contenant les images.
	print-poids-neurone:
		--reseau | -r [FILENAME]	Fichier contenant le réseau de neurones.
		--neurone | -n [int]	Numéro du neurone dont il faut afficher les poids.
```