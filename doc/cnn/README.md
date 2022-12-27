# Réseau de neurones convolutif [code](/src/cnn)

Cette partie du code implémente un réseau de neuron convolutif.

## Compilation
```bash
make cnn
```

## Fichiers
- [main](/src/cnn/main.c) [[Documentation](/doc/cnn/main.md)] Contient la fonction main et lit les arguments à son appel
- [train](/src/cnn/train.c) [[Documentation](/doc/cnn/train.md)] Contient la partie plus haut niveau du code entraînant le réseau. Implémente la programmation concurrentielle
- [convolution](/src/cnn/convolution.cu) [[Documentation](/doc/cnn/convolution.md)] Convolution de matrices, en C et en CUDA
- [jpeg](/src/cnn/jpeg.c) [[Documentation](/doc/cnn/jpeg.md)] Lecture d'un jeu de données constitué d'images au format JPEG
- [neuron io](/src/cnn/neuron_io.c) [[Documentation](/doc/cnn/neuron_io.md)] Lecture et écriture du réseau de neurone dans un fichier
- [preview](/src/cnn/preview.c) [[Documentation](/doc/cnn/preview.md)] Visualisation des données chargées pour un jeu de données de type JPEG
- [test network](/src/cnn/test_network.c) [[Documentation](/doc/cnn/test_network.md)] Ensemble de fonctions permettant d'évaluer un réseau depuis un jeu de tests et de classifier des images
- [utils](/src/cnn/utils.c) [[Header](/src/cnn/include/utils.h)] Fonctions utilitaires (Copie de réseau, vérification d'égalité, mélange de Knuth)

- [matrix multiplication](/src/cnn/matrix_multiplication.cu) [[Documentation](/doc/cnn/matrix_multiplication.md)] Maintenant inutilisé, test de multiplication de matrices en CUDA

- [backpropagation](/src/cnn/backpropagation.c) [[Documentation](/doc/cnn/backpropagation.md)]
- [cnn](/src/cnn/cnn.c) [[Documentation](/doc/cnn/cnn.md)]
- [creation](/src/cnn/creation.c) [[Documentation](/doc/cnn/creation.md)]
- [function](/src/cnn/function.c) [[Documentation](/doc/cnn/function.md)]
- [include](/src/cnn/include) [[Documentation](/doc/cnn/include.md)]
- [make](/src/cnn/make.c) [[Documentation](/doc/cnn/make.md)]
- [print](/src/cnn/print.c) [[Documentation](/doc/cnn/print.md)]
- [free](/src/cnn/free.c) [[Header](/src/cnn/include/free.h)]
- [update](/src/cnn/update.c) [[Header](/src/cnn/include/update.h)]
- [initialisation](/src/cnn/initialisation.c) [[Header](/src/cnn/include/initialisation.h)]
