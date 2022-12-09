# Réseau de neurones simple [code](/src/mnist)

Cette partie du code implémente un réseau de neuron simple (non convolutif)

## Compilation
```bash
make mnist
```

## Fichiers
- [main](/src/mnist/main.c) [[Documentation](/doc/mnist/main.md)] Contient la fonction main et les fonctions principales à appeler
- [mnist](/src/mnist/mnist.c) [[Documentation](/src/mnist/include/mnist.h)] Partagé avec le cnn, lit les fichiers du jeu de données [MNIST](http://yann.lecun.com/exdb/mnist/)
- [neural_network](/src/mnist/neural_network.c) [[Documentation](/src/mnist/include/neural_network.h)] Contient le coeur du nn: forward et backward propagation ainsi que quelques utilitaires (copie et patch)
- [neuron_io.c](/src/mnist/neuron_io.c) [[Documentation](/doc/mnist/neuron_io.md)] Écrire et lire le réseau de neurones depuis un fichier
- [neuron.h](/src/mnist/include/neuron.h) Définit la structure `Network` et les structures qui y sont liées

- [preview.c](/src/mnist/preview.c) [[Documentation](/doc/mnist/preview.md)] Afficher les images chargées
- [utils.c](/src/mnist/utils.c) [[Documentation](/doc/mnist/utils.md)] Contient un ensemble de fonctions utiles à des fins de débogage