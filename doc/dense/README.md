# Réseau de neurones simple [code](/src/dense)

Cette partie du code implémente un réseau de neuron simple (non convolutif)

## Compilation
```bash
make dense
```

## Fichiers
- [main](/src/dense/main.c) [[Documentation](/doc/dense/main.md)] Contient la fonction main et les fonctions principales à appeler
- [neural_network](/src/dense/neural_network.c) [[Documentation](/src/dense/include/neural_network.h)] Contient le coeur du nn: forward et backward propagation ainsi que quelques utilitaires (copie et patch)
- [neuron_io](/src/dense/neuron_io.c) [[Documentation](/doc/dense/neuron_io.md)] Écrire et lire le réseau de neurones depuis un fichier
- [neuron.h](/src/dense/include/neuron.h) Définit la structure `Network` et les structures qui y sont liées

- [preview](/src/dense/preview.c) [[Documentation](/doc/dense/preview.md)] Afficher les images chargées
- [utils](/src/dense/utils.c) [[Documentation](/doc/dense/utils.md)] Contient un ensemble de fonctions utiles à des fins de débogage