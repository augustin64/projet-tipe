# Entraînement du réseau

## Théorie
_définition_: Une **époque** est une itération sur l'entièreté du jeu de données. Une époque est divisée en batchs.  
_définition_: Un **batch** est une partie d'une époque à la fin de laquelle les modifications nécessaires calculées sont appliquées au réseau.  

Ainsi, chaque époque est divisé en batchs de taille égale.  
Les images sont itérées dans un ordre différent à chaque époque, permet d'éviter le sur-apprentissage dans une certaine mesure.

### Programmation en parallèle
Lorsque le multithreading est utilisé, chaque époque est divisé en un certain nombre de parties chacune de taille environ égale.
Chaque fil d'exécution se voir alors assigner une de ces parties.  
Lorsque chaque thread a fini son exécution, les modifications sont appliquées au réseau.


## Pratique

Le type [TrainParameters](/src/cnn/include/train.h#L15) est utilisé pour donner des arguments à chaque fil d'exécution ou au thread principal dans le cas échéant.  

### Sans multithreading
Aucune copie n'est nécessaire pour lancer la fonction [`train_thread`](/src/cnn/train.c#L27), qui est exécutée sur le fil d'exécution principal.  

### Avec multithreading
Le réseau principal est copié dans une partie différente de la mémoire pour chaque thread.  
En effet, la partie contenant les informations statiques peut-être conservé le même pour chaque thread,  
mais le tableau `input` du réseau de neurones contient les données de chaque forward propagation et se doit d'être différent.  
Il serait cependant envisageable de ne différencier que cette partie et d'utiliser des `mutex` pour garantir un accès sans conflit à la partie commune d'un même réseau.