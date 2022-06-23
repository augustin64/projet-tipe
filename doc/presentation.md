# Présentation du TIPE

Julien Chemillier  
Augustin Lucas  
Élèves en MP2I

---

## Objectif - Lien avec le sujet

Geoguessr
Google: map carrés

---

## Théorie

NN simple, fonctionnement sans trop de détails
Minimisation de l'écart etc

---

### Algorithme en Pseudo-code

```python
images = charger_les_images()
reseau = créer_le_reseau()
initialiser_le_reseau_aléatoirement(reseau)

for i in range(nombre_de_batchs):
    for image in images:
        résultat = tester_reseau(reseau, image)
        propagation_de_la_différence(
            reseau,
            résultat,
            valeur_souhaitée
        )
    mettre_à_jour_le_réseau(réseau)

sauvegarder_le_réseau(réseau)
```

Note:
On découpe l'apprentissage en plusieurs phases:  
L'algorithme s'applique sur un certain nombre de batchs qui sont un cycle complet sur toute la base de données (en l’occurrence constituée d'images)  
Dans chacun de ces batchs, l'algorithme cherche donc, pour chaque image, à voir dans quelle direction les poids doivent varier pour avoir de meilleurs résultats.  
On applique ensuite une variation infinitésimale de la somme de tous ces changements et on les intègre aux différents poids.


----

```python
initialiser_le_reseau_aléatoirement(reseau)
```

Note: Première étape cruciale, car les résultats seront les mêmes en appliquant deux fois d'affilé l'algorithme sur le même réseau, mais certains réseaux auront justement des dispositions de base qui amèneront à de meilleurs résultats (montrer le graphe d'une fonction que l'on souhaite minimiser).  

----

```python
résultat = tester_reseau(reseau, image)
propagation_de_la_différence(reseau, résultat, valeur_souhaitée)
```    
Note:
Pour chaque image tour à tour, on va donc:  
1. Tester le réseau actuel sur l'image  
3. Calculer la variation qui doit être appliquée sur chaque poids pour avoir un résultat plus proche de ce que l'on souhaite obtenir.

----

```python
mettre_à_jour_le_réseau(réseau)
```
Note:
Ces différentes variation de poids ne seront donc appliquées qu'une fois que l'on aura vu toutes les images

----

```python
sauvegarder_le_réseau(réseau)
```
Note:
Une fois toutes ces étapes passées, on stocke le réseau dans un fichier binaire,  
c'est à dire qu'il contient juste les bytes qu'on lui donne contrairement à un fichier texte habituel, ce qui permet de réduire la taille du fichier.  
On pourra ensuite lire ce fichier à nouveau pour tester la cohérence des résultats


---

### Plus de détails sur la Backpropagation


---

### État actuel - Objectifs d'évolution

----

![](https://augustin64.fr/tipe/mnist-nn.png)

Note: Actuellement, reconnaissance de chiffres

----

![](https://augustin64.fr/tipe/mnist-train.png)

Note: Pour un entraînement de 50s, le réseau obtient environ 90% de réussite sur un set de données indépendant de celui sur lequel il s'est entraîné

----

#### Problème de la puissance de calcul

- Une idée de la puissance requise <!-- .element: class="fragment" data-fragment-index="1" -->
- Implémentation à plusieurs coeurs <!-- .element: class="fragment" data-fragment-index="2" -->
- Répartir entre plusieurs ordinateurs <!-- .element: class="fragment" data-fragment-index="3" -->
- Problème du réseau de neurones convolutif <!-- .element: class="fragment" data-fragment-index="4" -->
- Utilisation de la carte graphique <!-- .element: class="fragment" data-fragment-index="5" -->

Note:
Malheureusement, le puissance de calcul nécessaire est une problématique qui risque de se faire ressentir dans la suite du projet  
Car pour le même projet, Des ingénieurs de Google ont publié dans leur papier de recherche qu'il sont dû utiliser 200 coeurs de CPU qui ont tourné pendant 2 mois.  
On peut récupérer un maximum de 20 coeurs  
-> Le calcul est donc vite fait: nous sommes en retard... (Le TIPE c'est dans moins de 20 mois)

Pour contrer au maximum ce problème, on a tout d'abord implémenté l'algorithme de manière à ce qu'il utilise les différents coeurs du processeur disponibles.  
-> fait

Nous avons aussi implémenté une méthode permettant de réaliser les calculs en parallèle sur plusieurs ordinateurs à la fois mais qui s'avère être moins efficace que d'utiliser un ordinateur indépendant (question de bande-passante)  
-> fait mais à réécrire différemment

En effet, ce qui va beaucoup changer sur une implémentation ultérieure serait d'utiliser un CNN (réseau de neurones convolutif) c'est à dire qu'au lieu d'utiliser un réseau de neurones avec une implémentation linéaire comme l'a présenté Julien, on utiliserait des multiplications de matrices, chaque coefficient deviendrait donc une matrice, ce qui demande beaucoup plus de calculs à l'ordinateur, qui peuvent être optimisés notamment avec de la programmation dynamique.

Une autre méthode pour gagner en puissance brute de calcul serait d'utiliser la carte graphique de l'ordinateur directement, car optimisée pour réaliser beaucoup plus d'opérations sur les flottants par seconde, mais demandant une implémentation particulière, avec moins de mémoire disponible.  
-> Commencé mais inachevé

---

### Ressources utilisées

----

#### Simple Neural Network
- [3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
- [Neptune.ai](https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide)
- [Simeon Kostadinov: Understanding Backpropagation](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)
- [Tobias Hill: Gradient Descent](https://towardsdatascience.com/part-2-gradient-descent-and-backpropagation-bf90932c066a)

----

#### Convolutional Neural Network
- Peu d'informations et d'implémentations disponibles librement sur Internet


----

#### Jeux de données
- [MNIST](http://yann.lecun.com/exdb/mnist/)

----

#### CUDA
- [Introduction à CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (Documentation Nvidia)
- [Gestion des erreurs](https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/) (Documentation Nvidia)
- [Unified Memory](https://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf) (Présentation Nvidia)

---

## Merci de votre attention