# Présentation du TIPE

Julien Chemillier  
Augustin Lucas  
Élèves en MP2I

---

## Objectif - Lien avec le sujet

![](https://augustin64.fr/tipe/geoguessr.png)

Note:
Est-ce que vous connaissez Geoguessr ?  
C’est un jeu de géographie dans lequel le joueur est placé à un endroit aléatoire sur google maps dont l'objectif est de retrouver sa position sur une carte en se déplaçant autour de la position de départ.  
Certains modes empêchent le déplacement et laissent donc le joueur avec une seule image pour retrouver sa position dans la monde le plus précisément possible sur toute la surface du globe.  

Ce problème semble impossible à résoudre et pourtant certains humains sont capables de repérer leur position à seulement quelques kilomètres près.  
Certaines personnes réussissent cependant à retrouver leur position à quelques kilomètres près, par exemple sur cette image, ce qui est assez impressionnant.  

La problématique qui s'est donc posée à nous est de savoir s'il était possible pour un ordinateur d'obtenir de tels résultats, c'est donc le sujet que nous avons choisi d'étudier pour notre TIPE.  
Nous considérons ce problème comme un problème de classification en découpant la carte du monde en milliers de cellules géographiques [IMAGE].  
L’objectif de notre programme est donc de trouver dans quelle cellule se trouve l’image qui lui est donnée.  
Regarder le problème comme une classification permet au programme d’exprimer son incertitude.  
En reconnaissant par exemple la tour Eiffel sur une photo, le programme peut montrer qu'il catégorise cela aussi bien comme la ville de Paris qu’à la ville de Las Vegas où se trouve une réplique.  
Alors qu’une autre approche forcerait le programme à ne mentionner qu’une seule de ces deux villes.

----

![](https://augustin64.fr/tipe/planet.png)

---

## Théorie

![](https://augustin64.fr/tipe/neuralnetwork.png)

----

![](https://augustin64.fr/tipe/nn.png)

Note:
Le programme que nous avons choisi d'utiliser doit donc prendre en argument les différents pixels de l'image que l'on souhaite qu'il analyse et renvoyer une probabilité pour chaque parcelle de la Terre.  
Pour réaliser cela nous avons choisi de réaliser un réseau de neurones convolutif dans le language C.  

Tout d'abord, qu'est ce qu'un réseau de neurones ? Un réseau de neurones est un modèle informatique dont la conception est inspirée du fonctionnement des neurones biologiques.
Pour parler de structure informatique de manière plus claire, il est représenté par différentes couches, chacune contenant un certain nombre de neurones:
- La couche d'entrée, sur laquelle chaque neurone correspond à un pixel de l'image
- Les couches intermédiaires dont les nombre peut varier entre 0 et 30 selon les besoins particuliers au set de données.  
- La couche de sortie, sur laquelle chaque neurone correspond à une réponse potentielle de l'algorithme.

Dans le modèle que nous avons utilisé, chaque neurone est relié à tous les neurones de la couche suivante avec une arête.
Chaque neurone et chaque arête porte une information qu'on appellera donc poids pour une arête et biais pour un neurone.

Lorsque l'algorithme tente de donner une prédiction pour une image, cela se fait en plusieurs étapes, qui s'appellent la forward propagation:
- tout d'abord, il active chaque neurone de la première couche selon la luminosité du pixel qui lui est associé (dans le cas d'une image en noir et blanc)
- Ensuite, il propage cette activation de proche en proche, couche par couche de la manière suivante:
    L'activation d'un neurone sur la couche (i+1) est égale à la somme du produit de l'activation de chacun des neurones fois le poids de l'arête qui relie ce neurone à celui que l'on souhaite calculer, auquel on retranche le biais du neurone de la couche (i+1), soit plus formellement:  
    $x_{j+1, i} = \sum\limits^{n}_{k=1} x_{j, k}*w_{j, k, i} - b_{j+1, i}$  

Pour rentrer plus en détail dans les calculs, on applique en fait la fonction sigmoid sur cette somme afin d'obtenir un résultat compris entre 0 et 1 car c'est l'intervalle dans lequel chaque valeur d'activation doit être compris.  

Par rapport à la quantité d'information dans un réseau de neurones, si on prend comme exemple ce réseau, fait pour reconnaître des chiffres entre 0 et 9 dans des images de 28x28 pixels, on a 784 neurones sur la première couche, puis 30 puis 10.  
On obtient donc un total de 25 408 poids et biais.  
L'objectif du réseau sera de trouver les valeurs de biais et poids telles que le réseau donne de bon résultats. Il va pour cela parcourir une très grande base de données et y appliquer un algorithme de backpropagation dont je vous parlerais plus en détail tout à l'heure.  

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

![](https://augustin64.fr/tipe/nn.png)

----

![](https://augustin64.fr/tipe/gradient-descent.png)

Note:
On va maintenant rentrer dans la partie la plus intéressante du réseau de neurones, à savoir: comment apprend-il ?  
En effet, si nous avons vu précédemment comment le réseau faisait pour donner des prédictions sur une image une fois entraîné,  
il faut maintenant savoir comment il fait pour apprendre de ses erreurs.  
La backpropagation, donc le fait que le réseau apprenne par lui-même, se fait après une forward propagation puisqu’elle a pour but de modifier le fonctionnement du réseau à partir du résultat de ces prédictions et des résultats voulus.  
Son objectif est donc de minimiser la fonction qui à une image associe l'écart entre le résultat du réseau de neurones et celui attendu.  

Reprenons l’exemple de la reconnaissance de chiffres avec cette image.  
Et supposons qu’après avoir effectué une forward propagation, la dernière couche soit comme ceci.  
Pour minimiser l'écart, on doit donc essayer de diminuer les valeurs de ces neurones et d'augmenter la valeur de celui-ci.  
L'objectif étant à terme de minimiser l'écart pour toutes les images dans le set de données d'entraînement.

En se restreignant à trois dimensions pour une représentation graphique, l'algorithme ici arrivera en ce point, qui est donc un minimum local.  
Pour cela, il fera simplement évoluer les poids dans la direction opposée au gradient de cette fonction.  

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
- Implémentation à plusieurs cœurs <!-- .element: class="fragment" data-fragment-index="2" -->
- Répartir entre plusieurs ordinateurs <!-- .element: class="fragment" data-fragment-index="3" -->
- Problème du réseau de neurones convolutif <!-- .element: class="fragment" data-fragment-index="4" -->
- Utilisation de la carte graphique <!-- .element: class="fragment" data-fragment-index="5" -->

Note:
Malheureusement, le puissance de calcul nécessaire est une problématique qui risque de se faire ressentir dans la suite du projet  
Car pour le même projet, Des ingénieurs de Google ont publié dans leur papier de recherche qu'il sont dû utiliser 200 cœurs de CPU qui ont tourné pendant 2 mois.  
On peut récupérer un maximum de 20 cœurs  
-> Le calcul est donc vite fait: nous sommes en retard... (Le TIPE c'est dans moins de 20 mois)

Pour contrer au maximum ce problème, on a tout d'abord implémenté l'algorithme de manière à ce qu'il utilise les différents cœurs du processeur disponibles.  
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
- [image-net](https://image-net.org)

----

#### CUDA
- [Introduction à CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (Documentation Nvidia)
- [Gestion des erreurs](https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/) (Documentation Nvidia)
- [Unified Memory](https://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf) (Présentation Nvidia)

---

## Merci de votre attention