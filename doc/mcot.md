Augustin LUCAS
Julien CHEMILLIER

# MCOT

## Reconnaissance de villes à l'aide d'un réseau de neurone convolutif
Depuis un certains nombre d'années, l'ordinateur est devenue très performant dans le domaine de la reconnaissance d'images notamment grâce aux réseaux de neurones. Cependant, trouver l'endroit ou a été prise une photo demeure très complexes pour les ordinateurs alors que certains humains arrivent à des résultats époustouflants.  
Nous nous sommes donc demander pourquoi si peu de travaux de recherches ont été effectués à ce sujet alors qu'il possède beaucoup de potentiel. Et si l'on peut concevoir un modèle rivalisant avec des experts dans le domaine.  

91 mots  

100 mots max

## Positionnement thématique

1. Informatique pratique
2. Mathématiques: Analyse

## Mots-clés
Français | Anglais
:---:|:---:
Réseau neuronal convolutif | Convolutional neural network
Programmation en parallèle | Multithreading
Matrices | Matrices
Reconnaissance d'images | Image recognition
Apprentissage supervisé | Supervised learning

## Bibliographie commentée (commune) (TO DO) 
contexte scientifique quelques travaux marquants

TIPE vis à vis du contexte scientifique
-> synthèse contexte scientifique
-> principes généraux
-> expérimentations
-> questions en suspens/sujet controversés

Dans le domaine scientifique, une méthode connue et approuvée pour résoudre des problèmes efficacement est le mimétisme: s'inspirer des propriétés/comportement de la nature. Et c'est de cette manière que sont né les réseaux de neurones, dans le but de s'inspirer de la structure du cerveau pour résoudre des problèmes et notamment la reconnaissance d'images. Ces réseaux neuronaux montrent des performances honorables et sont donc largement utilisés.
Mais de la compétition de classification d'image (ILSVRC) de 2012, un algorithme devance tous les autres: AlexNet. C'est le premier réseau neuronal convolutif (CNN) utilisé en compétition. Les performances d'AlexNet sont si impressionnantes que dans l'édition de ILSVRC de 2013, tous les algorithmes concurents ont basculé vers une architecture convolutive.
La majorité des algorithmes de reconnaissance d'image garde depuis la même structure de CNN.

Cependant, les premiers travaux sur la géolocalisation ont choisi une approche autre d'un réseau de neurones, à savoir, un algorithme des k plus proches voisins. Le but est d'utiliser les k plus proches images dans une base de données pour en déduire la localisation de l'image [7]. Mais cette approche est coûteuse en espace puisqu'elle demande de conserver une grande base de données d'images réparties sur la carte et beaucoup de calcul pour trouver les plus proches voisins.
C'est pour résoudre ces problèmes qu'est né l'algorithme PlaNet un CNN conçu par des chercheurs chez Google. C'est une révolution car il est à la fois moins coûteux que l'algorithme des plus proches voisins et obtient de meilleurs résultats. Et son efficacité vient en partie de son impressionnant apprentissage avec 200 coeurs de CPU qui ont été utilisés pendant 2.5 mois pour traiter les 120 millions d'images [5]. Ce travail de recherche est donc le point de départ de notre TIPE.

Nous allons voir ce problème comme un problème de classification. C'est-à-dire que l'on sépare la carte du monde en une multitude de parcelles et l'algorithme renvoye la probabilité que l'image soit dans chacune des parcelles. Les parcelles sont réparties de sorte qu'elle ait le même nombre d'images lors de l'entrainement.
L'algorithme que nous allons utiliser est un CNN qui comporte en sortie la probabilité pour chacune des parcelles. Et en entrée, il va prendre une image de taille 256x256 soit 196 608 pixels (256x256x3).
Pour simplifier le problème, nous allons non pas essayer de reconnaitre la localisation d'une image sur la Terre entière mais nous allons nous concentrer sur la carte des États-unis. Et la division de cette carte va se faire sous la forme de 50 parcelles représentant chacune un État américain . Et nous allons utiliser une base de données de 500 000 images réparties équitablement entre ces 50 États [6].

Pour atteindre cet objectif nous avons d'abord voulu comprendre le fonctionnement des réseaux de neurones convolutifs et pour cela, nous en avons conçu un en C que l'on a testé sur la base de donnée MNIST [3].
Nous avons ensuite ajouté de la programmation en parallèle pour accélérer le CNN lors de l'apprentissage.
Puis nous avons réutilisé ce CNN en le complexifiant pour l'adapter au problème que nous souhaitions résoudre.

512 mots

650 mots max

## Problématique retenue 
Il s'agit de savoir: dans quelle mesure un algorithme peut réussir à géolocaliser une photo ? Peut-il concurrencer des humains à cette tâche tout en ayant une puissance nécessaire raisonnable

30 mots

50 mots max

## Objectif du TIPE 
1. Mise en place d'un réseau de neurone non convolutionnel reconnaissant des chiffres
2. Optimisation de cet algorithme avec de la programmation en parallèle 
3. Amélioration du premier algorithme pour qu'il soit convolutif
4. Essayer de reconnaître des villes avec le programme
5. Analyse des résultats

## DOT
(4 à 8 faits marquants)  
_On a fait quoi en quand ?_
1. Avril 2022: début de l'étude de la théorie des réseaux de neurones
2. Fin avril 2022: Premiers résultats avec MNIST (80% de réussite)
3. Mi Mai 2022: Implémentation de la programmation concurrentielle + enregistrement des fichiers binaires etc donc fonctionnel + [voir](https://tipe.augustin64.fr/mnist)
4. Juin 2022: Tentative de contacts de chercheurs
4. Septembre 2022: début de l'implémentation du CNN
5. Début Novembre 2022: Implémentation des parties les plus coûteuses en CUDA
6. Novembre 2022: Premiers résultats du CNN, erronés (aléatoires pour MNIST)

## Références bibliographiques (TO DO)
(2 à 10 références)  
Numéro | Auteur | Titre | Informations
:---:|:---:|:---:|:---:
1 | NVIDIA | Documentation de CUDA |  [URL](https://docs.nvidia.com/cuda/)
2 |Grant Sanderson (3Blue1Brown) | _The basics of neural networks, and the math behind how they learn_ | [Playlist](https://www.3blue1brown.com/topics/neural-networks)
3 |Yann Lecun | MNIST | [BDD](http://yann.lecun.com/exdb/mnist/)
4 | Fei-Fei Li | ImageNet | [BDD](https://www.image-net.org/)
5 | Tobias Weyand, Ilya Kostrikov, James Philbin | _PlaNet - Photo Geolocation with Convolutional Neural Networks_ | [Archive](https://arxiv.org/abs/1602.05314)
6 | Sudharshan Suresh, Nathaniel Chodosh, Montiel Abello | Base de données _50States10k_ |  [URL](https://arxiv.org/pdf/1810.03077.pdf#Hfootnote.2)
7 | James Hays, Alexei A. Efros | Im2GPS | [URL](http://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf)
8 | Andrew Ng | Notes of a lecture of Stanford University | [URL](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf)

Livre Hemery (apprentissage artificiel concepts et algorithmes)  