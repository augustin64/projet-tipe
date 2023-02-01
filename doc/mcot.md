---
title: "MCOT"
author:
  - Julien CHEMILLIER
  - Augustin LUCAS
geometry: margin=2cm
output: pdf_document
---

# Reconnaissance de villes à l'aide d'un réseau de neurones convolutif

L'ordinateur étant devenu de plus en plus performant au cours des dernières années dans le domaine de la reconnaissance d'images, il peine toujours à classifier des images selon leur endroit de prise de vue.  
J'ai ainsi voulu tenter de concevoir un modèle rivalisant avec des experts dans le domaine.

## Ancrage

Les villes peuvent sembler très similaire d'un premier point de vue mais se distinguent très nettement de par leur architecture et les couleurs de l'environnement alentour, mon réseau de neurones tentera donc de déceler ces différences difficiles à établir pour un observateur humain.

## Positionnement thématique

1. Informatique pratique
2. Informatique théorique

## Mots-clés
Français | Anglais
:---:|:---:
Réseau neuronal convolutif | Convolutional neural network
Programmation concurrente | Concurrent computing
Reconnaissance d'images | Image recognition
Apprentissage supervisé | Supervised learning
Convolution | Convolution

## Bibliographie commentée

Dans le domaine scientifique, une méthode connue et approuvée pour résoudre des problèmes efficacement est le mimétisme: s'inspirer des propriétés/comportements de la nature. Et c'est de cette manière que sont nés les réseaux de neurones, essayant de s'inspirer de la structure du cerveau pour résoudre des problèmes dont notamment la reconnaissance d'images. Ces réseaux neuronaux montrent des performances honorables et sont donc largement utilisés.  

C'est en 1943 qu'apparaît le premier réseau de neurones artificiels conçu par Warren McCulloch et Walter Pitts dont l'objectif était d'optimiser des commandes à l'aide d'un système mathématique et informatique. Ces réseaux vont ainsi évoluer en conservant une structure relativement similaire jusque dans les années 1980 avec l'arrivée des réseaux de neurones convolutionnels (CNN). Les CNN se démarquent par leur capacité à s'adapter à différents problèmes sans avoir à changer leur architecture. Le réseau LeNet5 est le premier réseau de cette catégorie à obtenir des résultats sur la reconnaissance d'images dans les années 1990.[1]  

Mais c'est la compétition de classification d'image (ILSVRC) de 2012, qui démontre les réelles performances des CNN avec un algorithme qui devance tous les autres: AlexNet[2]. C'est le premier réseau neuronal convolutif utilisé en compétition et sort victorieux de sa première compétition. Les performances d'AlexNet creusent un tel écart avec les autres algorithmes que dans l'édition de ILSVRC de 2013, tous les concurrents ont basculé vers une architecture convolutive.  
Depuis, la majorité des algorithmes de reconnaissance d'images ont adopté la structure de CNN.[3]  

Cependant, les premiers travaux sur la géolocalisation ont choisi une approche différente d'un réseau de neurones, à savoir, un algorithme des k plus proches voisins. Le but est d'utiliser les k plus proches images dans une base de données pour en déduire la localisation de l'image [4]. Mais cette approche est coûteuse en espace puisqu'elle demande de conserver une grande base de données d'images réparties sur la carte et beaucoup de calculs pour trouver les plus proches voisins.  
C'est pour résoudre ces problèmes qu'est né le modèle PlaNet, un CNN conçu par des chercheurs chez Google. C'est une grande avancée car il est à la fois moins coûteux que l'algorithme des plus proches voisins et obtient de meilleurs résultats. Et son efficacité vient en partie de son coûteux apprentissage avec 200 coeurs de CPU qui ont été utilisés pendant 2.5 mois pour traiter les 120 millions d'images [5]. Ce travail de recherche est donc le point de départ de notre TIPE.  

Ce problème est alors abordé comme un problème de classification. C'est-à-dire que l'on sépare la carte du monde en une multitude de parcelles et l'algorithme renvoie selon sa prédiction la probabilité que l'image soit dans chacune des parcelles. Les parcelles sont réparties de sorte qu'elle ait le même nombre d'images lors de l'entraînement. Pour cela il existe une BDD (Base De Données) sur lequel les algorithmes peuvent s'entrainer [6]. Pour simplifier le problème, je ne vais non pas essayer de trouver la localisation d'une image sur la Terre entière mais plutôt me concentrer sur la carte des États-unis. On obtient alors un algorithme entrainé permettant de catégoriser les images dans les différents états des Etat-Unis.

## Problématique retenue 
Est-il possible de concevoir un algorithme de géolocalisation permettant de concurrencer des humains à cette tâche tout en conservant un temps d'entrainement raisonnable ?

## Objectifs du TIPE - Julien
1. Mise en place d'un réseau de neurones non convolutif reconnaissant des chiffres
2. Optimisation de cet algorithme avec de la programmation en parallèle
3. Amélioration du premier algorithme pour qu'il soit convolutif
4. Tentative de reconnaissance des villes avec le programme
5. Analyse des résultats
6. Comparaison de ces résultats à ceux d'humains

## Objectifs du TIPE - Augustin
L'objectif de ce TIPE étant de construire un réseau de neurones convolutif classifiant des villes tout en comprenant comment fonctionne la technologie derrière, je progresserai par étapes successives en réalisant dans un premier temps un réseau de neurones non convolutif classifiant uniquement des photographies de chiffres.  
Dans un second temps, je réaliserait un réseau convolutif plus élaboré afin de classifier des images de plus haute résolution représentant des villes.

## Références bibliographiques
Numéro | Auteur | Titre | Informations
:---:|:---:|:---:|:---:
1 | Yann LeCun | LeNet5 | http://yann.lecun.com/exdb/lenet/index.html
2 | Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton| AlexNet | https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
3 | Jiuxiang Gu, Zhenhua Wang, Jason Kuen, et al| Recent Advances in Convolutional Neural Networks | https://arxiv.org/pdf/1512.07108.pdf%C3%A3%E2%82%AC%E2%80%9A
4 | James Hays, Alexei A. Efros | Im2GPS | http://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf
5 | Tobias Weyand, Ilya Kostrikov, James Philbin | _PlaNet - Photo Geolocation with Convolutional Neural Networks_ | https://arxiv.org/abs/1602.05314
6 | Sudharshan Suresh, Nathaniel Chodosh, Montiel Abello | Base de données _50States10k_ |  https://arxiv.org/pdf/1810.03077.pdf#Hfootnote.2
7 |Yann Lecun | MNIST | http://yann.lecun.com/exdb/mnist/


---------------------------Hors Mcot-----------------------------


## DOT
1. Avril 2022: début de l'étude de la théorie des réseaux de neurones
2. Fin avril 2022: Premiers résultats avec MNIST (80% de réussite)
3. Mi Mai 2022: Implémentation de la programmation concurrentielle + enregistrement des fichiers binaires ainsi que ce qui est nécessaire pour que le tout soit fonctionnel (https://tipe.augustin64.fr/mnist)
4. Juin 2022: Tentative de contacts de chercheurs
4. Septembre 2022: début de l'implémentation du CNN
5. Début Novembre 2022: Implémentation des parties les plus coûteuses en CUDA
6. Novembre 2022: Premiers résultats du CNN, non satisfaisant
