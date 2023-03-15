# Compte rendu

### **22 Avril 2022** MNIST, premiers résultats sur le réseau dense. [b30bedd](https://github.com/julienChemillier/TIPE/commit/b30bedd375e23ec7c2e5b10acf397a10885d8b5e)
Le réseau minimise la fonction d'erreur (différence entre sortie voulue et obtenue).  
Cela donne comme résultat une précision de 10.2% en moyenne soit à peine mieux qu'aléatoire.  
Chaque image renvoie les mêmes poids sur la dernière couche.  
Voici un tableau comparant la fréquence d'apparition de chaque chiffre et l'activation associée sur la dernière couche :

| Chiffre | Nombre d’occurrences dans le set d'entraînement | Activation du neurone sortant | Rapport |
| --- | --- | --- | --- |
| 0 | 23692 | 0.483112 | 49040 |
| 1 | 26968 | 0.508133 | 53072 |
| 2 | 23832 | 0.492748 | 48365 |
| 3 | 24524 | 0.536703 | 45693 |
| 4 | 23368 | 0.532142 | 43913 |
| 5 | 21684 | 0.501488 | 43239 |
| 6 | 23672 | 0.518371 | 45666 |
| 7 | 25060 | 0.499134 | 50206 |
| 8 | 23404 | 0.512515 | 45665 |
| 9 | 23796 | 0.556504 | 42759 |

<br/>
<br/>
<br/>

### **25 Avril 2022** Optimisation de la taille des époques. [698e72f](https://github.com/julienChemillier/TIPE/commit/698e72f56ed93aa6f5d9c81912ee98461f534410)
Le réseau donne des probabilités dont la somme est de 1 (grâce à softmax).  
Un problème de sur-ajustement apparaît, résultant à de mauvais résultats sur des nouvelles données.  
Plus le réseau contient de couches, plus sa convergence vers des probabilités convenables est longue.  
Voici un tableau comparant les exactitudes des différentes époques et les dimensions du réseau sur les 60 000 images (train) :


| Dimensions | Epoque  0  | Epoque  1  | Epoque  2  | Epoque  3  | Epoque  4  | Epoque  5  | Epoque  6  | Epoque  7  | Epoque  8  | Epoque  9  | Epoque  10  | Epoque  11  | Epoque  12  | Epoque  13  | Epoque  14  | Epoque  15  | Epoque  16  | Epoque  17  | Epoque  18  | Epoque  19  | Epoque  20  | Epoque 1 nouveau dataset (t10k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
| 784x10 | 10.5% |15.4% | 26.6% | 38.5% | 50.2% | 55.3% | 59.8% | 63.0% | 65.9% | 68.1% | 70.0% | 71.5% | 72.9% | 74.0% | 74.9% | 75.8% | 76.6% | 77.3% | 78.0% | 78.5% | 79.0% | 80.0% |
| 784x16x10 | 10.9% | 14.7% | 18.3% | 21.9% | 24.7% | 26.9% | 28.8% | 30.2% | 31.4% | 32.5% | 33.9% | 35.1% | 36.2% | 37.2% | 38.0% | 38.7% | 39.5% | 40.1% | 40.6% | 41.1% | 41.5% | 42.8% |
| 784x16x16x10 | 9.1% | 9.5% | 10.8% | 12.9% | 14.4% | 15.4% | 16.1% | 16.6% | 17.1% | 17.6% | 18.1% | 18.6% | 19.1% | 19.6% | 20.0% | 20.4% | 20.8% | 21.2% | 21.6% | 21.9% | 22.2% | 23.0% |
| 784x16x16x16x10 | 11.0% | 11.0% | 11.1% | 11.2% | 11.1% | 11.2% | 11.2% | 11.2% | 11.3% | 11.6% | 11.8% | 12.3% | 12.9% | 13.5% | 14.0% | 14.5% | 15.0% | 15.3% | 15.6% | 15.9% | 16.1% | 16.1% |

<br/>
<br/>
<br/>

### **14 Mai 2022** Implémentation du multithreading. [d40212d](https://github.com/julienChemillier/TIPE/commit/d40212d313b3e8260cb9f5527f261d5d86ad2d1b)
Le problème qui se posera dans le futur est celui de la puissance de calcul nécessaire.  
Pour l'optimiser, il faut donc utiliser au maximum les ressources disponibles.

<br/>
<br/>
<br/>

### **28 Septembre 2022** Enregistrement des fichiers du CNN [a478a45](https://github.com/julienChemillier/TIPE/commit/a478a454fd1698585b2de83c8abbdca36eb2111b)
Le réseau neuronal simple donnant des résultats convaincants (approximativement 90% de réussite sur l'échantillon de test),
Le réseau neuronal convolutif a commencé à être développé depuis [6532ad2](https://github.com/julienChemillier/TIPE/commit/6532ad2545f8882638209cc6918bf37a9f816840).  
Ce commit introduit l'enregistrement du réseau de neurones convolutif,
suivi de près par les tests unitaires correspondants [b12a03c](https://github.com/julienChemillier/TIPE/commit/b12a03c1baa8e8505066fa07ae2f20882a24854b).

<br/>
<br/>
<br/>

### **30 Novembre 2022** Réparation du NN simple [ffc0c6e](https://github.com/julienChemillier/TIPE/commit/ffc0c6ea9fe30c7e98624ca26867d984ec90c693)
Après un peu de débogage sur le réseau convolutif, on s'est rendu compte que le réseau simple ne fonctionnait pas bien avec des couches intermédiaires et du multithreading.  
Le problème principal étant la découpe des batches étant la même à chaque époque.  
Après implémentation d'un mélange de Knuth, tout est rentré dans l'ordre donnant des résultats encore plus satisfaisants pour les couches intermédiaires. (De l'ordre de 70% avec deux couches intermédiaires et utilisation du multithreading).

<br/>
<br/>
<br/>

### **25 Janvier 2023** Premiers résultats sur la seconde implémentation [220d0a7](https://github.com/julienChemillier/TIPE/commit/220d0a71be2a28f63ba1e1c7804e2e9fd909e12d)
Premiers résultats sur le réseau `simple_one` qui suit la structure du premier réseau.  
La backpropagation des poids ne fonctionne cependant pas sur les couches de convolution et de pooling.

<br/>
<br/>
<br/>

### **18 Février 2023** Amélioration des résultats sur la seconde implémentation [a461e07](https://github.com/julienChemillier/TIPE/commit/a461e0756a6ef1bd1e9fe587f84de7f406f1fd12)
Améliorations significative des résultats sur le réseau `simple_one` atteignant:
-92% avec une linéarisation et une dense (1x32x32, 80, 10)
La backpropagation ne fonctionne pas encore sur la convolution.