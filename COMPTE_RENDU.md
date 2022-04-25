# Compte rendu

### 22 Avril 2022 [b30bedd](https://github.com/julienChemillier/TIPE/commit/b30bedd375e23ec7c2e5b10acf397a10885d8b5e)
Le réseau minimise la fonction d'erreur (différence entre sortie voulue et obtenue).  
Cela donne comme résultat une précision de 10.2% en moyenne soit à peine mieux qu'aléatoire.  
Chaque image renvoie les mêmes poids sur la dernière layer.
Voici un tableau comparant la fréquence d'apparition de chaque chiffre et l'activation associée sur la dernière layer :  

| Chiffre | Nombre d'occurences dans le set d'entraînement | Activation du neuron sortant | Rapport |
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
