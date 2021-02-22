#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:14:02 2020

@author: GIANG Phuong Thu, Cécile (3530406)
"""

################################## [LU3IN005] PROJET 1: BATAILLE NAVALE ###########################################

'''
    Le jeu de la bataille navale se joue sur une grille de 10 cases par 10 cases. L’adversaire place sur cette 
grille un certain nombre de bateaux qui sont caractérisés par leur longueur :
    • un porte-avions (5 cases)
    • un croiseur (4 cases)
    • un contre-torpilleurs (3 cases)
    • un sous-marin (3 cases)
    • un torpilleur (2 cases)

Il a le droit de les placer verticalement ou horizontalement. Le positionnement des bateaux reste secret, 
l’objectif du joueur est de couler tous les bateaux de l’adversaire en un nombre minimum de coup. A chaque tour 
de jeu, il choisit une case où il tire : si il n’y a aucun bateau sur la case, la réponse est vide ; si la case est 
occupée par une partie d’un bateau, la réponse est touché ; si toutes les cases d’un bateau ont été touchées,
alors la réponse est coulé et les cases correspondantes sont révélées. Lorsque tous les bateaux ont été coulés, 
le jeu s’arrête et le score du joueur est le nombre de coups qui ont été joués. 
Plus le score est petit, plus le joueur est performant.
'''

################################### PARTIE 1: MODÉLISATION ET FONCTIONS SIMPLES ###################################

import math
import numpy as np
import matplotlib.pyplot as plt

import random as rd

### CONSTANTES ET VARIABLES


# Dimensions de la grille
DIMX = 10
DIMY = 10

''' On crée un dictionnaire où chaque bateau est caractérisé par un identifiant chiffré 
(sa clé) et a pour valeur le nombre de cases qu'il occupe dans la grille:
    • un porte-avion (1): 5 cases
    • un croiseur (2): 4 cases
    • un contre-torpilleurs (3): 3 cases
    • un sous-marin (4): 3 cases
    • un torpilleur (5): 2 cases
'''

dic_bateaux = {1: 5, 2: 4, 3: 3, 4: 3, 5: 2}

''' Crée une grille, modélisée par une matrice d'entiers initialisés à 0 (cases vides) et 
de taille DIMX par DIMY.
'''
def creer_grille():
    return np.zeros((DIMX,DIMY), dtype=int)


''' Fonction qui renvoie True si le bateau peut être placé à la position et
dans la direction indiqués.
'''
def peut_placer(grille, bateau, position, direction):
   
    placable = True
    
    if(direction == 'h'): #direction horizontale
        if(position[1] + dic_bateaux[bateau]-1 > DIMY-1): 
            placable = False #Cas où les coordonnées dépassent
        else:
            for i in range(dic_bateaux[bateau]):
                if(grille[position[0], position[1] + i] != 0):
                    placable = False
    
    elif(direction == 'v'): #direction verticale
        if(position[0] + dic_bateaux[bateau]-1 > DIMX-1): 
            placable = False #Cas où les coordonnées dépassent
        else:
            for i in range(dic_bateaux[bateau]):
                if(grille[position[0] + i, position[1]] != 0):
                    placable = False
    else:
        print("Erreur: direction non valide")
        return None
    
    return placable


''' Fonction qui rend la grille modifiée s'il est possible de placer le bateau.
Chaque case occupée par un bateau contient son identifiant dans le dictionnaire dic_bateaux.
'''
#NOTE: Vérifier au préalable que l'on peut placer le bateau

def placer(grille, bateau, position, direction):
    
    if(direction == 'h'):
        for i in range(dic_bateaux[bateau]):
            grille[position[0], position[1] + i] = bateau
    
    elif(direction == 'v'):
        for i in range(dic_bateaux[bateau]):
            grille[position[0] + i, position[1]] = bateau
    else:
        print("Erreur: direction non valide")
        return None
    
    return grille


''' Fonction qui place aléatoirement le bateau dans la grille
'''

def placer_alea(grille, bateau):
    
    estPlace = False
    position = None
    direction = None
    
    while(estPlace == False):
        
        position = (rd.randint(0,DIMX-1), rd.randint(0,DIMY-1))
        if(rd.random() < 0.5): direction = 'h'
        else: direction = 'v'
        
        estPlace = peut_placer(grille, bateau, position, direction)
    
    placer(grille, bateau, position, direction)
    return position, direction


''' Fonction qui affiche la grille de jeu
'''

from matplotlib import pyplot as plt
from matplotlib import colors

def affiche(grille):
    fig = plt.figure(figsize=(6,6))
    fig.suptitle("---------- BATAILLE NAVALE ----------\n")
    plt.pcolor(grille, edgecolors='w', linewidth=0.5)
    plt.show()
    
''' Retourne True si les deux grilles entrées en argument sont égales. 
'''

def eq(grilleA, grilleB):
    return np.array_equal(grilleA, grilleB)


''' Génère une copie de la grille passée en argument
'''

def grille_copie(grille):
    copie = creer_grille()
    for i in range(DIMX):
        for j in range(DIMY):
            copie[i,j] = grille[i,j]
    return copie
            

''' Génère une grille avec les bateaux placés de manière aléatoire.
Si l'on précise en argument une liste de bateaux, on construira la grille à partir
des bateuax spécifiés (1 occurrence de chaque bateau pour chaque joueur).
'''    

def genere_grille(liste_bateaux = [1,2,3,4,5]):
    grille = creer_grille()
    
    # Liste des positions et directions de bateaux générés pour chaque joueur
    positions_bateaux = {1:[], -1:[]}
    
    # On place les bateaux des 2 joueurs
    for joueur in [1,-1]:
        for id_bateau in liste_bateaux:
            position, direction = placer_alea(grille, id_bateau)
            for i in range(dic_bateaux[id_bateau]):
                x, y = position
                if direction == 'h':
                    positions_bateaux[joueur].append((x, y + i))
                else:
                    positions_bateaux[joueur].append((x + i, y))
    return grille, positions_bateaux

# --------------------------------------------- TESTS INTERMÉDIAIRES ---------------------------------------------

def testTME1():
    grille, positions_bateaux = genere_grille()
    affiche(grille)

testTME1()



########################################## PARTIE 2: COMBINATOIRE DU JEU ##########################################


''' QUESTION 1: Donner une borne supérieure du nombre de configurations possibles pour la liste complète 
                de bateaux sur une grille de taille 10 (calcul à la main).

           
RÉPONSE:
    
    Comme l'on ne veut qu'une borne supérieure du nombre de configurations possibles pour la liste complète 
                de bateaux sur la grille, on prendra aussi en compte les cas où des bateaux sont superposés
                (ce que l'on ne devrait normalement pas faire).
                
    Ainsi POUR CHAQUE JOUEUR:
    
        * Placer un bateau de 5 cases (5 cases côte à côte):
           Sur une ligne ou une colonne: (10 - 5) + 1 = 6 possibilités. 
           --> 6*20 = 120 (horizontal et vertical) façons de placer un porte-avion dans la grille (10 lignes, 10 colonnes)
                
        * Puis placer un bateau de 4 cases (4 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 4) + 1 = 7 possibilités.
            --> 7*20 = 140 (horizontal et vertical) façons de placer un croiseur dans la grille (10 lignes, 10 colonnes)
        
        * Puis placer un bateau de 3 cases (3 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 3) + 1 = 8 possibilités.
            --> 8*20 = 160 (horizontal et vertical) façons de placer un contre-torpilleur dans la grille (10 lignes, 10 colonnes)
            
        * Puis placer un bateau de 3 cases (3 cases côte à côte):
            Sur une ligne ou une colonne: (10 - 3) + 1 = 8 possibilités.
            --> 8*20 = 160 (horizontal et vertical) façons de placer un sous-marin dans la grille (10 lignes, 10 colonnes)
            
        * Puis placer un bateau de 2 cases (2 cases côte à côte):
            Sur une ligne/colonne: (10 - 2) + 1 = 9 possibilités.
            --> 9*20 = 180 (horizontal et vertical) façons de placer un torpilleur dans la grille (10 lignes, 10 colonnes)
        
    ---> Il y a donc 120 * 140 * 160 * 160 * 180 = 77 414 400 000 configurations possibles de la grille pour 1 joueur
    ---> et donc (120 * 140 * 160 * 160 * 180)² configurations possibles pour 2 joueurs
'''



''' QUESTION 2: Donner d’abord une fonction qui permet de dénombrer le nombre de façons de placer
                un bateau donné sur une grille vide. Comparer au résultat théorique
'''

''' Retourne le nombre de façons de placer un bateau d'identifiant id_bateau dans une grille vide.
'''

def question2(id_bateau):
    grille = creer_grille()
    cpt = 0;
    for i in range(DIMX):
        for j in range(DIMY):
            if peut_placer(grille, id_bateau, (i,j), 'h'):
                cpt +=1
            if peut_placer(grille, id_bateau, (i,j), 'v'):
                cpt +=1
    return cpt

### Comparaison avec la réponse donnée à la question 1

'''
assert(question2(1) == 120)
assert(question2(2) == 140)
assert(question2(3) == 160)
assert(question2(4) == 160)
assert(question2(5) == 180)

#print("QUESTION 2: Le test est un succès.")
'''

''' A décommenter

print("Il y a...")
print("\t", question2(1), " façons de placer un porte-avion de 5 cases")
print("\t", question2(2), " façons de placer un croiseur de 4 cases")
print("\t", question2(3), " façons de placer un contre-torpilleur de 3 cases")
print("\t", question2(4), " façons de placer un sous-marin de 3 cases")
print("\t", question2(5), " façons de placer un torpilleur de 2 cases")
'''


''' QUESTION 3: Donner une fonction qui permet de dénombrer le nombre de façon de placer une liste
                de bateaux sur une grille vide. Calculer le nombre de grilles différentes pour 1, 2 et 3
                bateaux. Est-il possible de calculer de cette manière le nombre de grilles pour la liste
                complète de bateau ?
'''

''' Retourne le nombre de façon de placer une liste de bateaux sur une grille vide
'''

def question3(liste_bateaux, grille):
        
    #COMMANDE: question3(liste_bateaux, creer_grille())
    
    if(len(liste_bateaux) == 0):
        return 0
    
    cpt = 0
    bateau = liste_bateaux[0]
    
    if(len(liste_bateaux) == 1):
        
        cpt = question2(bateau)

    if(len(liste_bateaux) > 1):
        
        for i in range(DIMX):
            for j in range(DIMY):
                
                if peut_placer(grille, bateau, (i,j), 'h'):
                    cpt += question3(liste_bateaux[1:], placer(grille_copie(grille), bateau, (i,j), 'h'))
                    
                if peut_placer(grille, bateau, (i,j), 'v'):
                    cpt += question3(liste_bateaux[1:], placer(grille_copie(grille), bateau, (i,j), 'v'))
    
    return cpt

''' A décommenter

print("\nQUESTION 3:")
print("\t\t Nombre de grilles pour une liste de 1 bateau [1]: ", question3([1], creer_grille()))
print("\t\t Nombre de grilles pour une liste de 2 bateau [1,2]:", question3([1,2], creer_grille()))
print("\t\t Nombre de grilles pour une liste de 3 bateau [1,2,3]:", question3([1,2,3], creer_grille()))
'''


''' Est-il possible de calculer de cette manière le nombre de grilles pour la liste
complète de bateau ?

RÉPONSE:
    
    Il est possible de calculer de cette manière le nombre de grilles pour la liste
complète de bateau, bien que le calcul prendra beaucoup de temps à s'exécuter.
'''



''' QUESTION 4: En considérant toutes les grilles équiprobables, quel est le lien entre le nombre de
                grilles et la probabilité de tirer une grille donnée ? Donner une fonction qui prend en
                paramètre une grille, génère des grilles aléatoirement jusqu’à ce que la grille générée
                soit égale à la grille passée en paramètre et qui renvoie le nombre de grilles générées.
'''

'''
RÉPONSE:
    Si toutes les grilles sont équiprobables, alors soit p la probabilité de tirer une grille donnée et n le
    nombre de grille. Alors p = 1/n.
'''

def question4(grille):
    cpt = 0
    while np.array_equal(grille, genere_grille()) == False:
        cpt += 1
    return cpt


''' A décommenter

print("\nQUESTION 4:")
print("\t Nombre de tirages avant d'obtenir une grille générée aléatoirement avec la liste de bateaux [1,2,3]:", question4(genere_grille([1,2,3])))
'''

''' QUESTION 5: Donner un algorithme qui permet d’APPROXIMER le nombre total de grilles pour une
                liste de bateaux. Comparer les résultats avec le dénombrement des questions précédentes. 
                Est-ce une bonne manière de procéder pour la liste complète de bateaux ?
'''

'''
RÉPONSE:
    Comme cet algorithme renvoie une APPROXIMATION et non un nombre précis, nous nous appuierons sur le même modèle
    de calcul que dans la question 1 de la partie 2, ie en ne supprimant pas les cas où l'on a superposition de
    bateaux.
'''

def question5(liste_bateaux = [1,2,3,4,5]):
    if(len(liste_bateaux)==0):
        return 0
    
    cpt = 1
    for bateau in liste_bateaux:
        cpt *= (DIMX + DIMY) * (DIMX - dic_bateaux[bateau] + 1)
        print(cpt)
    return cpt*cpt #car on fait le calcul pour 2 joueurs

''' A décommenter

print("\nQUESTION 5:")
print("\t APPROXIMATION du nombre total de grilles pour une liste de bateaux [1,2,3,4,5]:", question5())
'''


################################### PARTIE 3: MODÉLISATION PROBABILISTE DU JEU ###################################

'''
Pour les questions suivantes, il est conseillé d’utiliser une classe Bataille qui contient une grille initiée aléatoirement et des méthodes 
telles que joue(self,position), victoire(self), reset(self), ...et des classes Joueur qui implémentent les différentes stratégies.
'''
    
class Bataille():
    '''
    Bataille générique, qui contient une grille initialisée aléatoirement et deux joueurs.
    * joue(self): simule une partie de bataille navale entre les 2 joueurs jusqu'à ce que l'un d'entre eux gagne (coups = 17)
    * next(self,coup): met à jour le score du joueur courant s'il a trouvé une position adverse, et change de tour
    * win(self): retourne le joueur gagnant, None si la partie n'est pas terminée
    * reset(self): réinitialise la partie
    
    '''
    
    def __init__(self, j1 = None, j2 = None):
        self.grid, self.positions_bateaux = genere_grille()
        self.joueurs = {1:j1,-1:j2}
        self.courant = 1 # j1 commencera toujours la partie
        self.scores = {1:0, -1:0} # nombre d'emplacements trouvés par j1 et j2
        self.touche = {1:False, -1:False} # Devient True si un joueur a touché une case adverse
        self.winner = None
        
    def joue(self):
        
        #affiche(self.grid)
        while self.win() == None:
            coup = self.joueurs[self.courant].get_action(self.grid,self.touche[self.courant]) # coup sous la forme d'un 3-uplet x, y, direction
            self.next(coup)
            
    def next(self, coup):
        if coup != None: 
            x, y = coup
            if (x,y) in self.positions_bateaux[self.courant*(-1)]: # si la position proposée est celle d'un bateau de l'adversaire
                self.touche[self.courant] = True
                self.scores[self.courant] += 1
            else: self.touche[self.courant] = False
            self.courant *= -1 # On change de joueur
        
    def win(self):
        winner = None
        if self.scores[1] == 17:
            winner = 1
            self.winner = self.joueurs[1]
        elif self.scores[-1] == 17:
            winner = -1
            self.winner = self.joueurs[-1]
        return winner

    def getWinner(self):
        return self.winner
    
    def getScore(self):
        return self.scores;
    
    def reset(self):
        self.grid, self.positions_bateaux = genere_grille()
        self.joueurs = {1:self.j1,-1:self.j2}
        self.courant = 1 # j1 commencera toujours la partie
        self.scores = {1:0, -1:0} # nombre d'emplacements trouvés par j1 et j2


class Joueur:
    """ Classe de joueur générique.
    * get_action(self) qui renvoie l'action correspondant a l'etat du jeu state"""
    def __init__(self):
        pass
    def get_action(self,grid=None):
        pass
    def get_nb_coups(self):
        pass
    def reset(self):
        pass
    

####### VERSION DU JOUEUR ALÉATOIRE
        
'''
QUESTION 1:
    
    En faisant des hypothèses que vous préciserez, quelle est l’espérance du nombre de coups joués avant de couler 
    tous les bateaux si on tire aléatoirement chaque coup? 
'''

'''
RÉPONSE:
    Soit X la variable aléatoire du nombre de coups qu'un joueur J doit jouer pour gagner.
    
    Pour connaître le nombre moyen de coups à jouer pour gagner, on fait un calcul 
    d'espérance.
    
    Formule de l'espérance:
        E(X) = somme des xi * P(X = xi)
    
    Ici: E(X) = somme pour i allant de 1 à 100 des i*P(X = i)
    
    ... avec pour tout i: P(X = i) = (16 parmi (i-1))/(17 parmi 100) 
        car la dernière case tirée en cas de victoire est forcément noire.
'''

def comb(n, k):
    """Nombre de combinaisons de n objets pris k a k (k parmi n)"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x

def esperance_theorique():
    e = 0
    for i in range(1,101):
        e += i*(comb(i-1,16)/comb(100,17))
    return e


## A décommenter
    
#print(esperance_theorique())

'''
 Programmer une fonction qui joue aléatoirement au jeu : une grille aléatoire est tirée, chaque coup du jeu est ensuite tiré 
    aléatoirement (on pourra tout de même éliminer les positions déjà jouées) jusqu’à ce que tous les bateaux soient touchés. 
    Votre fonction devra renvoyer le nombre de coups utilisés pour terminer le jeu. 
'''

class Joueur_aleatoire(Joueur):
    
    def __init__(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
    def get_action(self, grid=None, touche=False):
        coup = rd.choice(self.actions)
        self.actions.remove(coup)
        self.coups += 1
        return coup
    def get_nb_coups(self):
        return self.coups
    def reset(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)]
        
'''
Comment calculer la distribution de la variable aléatoire correspondant au nombre de coups pour terminer une partie? 
    Tracer le graphique de la distribution. Comparer à l’espérance théorique précédemment calculer. 
    Que remarquez vous par rapport aux hypothèses formulées? Y’a-t-il une justiﬁcation?
'''
def distribution(joueur_1, joueur_2, nb_tests, show_score = True):
    
    score = {1:0, -1:0}
    
    resultats = dict()
    for i in range(1,101):
        resultats[i] = 0
        
    for i in range(nb_tests):
        joueur_1.reset()
        joueur_2.reset()
        bataille = Bataille(joueur_1, joueur_2)
        bataille.joue()
        score[bataille.win()] += 1
        if bataille.getWinner().get_nb_coups() in resultats.keys(): resultats[bataille.getWinner().get_nb_coups()] += 1
    
    if show_score == True:
        print(score)
        
    nb_coups_moyens = sum([key*resultats[key] for key in resultats.keys()])/(1.*nb_tests)
    return nb_coups_moyens, score
    

### TEST: Bataille navale entre 2 joueurs aléatoires
    
''' A décommenter

### Traçage du graphe de la distribution

joueur_1 = Joueur_aleatoire()
joueur_2 = Joueur_aleatoire()

print("Distribution aléatoire:")
print("\tNombre théorique de coups pour gagner: ", esperance_theorique())
print("\tNombre de coups moyens pour gagner: ", distribution(joueur_1, joueur_2, 100))

X = [10*n for n in range(1,100)]
Y = []

for i in X:
    Y.append(distribution(joueur_1, joueur_2, i)[0])
    joueur_1.reset()
    joueur_2.reset()

plt.plot(X,Y)
'''

####### VERSION DU JOUEUR HEURISTIQUE

#Liste des positions relatives des voisins d'une case
voisinage = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class Joueur_heuristique(Joueur):
    
    def __init__(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.entourage = [] #Liste des cases à tester en priorité (voisins des cases touchées)
        
    def get_action(self, grid, touche=False):
        if(touche==True or self.entourage != []):
             
                if(touche==True):
                    #self.entourage = []
                    for v in voisinage:
                        v_x = self.last_coup[0] + v[0] # abscisse du voisin considéré
                        v_y = self.last_coup[1] + v[1] # ordonnée du voisin considéré
                        
                        if (v_x, v_y) in self.actions and self.last_coup != None:
                            if grid[v_x,v_y] == grid[self.last_coup[0],self.last_coup[1]]:
                                self.entourage.append((v_x,v_y))
            
                if self.entourage == []: coup = rd.choice(self.actions)
                else:
                    coup = rd.choice(self.entourage)
                    self.entourage.remove(coup)
        else:
            coup = rd.choice(self.actions)
    
        if coup in self.actions: self.actions.remove(coup)
        self.coups += 1
        self.last_coup = coup
        return coup
    
    def get_nb_coups(self):
        return self.coups
    
    def reset(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.entourage = []


### TEST: Bataille navale entre un joueur aléatoire et un joueur heuristique
        
''' A décommenter


### Traçage du graphe de la distribution

joueur_3 = Joueur_aleatoire()
joueur_4 = Joueur_heuristique()

print("\nDistribution heuristique:")

X = [10*n for n in range(1,100)]
Y = [] #Nombre moyen de coups pour gagner
total_score = {1:0,-1:0} #Gagnant


for i in X:
    nb_coups, score = distribution(joueur_3, joueur_4, i, False)
    Y.append(nb_coups)
    total_score[1] += score[1]
    total_score[-1] += score[-1]
    
    joueur_3.reset()
    joueur_4.reset()

print("\n\nNombre de victoires par joueur:", total_score)

print("\n\n   ------- Nombre de coups moyens pour gagner -------")
plt.plot(X,Y)
'''


####### VERSION DU JOUEUR PROBABILISTE SIMPLIFIÉ

'''
dic_bateaux = {1: 5, 2: 4, 3: 3, 4: 3, 5: 2}
'''

class Joueur_probabiliste_simplifie(Joueur):
    
    def __init__(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.entourage = [] #Liste des cases à tester en priorité (voisins des cases touchées)
        
        self.case_prio = (None, None, 0) # (id, x, y, proba) case prioritaire par sa probabilité élevée de contenir un bateau
        self.grid = np.zeros((DIMX, DIMY), dtype=int) # grille où le joueur stockera les cases adverses déjà touchées
        self.coup = None
        self.liste_bateaux = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} #liste du nombre de cases touchées par bateau
        
        self.bateaux_restants = [1,2,3,4,5]
    
    def get_action(self, grid = None, touche=False):
        
        self.case_prio = (None, None, 0)
        
        if(touche==True or self.entourage != []):
             
                if(touche==True):
                    #self.entourage = []
                    
                    id = grid[self.last_coup[0], self.last_coup[1]]
                    self.grid[self.last_coup[0], self.last_coup[1]] = id
                    
                    if(id in self.liste_bateaux.keys()):
                            self.liste_bateaux[id] += 1
                            if self.liste_bateaux[id] == dic_bateaux[id]: self.bateaux_restants.remove(id)
                    
                    for v in voisinage:
                        v_x = self.last_coup[0] + v[0] # abscisse du voisin considéré
                        v_y = self.last_coup[1] + v[1] # ordonnée du voisin considéré
                        
                        if (v_x, v_y) in self.actions and self.last_coup != None:
                            if grid[v_x,v_y] == grid[self.last_coup[0],self.last_coup[1]]:
                                self.entourage.append((v_x,v_y))
            
                if self.entourage == [] and self.actions!=[]: coup = rd.choice(self.actions)
                else:
                    if self.entourage!=[]: coup = rd.choice(self.entourage)
                    self.entourage.remove(coup)
                    
                
                self.last_coup = self.coup

        else:       
            
            probas = np.zeros((DIMX, DIMY), dtype=float) # grille des probas pour chaque case de contenir un bateau
            nb_places_totales = 0
                
            for bateau in self.bateaux_restants:
                #print("bateau: ", bateau)
        
                for i in range(DIMX):
                    for j in range(DIMY):
                        if peut_placer(self.grid, bateau, (i,j), 'h'):
                            for t in range(dic_bateaux[bateau]):
                                probas[i, j + t] += 1
                            nb_places_totales += 1
                        if peut_placer(self.grid, bateau, (i,j), 'v'):
                            for t in range(dic_bateaux[bateau]):
                                probas[i + t, j] += 1
                            nb_places_totales += 1
                
            for i in range(DIMX):
                for j in range(DIMY):
                    if(nb_places_totales!=0): proba = probas[i,j]/nb_places_totales
                    else: proba = 0
                    if proba > self.case_prio[2] and (i,j) in self.actions:
                        if self.actions!=[]: self.case_prio = (i, j, proba)
            #print("case_prio = ", self.case_prio)
            #print(probas)                
            # On a choisi la case prioritaire
            # On retourne le coup choisi:
            #print("bateau_prio = ", self.case_prio)
            x, y, proba = self.case_prio
            
            self.last_coup = (x, y)
            self.coup = (x,y)
     #       print("id_bateau = ", id)
     
        #print(self.grid)
        
        #print("coup = ", self.coup)
        if(self.coup in self.actions): self.actions.remove(self.coup)
        self.coups += 1
        
        
        return self.coup
    
    def get_nb_coups(self):
        return self.coups
    
    def reset(self):
        self.coups = 0 # nombre de coups joués
        self.actions = [(i,j) for i in range(10) for j in range(10)] # liste des positions et directions pas encore testées par le joueur
        self.last_coup = None
        self.case_prio = (None, None, 0) # (id, x, y, proba) case prioritaire par sa probabilité élevée de contenir un bateau
        self.grid = np.zeros((DIMX, DIMY), dtype=int) # grille où le joueur stockera les cases adverses déjà touchées
        self.coup = None
        self.liste_bateaux = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} #liste du nombre de cases touchées par bateau
        self.bateaux_restants = [1,2,3,4,5]
    


###

''' A décommenter
'''
print("\nDistribution probabiliste simplifié:")
#distribution(joueur_5, joueur_6, 10)

joueur_5 = Joueur_heuristique()
joueur_6 = Joueur_probabiliste_simplifie()

### Traçage du graphe de la distribution

X = [10*n for n in range(1,50)]
Y = [] #Nombre moyen de coups pour gagner
total_score = {1:0,-1:0} #Gagnant


for i in X:
    print(i)
    nb_coups, score = distribution(joueur_5, joueur_6, i, False)
    Y.append(nb_coups)
    total_score[1] += score[1]
    total_score[-1] += score[-1]
    
    joueur_5.reset()
    joueur_6.reset()

print("\nDistribution probabiliste simplifié:")
print("\n\nNombre de victoires par joueur:", total_score)

print("\n\n   ------- Nombre de coups moyens pour gagner -------")
plt.plot(X,Y)
