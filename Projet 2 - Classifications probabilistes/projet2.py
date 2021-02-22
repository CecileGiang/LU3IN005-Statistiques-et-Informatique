# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:35:06 2020

@author: GIANG
"""

################################### [LU3IN005] PROJET 2: CLASSIFICATIONS PROBABILISTES ####################################


####### IMPORTATION DES MODULES ET VARIABLES GLOBALES

import math
import scipy
import matplotlib.pyplot as plt
import pandas as pd

from utils import *

######################################## PARTIE 1: CLASSIFICATION A PRIORI ################################################


def getPrior(df):
    '''
    Calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de confiance à 95%
    pour l'estimation de cette probabilité
    @param df: la dataframe à étudier
    @return : un dictionnaire contenant 3 clés 'estimation', 'min5pourcent', 'max5pourcent'
    '''
    
    #Dictionnaire à renvoyer
    aPriori = dict()
    
    #Calcul intervalle de confiance à 95%
    moyenne =df.loc[:, "target"].mean()
    ecart_type=df.loc[:, "target"].std()
    
    aPriori['estimation'] = moyenne
    aPriori['min5pourcent'] = moyenne - 1.96*(ecart_type/math.sqrt(df.shape[0]))
    aPriori['max5pourcent'] = moyenne + 1.96*(ecart_type/math.sqrt(df.shape[0]))
    
    return aPriori


### Classifieur A Priori

class APrioriClassifier(AbstractClassifier):
    
    def __init__(self):
        pass
    
    def estimClass(self,df):
        # La classe majoritaire est la classe 1
        return 1
    
    def statsOnDF(self,df):
        
        vp = 0 # Vrais positifs
        vn = 0 # Vrais négatifs
        fp = 0 # Faux positifs
        fn = 0 # Faux négatifs
        precision = 0
        rappel = 0
        
        for t in df.itertuples():
            
            dic = t._asdict()
            dic.pop('Index', None)
            
            # Cas test positif
            if dic['target']==1:
              if self.estimClass(dic) == 1: vp+=1
              else: fn+=1
            # Cas test négatif
            else:
              if self.estimClass(dic)==1: fp+=1
              else: vn+=1
              
        precision = vp/(vp+fp)
        rappel = vp/(fn+vp)
        return {'VP':vp, 'VN': vn, 'FP': fp, 'FN': fn, 'Précision':precision , 'Rappel':rappel}


########################## PARTIE 2: CLASSIFICATION PROBABILISTE A 2 DIMENSIONS #########################################

def P2D_l(df, attr):
        '''
        Calcule dans la dataframe la probabilité P(attr|target) sous la forme
        d'un dictionnaire associant à la valeur 't' un dictionnaire associant à la
        valeur 'a' la probabilité P(attr = a | target = t)
        '''
        
        dic = {1 :None, 0: None}
        dic1 = {a:None for a in df[attr]}
        dic0 = {a:None for a in df[attr]}

        target_probs = [df[df.target ==0].shape[0]/len(df),df[df.target == 1].shape[0]/len(df)]
        #df.groupby('target').size().div(len(df)) proba de target
        
        #print (proba_inter.div(target_probs))
        #print df.groupby([attr, 'target']).size().div(len(df))
        for a in dic1.keys():
            inter_0a = df[(df['target'] ==0) & (df[attr] == a)].shape[0]/len(df)
            inter_1a = df[(df['target'] ==1) & (df[attr] == a)].shape[0]/len(df)
            dic0[a] = inter_0a/target_probs[0]
            dic1[a] = inter_1a/target_probs[1]
            
            dic[0] = dic0
            dic[1] = dic1
            
        return dic
      
def P2D_p(df, attr):
    """ Calcule dans le dataframe la probabilite P(target|attr) sous la forme
    d un dictionnaire associant a la valeur t un dictionnaire associant a la
    valeur 't' la probabilite P(target = t| attr = a)"""
    dic = {a:None for a in df[attr]}
    for a in dic.keys():
        dic_a = {1: None, 0: None}
        inter_0a = df[(df['target'] ==0) & (df[attr] == a)].shape[0]/len(df)
        inter_1a = df[(df['target'] ==1) & (df[attr] == a)].shape[0]/len(df)
        proba_a = df[df[attr] == a].shape[0]/len(df)
        dic_a[0] = inter_0a/proba_a
        dic_a[1] = inter_1a/proba_a
         
        dic[a] = dic_a

    return dic


class ML2DClassifier(APrioriClassifier):

  def __init__(self, df, attr):
    self.P2DL = P2D_l(df, attr)
    self.attr = attr

  def estimClass(self, attrs):
    attr=attrs[self.attr]
    attr_0 = self.P2DL[0][attr]
    attr_1 = self.P2DL[1][attr]
    if attr_0 >= attr_1:
      return 0
    return 1


class MAP2DClassifier(APrioriClassifier):
  
  def __init__(self, df, attr):
    self.P2Dp = P2D_p(df, attr)
    self.attr = attr

  def estimClass(self, attrs):
    attr=attrs[self.attr]
    attr_0 = self.P2Dp[attr][0]
    attr_1 = self.P2Dp[attr][1]
    if attr_0 >= attr_1:
      return 0
    return 1

def nbParams(df, attrs = None):
    '''
    Calcule la taille mémoire de ces tables  P(target|attr1,..,attrk) étant donné un dataframe 
    et la liste [target,attr1,...,attrl] en supposant qu'un float est représenté sur 8 octets.
    '''
    if attrs == None:
        attrs=[str(key) for key in df.keys()]
    
    cpt = 1
    for attr in attrs:
        dic = {a:None for a in df[attr]}
        cpt*= len(dic)

    #Affichage:
    nb_octets=cpt*8
    aff=""
    aff+=str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
  
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
        
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024	
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)
    return 


def nbParamsIndep(df, attrs = None):
    '''
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilité 
    étant donné un dataframe, en supposant qu'un float est représenté sur 8octets et en supposant 
    l'indépendance des variables.
    '''
    if attrs == None:
        attrs=[str(key) for key in df.keys()]

    cpt = 0
    for attr in attrs:
        dic = {a: None for a in df[attr]}
        cpt+=len(dic)

    #Affichage:
    nb_octets=cpt*8
    aff=""
    aff+=str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
    
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)
    return 


######################################### PARTIE 3: MODELES GRAPHIQUES ##################################################
    

def drawNaiveBayes(df,attr):
    '''
    Dessine un modèle naïve bayes à partir d'un dataframe df et de la racine
    donnée par attr
    '''
    s = ""
    attrs=[str(key) for key in df.keys()]
    for fils in attrs:
        if (fils!=attr):
            s+=attr+"->"+str(fils)+";"
            
    s = s[:-2]
    return drawGraph(s)


def nbParamsNaiveBayes(df,attr,attrs=None):
    '''
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilité étant donné un
    dataframe df et de la racine attr, en supposant qu'un float est représenté sur 8octets et en
    utilisant l'hypothèse du Naive Bayes
    '''
  
    if attrs == None:
        attrs=[str(key) for key in df.keys()]
  
    #Nombre d'octets pour attr
    nb_octets=8*len({val:None for val in df[attr]})
  
    if attrs != []:
        cpt = 0
        for fils in attrs:
            if(fils!=attr):
                dic = {a:None for a in df[fils]}
                cpt+= len(dic)

    #Affichage:
        nb_octets+=cpt*nb_octets
        
    aff=""
    aff+=str(len(attrs))+" variable(s) : "+str(nb_octets)+" octets"

    s_o = ""
    s_ko=""
    s_mo=""
    s_go=""
    if nb_octets>=1024:
        aff = aff + " = "
        nb_ko=nb_octets//1024
        nb_octets=nb_octets%1024
        s_o = str(nb_octets) + "o"
        s_ko = str(nb_ko) + "ko "
        if nb_ko>=1024:
            nb_mo=nb_ko//1024
            nb_ko=nb_ko%1024
            s_ko = str(nb_ko) + "ko "
            s_mo = str(nb_mo) + "mo "
            if nb_mo>=1024:
                nb_go=nb_mo//1024
                nb_mo=nb_mo%1024	
                s_mo = str(nb_mo) + "mo "
                s_go = str(nb_go) + "go "

    aff=aff + s_go + s_mo + s_ko + s_o
    print(aff)
  
    return 


class MLNaiveBayesClassifier(APrioriClassifier):
    '''
    Utilise le maximum de vraisemblance pour estimer la classe d'un individu en
    utilisant l'hypothèse du Naïve Bayes.
    '''
    def __init__(self, df):
        self.keys={str(key):None for key in df.keys()}
        self.P2DL_dic = {attr:P2D_l(df, attr) for attr in self.keys}
        self.dic = {0: None, 1: None}
    def estimProbas(self, attrs):
        attr_0=1
        attr_1=1

        for attr in attrs.keys():
            if attrs[attr] not in self.P2DL_dic[attr][0].keys():
                self.dic[0] = 0
                self.dic[1] = 0
                return self.dic
      
            if attr != "target":
                p0 = self.P2DL_dic[attr][0][attrs[attr]]
                p1 = self.P2DL_dic[attr][1][attrs[attr]]

                attr_0 = attr_0*p0
                attr_1 = attr_1*p1

        self.dic[0] = attr_0
        self.dic[1]= attr_1
        return self.dic

    def estimClass(self, attrs):
        dic = self.estimProbas(attrs)
        if dic[0] >= dic[1]:
            return 0
        return 1


class MAPNaiveBayesClassifier(APrioriClassifier):
    '''
    Utilise le maximum a posteriori pour estimer la classe d'un individu en
    utilisant l'hypothèse du Naïve Bayes.
    '''
  
    def __init__(self, df):
        self.keys={str(key):None for key in df.keys()}
        self.P2DL_dic = {attr:P2D_l(df, attr) for attr in self.keys}
        self.dic = {0: None, 1: None}
        self.df = df

    def estimProbas(self, attrs):
        attr_0=1
        attr_1=1

        somme_den = 0

        #Calcul P(target)
        p_t0 = self.df[(self.df['target'] == 0)].shape[0]/len(self.df) #P(target=0)
        p_t1 = self.df[(self.df['target'] == 1)].shape[0]/len(self.df) #P(target=1)
    
        for attr in attrs.keys():
            if attrs[attr] not in self.P2DL_dic[attr][0].keys():
                self.dic[0] = 0
                self.dic[1] = 0
                return self.dic
      
            if attr != "target":
                p0 = self.P2DL_dic[attr][0][attrs[attr]] #P(attr|target=0)
                p1 = self.P2DL_dic[attr][1][attrs[attr]] #P(attr|target=1)

                attr_0 = attr_0*p0
                attr_1 = attr_1*p1

        
        
        somme = attr_0*p_t0+attr_1*p_t1
    
        if somme == 0:
            self.dic[0] = 0
            self.dic[1] = 0
            return self.dic
    
        self.dic[0] = attr_0*p_t0/somme
        self.dic[1] = attr_1*p_t1/somme
        return self.dic


    def estimClass(self, attrs):
        dic = self.estimProbas(attrs)
        if dic[0] >= dic[1]:
            return 0
        return 1




def isIndepFromTarget(df,attr,x):
  contingency_table=pd.crosstab(df['target'], df[attr])
  chi2, p, dof, ex = scipy.stats.chi2_contingency(contingency_table, correction=False)
  return p>=x



class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Utilise le maximum de vraisemblance pour estimer la classe d'un individu en
    utilisant l'hypothèse du Naïve Bayes.
    """
    
    def __init__(self, df,x):
        self.keys={str(key):None for key in df.keys()}
        self.P2DL_dic = {attr:P2D_l(df, attr) for attr in self.keys}
        self.dic = {0: None, 1: None}
        self.df=df
        self.x=x
    
    def estimProbas(self, attrs):
        attr_0=1
        attr_1=1

        #Liste des attributs dépendant de 'target'
        new_attrs=dict()
        for attr in attrs.keys():
            if not isIndepFromTarget(self.df,attr,self.x):
                new_attrs[attr]=attrs[attr]
      
        attrs=new_attrs

        for attr in attrs.keys():
            if attrs[attr] not in self.P2DL_dic[attr][0].keys():
                self.dic[0] = 0
                self.dic[1] = 0
                return self.dic
      
            if attr != "target":
                p0 = self.P2DL_dic[attr][0][attrs[attr]]
                p1 = self.P2DL_dic[attr][1][attrs[attr]]

                attr_0 = attr_0*p0
                attr_1 = attr_1*p1

        self.dic[0] = attr_0
        self.dic[1]= attr_1
        return self.dic

    def estimClass(self, attrs):
        dic = self.estimProbas(attrs)
        if dic[0] >= dic[1]:
            return 0
        return 1

    def draw(self):
        attrs = [str(key) for key in self.df.keys()]
        new_attrs=[]
    
        for attr in attrs:
            #Pour chaque clé, test d'indépendance avec target
            #On ne garde que les clés dépendant de target
            if not isIndepFromTarget(self.df,attr,self.x):
                new_attrs.append(attr)
            else:
                print("Attributs indépendants de target:",attr)

        s=""
        for fils in new_attrs:
            if (fils!='target'):
                s+=attr+"->"+str(fils)+";"
        s = s[:-2]
        return drawGraph(s)
    
    
class ReducedMAPNaiveBayesClassifier(MLNaiveBayesClassifier):
  def __init__(self, df, x):
    self.keys={str(key):None for key in df.keys()}
    self.P2DL_dic = {attr:P2D_l(df, attr) for attr in self.keys}
    self.dic = {0: None, 1: None}
    self.df = df
    self.x=x
    
  def estimProbas(self, attrs):
    attr_0=1
    attr_1=1

    somme_den = 0

    #Liste des attributs dépendant de 'target'
    new_attrs=dict()
    for attr in attrs.keys():
        if not isIndepFromTarget(self.df,attr,self.x):
          new_attrs[attr]=attrs[attr]
      
    attrs=new_attrs

    #Calcul P(target)
    p_t0 = self.df[(self.df['target'] == 0)].shape[0]/len(self.df) #P(target=0)
    p_t1 = self.df[(self.df['target'] == 1)].shape[0]/len(self.df) #P(target=1)
    
    for attr in attrs.keys():
      if attrs[attr] not in self.P2DL_dic[attr][0].keys():
        self.dic[0] = 0
        self.dic[1] = 0
        return self.dic
      
      if attr != "target":
        p0 = self.P2DL_dic[attr][0][attrs[attr]] #P(attr|target=0)
        p1 = self.P2DL_dic[attr][1][attrs[attr]] #P(attr|target=1)

        attr_0 = attr_0*p0
        attr_1 = attr_1*p1

        
        
    somme = attr_0*p_t0+attr_1*p_t1
    
    if somme == 0:
      self.dic[0] = 0
      self.dic[1] = 0
      return self.dic
    
    self.dic[0] = attr_0*p_t0/somme
    self.dic[1] = attr_1*p_t1/somme
    return self.dic

  def estimClass(self, attrs):
    dic = self.estimProbas(attrs)
    if dic[0] >= dic[1]:
      return 0
    return 1
  
  def draw(self):
    attrs = [str(key) for key in self.df.keys()]
    new_attrs=[]
    
    for attr in attrs:
      #Pour chaque clé, test d'indépendance avec target
      #On ne garde que les clés dépendant de target
      if not isIndepFromTarget(self.df,attr,self.x):
        new_attrs.append(attr)
      else:
        print("Attributs indépendants de target:",attr)

    s=""
    for fils in attrs:
      if (fils!='target'):
        s+=attr+"->"+str(fils)+";"
    s = s[:-2]
    return drawGraph(s)



def mapClassifiers(dic,df):
    '''
    À partir d'un dictionnaire dic de classifiers et d'un dataframe df, représente graphiquement
    les classifiers dans l'espace (précision,rappel)
    '''
    plt.figure(figsize=(7,5))
  
    for key in dic:
        cl=dic[key].statsOnDF(df)
        precision=cl['Précision']
        rappel=cl['Rappel']
        print(key,": ",precision,rappel)
        plt.scatter(precision,rappel,marker='x',color='r')
        plt.annotate(key,(precision,rappel))
  
    return
