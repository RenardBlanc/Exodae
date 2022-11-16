'''
---------------------Profil similaire - Clustering---------------------
n

Created: 1/08/2022
Updated: 1/08/2022
@Auteur: Ilyas Baktache
'''
# %%
# Librairies
# -------------------------------------------------------------
# Import from other class
import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from data.pre_processing import *

# Import other module
from data.pre_processing import *
import pandas as pd
import pickle

class proche():
    '''
    Cette classe regroupe des fonctions dans le but est de
    trouver des profils similaire à un profils donnée à 
    l'aide du clustering
    '''

    def importRe(Re,dir = r"data/post_processed_data/"):
        # Cette fonction permet d'importer les données associée à 
        # un nombre de Reynolds
        if Re == 50000:
            with open(dir + r"Re_50000.pickle", "rb") as file:
                dict_Re = pickle.load(file)
        elif Re == 100000 : 
            with open(dir + r"Re_100000.pickle", "rb") as file:
                dict_Re = pickle.load(file)
        elif Re == 200000 : 
            with open(dir + r"Re_200000.pickle", "rb") as file:
                dict_Re = pickle.load(file)
        elif Re == 500000 : 
            with open(dir + r"Re_500000.pickle", "rb") as file:
                dict_Re = pickle.load(file)
        elif Re == 1000000 : 
            with open(dir + r"Re_1000000.pickle", "rb") as file:
                dict_Re = pickle.load(file)
        else:
            error
        return dict_Re

    def coordForThisRe(nom_profil,ally,dict_Re):
        # On importe les données associées au nombre de Reynolds
        airfoil_name = dict_Re['nom']
  
        # Définition des paramètres qui caractérise la matrice
        # des coordonnées selon y des profils
        p = np.shape(ally)[0] # nb coordonnées
        n = np.shape(ally)[1] # nb profils
        # Extraction des noms du dict associé au nombre de 
        # Reynolds
        airfoil_name = dict_Re['nom']
        
        # Liste des indices qui ne sont pas dans les données
        # associée à ce nombre de Reynolds
        notinthisRe = []

        # Initialisation des paramètres de la boucle
        i = 0
        j = 0
        for i in range(len(nom_profil)):
            if nom_profil[i]!= airfoil_name[j]:
                notinthisRe.append(i)
            else :
                j +=1

        # On crée alors une nouvelle liste d'indice sans ceux 
        # qui ne sont pas dans les données associée à ce Re
        listdes_i = [i for i in range(n)]
        for i in range(len(notinthisRe)):
            listdes_i.remove(i)

        # On définis la matrice des coordonnées associée à ce 
        # Re
        ally_Re= np.zeros((p,n-len(notinthisRe)))
        i_temp = 0
        for i in listdes_i:
            for j in range(p):
                ally_Re[j,i_temp] = ally[j,i]
            i_temp +=1
        return ally_Re

    def modele_clust_Re(dict_Re,nom_profil,ally):
        
        # On importe les données associées au nombre de Reynolds
        A = dict_Re['aire']
        finesse = dict_Re['finesse']

        # On définis la matrice des coordonnées selon y pour tous les 
        # profils qu'on a dans la base de données pour ce Re
        ally_Re = proche.coordForThisRe(nom_profil,ally,dict_Re)
        p_1 = np.shape(ally_Re)[0] # nb coordonnées
        n_1 = np.shape(ally_Re)[1] # nb profils pour ce Re

        # Défintion de la matrice du modèle
        y_model = np.zeros((p_1+2,n_1)) 
        for i in range(p_1):
            for j in range(n_1):
                y_model[i,j] = ally_Re[i,j]
        # On ajoute deux paramètres à la 
        # matrice des coordonnées: l'aire et la finesse pour ce Re
        for j in range(n_1):
            y_model[-2,j] = A[j]
            y_model[-1,j] = finesse[j]
        return y_model

    def profil_tableau(Re,ally, nom_profil, nb_cluster = 400):
        
        # On importe les données associées au nombre de Reynolds
        dict_Re = proche.importRe(Re)
        nom_profil_Re = dict_Re['nom']
        aire = dict_Re['aire']
        finesse = dict_Re['finesse']

        # On importe le modèle
        y_model=proche.modele_clust_Re(dict_Re,nom_profil,ally)
        # On réalise la réduction dimensionnelle 
        # ainsi que le clustering
        reduced_data = decomposition.PCA(n_components=2).fit_transform(y_model.T)
        kmeans = cluster.KMeans(init="k-means++", n_clusters=nb_cluster)
        kmeans.fit(reduced_data)
        
        # Définition des colones et lignes du tableau pandas
        columns = ['Classe','Aire','Finesse']
        index = nom_profil_Re
        k = len(nom_profil_Re)

        # On crée une matrice data qui regroupe les données qu'on
        # veut afficher
        data = np.zeros((k,3))
        for i in range(len(nom_profil_Re)):
            data[i,0] = list(kmeans.labels_)[i]
            data[i,1] = aire[i]
            data[i,2] = finesse[i]
        
        # On crée le tableau
        table = pd.DataFrame(data=data,index=index,columns=columns)
        table = table.sort_index(axis = 1, ascending = False)

        mainFileName = utils.createMainFile('results',bigfolder='Clustering')
        Re = dict_Re['reynoldsNumber']
        name = os.path.join(mainFileName,"Re_{}_clust_{}.pickle".format(Re,nb_cluster))
        with open(name, "wb") as tf:
            table.to_pickle(name)
        return table

    def profilWithMemory(name,Re,nb_cluster = 400):
        # Cette fonction permet de lire les fichier pickle de clustering
        # Afin d'assurer un résultats cohérent du fait du haut indice de rand
        mainFileName =  os.path.join('Clustering',"results")
        name_file = os.path.join(mainFileName,"Re_{}_clust_{}.pickle".format(Re,nb_cluster))
        table = pd.read_pickle(name_file)
        
        # Recuperation de l'indice du profil dans la list des noms
        index_name = list(table.index).index(name)
        # Récupération de la classe du profil
        index_clust = list(table['Classe'])[index_name]
        
        newtable = table[(table.Classe == index_clust) & (table.index != name)]
        return newtable
    
    def plotWithMemory(name,Re,nb_cluster = 400):
        # Cette fonction permet de lire les fichier pickle de clustering
        # Afin d'assurer un résultats cohérent du fait du haut indice de rand
        profil_proche = list(proche.profilWithMemory(name,Re,nb_cluster = 400).index)

        plt.figure(figsize = (12,8))
        x_inter,y_inter = format.rawtointer(name)
        plt.plot(x_inter,y_inter,label = name)
        for name in profil_proche:
            x_inter,y_inter = format.rawtointer(name)
            plt.plot(x_inter,y_inter,label = name)
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.legend()
        plt.show()

    def profilWithAlly_1(name,nom_profil,ally,dir = r'data/Airfoil_Polar' ):
        profil_proche = []
        reduced_data = decomposition.PCA(n_components=2).fit_transform(ally.T)
        kmeans = cluster.KMeans(init="k-means++", n_clusters=400)
        kmeans.fit(reduced_data)
        # Recuperation de l'indice du profil dans la list des noms
        index_name = nom_profil.index(name)
        # Récupération de la classe du profil
        index_clust = list(kmeans.labels_)[index_name]

        list_proche =  np.where(np.array(list(kmeans.labels_)) == index_clust)[0]

        for i in list_proche:
            profil_proche.append(nom_profil[i])
        profil_proche.remove(name)
        
        return profil_proche

    def profilWithAlly(name,nom_profil,ally,aire_all,dir = r'data/Airfoil_Polar' ):
        profil_proche = []
        reduced_data = decomposition.PCA(n_components=2).fit_transform(ally.T)
        kmeans = cluster.KMeans(init="k-means++", n_clusters=400)
        kmeans.fit(reduced_data)
        # Recuperation de l'indice du profil dans la list des noms
        index_name = nom_profil.index(name)
        # Récupération de la classe du profil
        index_clust = list(kmeans.labels_)[index_name]

        list_proche =  np.where(np.array(list(kmeans.labels_)) == index_clust)[0]

        for i in list_proche:
            profil_proche.append(nom_profil[i])
        profil_proche.remove(name)
        try :
            table = label_fin.table_all_fin_max(ally,nom_profil,aire_all,kmeans,dir = dir)
            return table, profil_proche
        except:
            return profil_proche
    
    def plotWithAlly(name,nom_profil,ally,dir = r"data/Airfoil_Coordinate"):
        profil_proche = proche.profilWithAlly_1(name,nom_profil,ally, dir = dir)
        plt.figure(figsize = (12,8))
        x_inter,y_inter = format.rawtointer(name,dir = dir)
        plt.plot(x_inter,y_inter,label = name)
        for name in profil_proche:
            x_inter,y_inter = format.rawtointer(name,dir = dir)
            plt.plot(x_inter,y_inter,label = name)
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.legend()
        plt.show()
    


class label_fin():
    def finesse_max(ally,nom_profil,dir = r'data/Airfoil_Polar'):
        # On cherche les données de polaire pour un nombre de Mach nul et 
        # des nombres de Reynolds allant de 50000 à 1000000
        M = 0
        Re_list=[50000,100000,200000,500000,1000000]

        # On note dans cette liste les finesses maximales
        finesse_max = np.zeros((len(nom_profil),len(Re_list)))
        no_data_all = [] 
        for j in range(len(Re_list)):
            Re = Re_list[j]
            # Certaines données de polaire ne sont pas disponible pour tous
            # les profils
            no_data = [] 
            for i in range(len(nom_profil)):
                name = nom_profil[i]
                alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re,mainFileName = dir)
                # Ici on choisit alpha = 0
                try :
                    cL = np.array(cL)
                    cD = np.array(cD)
                    finesse = cL/cD
                    finesse_max[i,j] = np.max(finesse)
                except:
                    no_data.append(name)
            no_data_all.append(no_data)

        finesse_max = finesse_max.round(1).T

        # M = 0, Re = 50000
        ally_0_50000 = ally.copy()
        nom_profil_0_50000 = nom_profil.copy()
        finesse_max_0_50000 = list(finesse_max[0])
        z = [False for _ in range(len(nom_profil_0_50000))]
        for nom in no_data_all[0]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_50000.pop(index)
            nom_profil_0_50000.pop(index)
        ally_0_50000 = ally_0_50000.compress(np.logical_not(z), axis = 1)

        # M = 0, Re = 100000
        ally_0_100000 = ally.copy()
        nom_profil_0_100000 = nom_profil.copy()
        finesse_max_0_100000 = list(finesse_max[1])
        z = [False for _ in range(len(nom_profil_0_100000))]
        for nom in no_data_all[1]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_100000.pop(index)
            nom_profil_0_100000.pop(index)
        ally_0_100000 = ally_0_100000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 200000
        ally_0_200000 = ally.copy()
        nom_profil_0_200000 = nom_profil.copy()
        finesse_max_0_200000 = list(finesse_max[2])
        z = [False for _ in range(len(nom_profil_0_200000))]
        for nom in no_data_all[2]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_200000.pop(index)
            nom_profil_0_200000.pop(index)
        ally_0_200000 = ally_0_200000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 500000
        ally_0_500000 = ally.copy()
        nom_profil_0_500000 = nom_profil.copy()
        finesse_max_0_500000 = list(finesse_max[3])
        z = [False for _ in range(len(nom_profil_0_500000))]
        for nom in no_data_all[3]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_500000.pop(index)
            nom_profil_0_500000.pop(index)
        ally_0_500000 = ally_0_500000.compress(np.logical_not(z), axis = 1)


        # M = 0, Re = 1000000
        ally_0_1000000 = ally.copy()
        nom_profil_0_1000000 = nom_profil.copy()
        finesse_max_0_1000000 = list(finesse_max[4])
        z = [False for _ in range(len(nom_profil_0_1000000))]
        for nom in no_data_all[4]:
            index = nom_profil.index(nom)
            z[index] = True
            finesse_max_0_1000000.pop(index)
            nom_profil_0_1000000.pop(index)
        ally_0_1000000 = ally_0_1000000.compress(np.logical_not(z), axis = 1)

        Re_list = [50000,100000,200000,500000,1000000]
        finesse_max = [finesse_max_0_50000,finesse_max_0_100000,finesse_max_0_200000,finesse_max_0_500000,finesse_max_0_1000000]
        nom_profil_tt_Re = [nom_profil_0_50000,nom_profil_0_100000,nom_profil_0_200000,nom_profil_0_500000,nom_profil_0_1000000]

        return Re_list,finesse_max,nom_profil_tt_Re
    
    def all_data_table(ally,nom_profil,aire_all,dir = r'data/Airfoil_Polar'):
        # On calcule la finesse max suivant les données de polaire
        # qu'on possede
        Re_list,finesse_max,nom_profil_tt_Re = label_fin.finesse_max(ally,nom_profil,dir = dir)

        # Définition des colones et lignes du tableau pandas
        columns = ['Aire']
        index = nom_profil
        k = len(nom_profil)
        # On crée une matrice data qui regroupe les données qu'on
        # veut afficher
        data = np.zeros((k,1+len(Re_list)))

        for i in range(len(nom_profil)):
            data[i,0] = aire_all[i]
            
        for i in range(len(Re_list)):
            nom_profil_Re = nom_profil_tt_Re[i]
            finesse_max_Re = finesse_max[i]
            Re = Re_list[i]
            columns.append('Re = {}'.format(Re))
            f = 0
            for j in range(len(nom_profil)):
                if nom_profil[j] == nom_profil_Re[f] : 
                    data[j,i+1] = finesse_max_Re[f]
                    f +=1
                else :
                    data[j,i+1] = None
                

        # On crée le tableau
        table = pd.DataFrame(data=data,index=index,columns=columns)
        return table

    def table_all_fin_max(ally,nom_profil,aire_all,kmeans,dir = r'data/Airfoil_Polar'):
        # On calcule la finesse max suivant les données de polaire
        # qu'on possede
        Re_list,finesse_max,nom_profil_tt_Re = label_fin.finesse_max(ally,nom_profil,dir = r'../data/Airfoil_Polar')

        # Définition des colones et lignes du tableau pandas
        columns = ['Classe','Aire']
        index = nom_profil
        k = len(nom_profil)
        # On crée une matrice data qui regroupe les données qu'on
        # veut afficher
        data = np.zeros((k,2+len(Re_list)))

        for i in range(len(nom_profil)):
            data[i,0] = list(kmeans.labels_)[i]
            data[i,1] = aire_all[i]
            
        for i in range(len(Re_list)):
            nom_profil_Re = nom_profil_tt_Re[i]
            finesse_max_Re = finesse_max[i]
            Re = Re_list[i]
            columns.append('Re = {}'.format(Re))
            f = 0
            for j in range(len(nom_profil)):
                if nom_profil[j] == nom_profil_Re[f] : 
                    data[j,i+2] = finesse_max_Re[f]
                    f +=1
                else :
                    data[j,i+2] = None
                

        # On crée le tableau
        table = pd.DataFrame(data=data,index=index,columns=columns)
        return table