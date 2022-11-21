
'''
---------------------Application---------------------
Dans ce module, on retrouve les différentes fonctionnalité 
développé au cours de ce projet.

Created: 14/11/2022
Updated: 14/11/2022
@Auteur: Ilyas Baktache
'''

# Librairies

# On importe les données de pré-traitement
import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath('airfoil_Optimisation_ML') ) ) )

# On importe les fonctions necessaire à cette section
from data.pre_processing import *
from Clustering.similaire_cluster import *

class fcl():
    # Fonction sur la classification de profils avec le clustering
    x,ally,nom_profil,marchepas = format.coordinate(dir = r"data/Airfoil_Coordinate")

    def proche_mod_1(name):
        x,ally,nom_profil,marchepas = format.coordinate(dir = r"data/Airfoil_Coordinate")
        profil_proche = proche.profilWithAlly_1(name,nom_profil,ally)
        return profil_proche
    
    def plot_mod_1(name):
        x,ally,nom_profil,marchepas = format.coordinate(dir = r"data/Airfoil_Coordinate")
        proche.plotWithAlly(name,nom_profil,ally,dir = r"data/Airfoil_Coordinate")

    def mat_mod_2(name,nom_profil,ally):
        p = np.shape(ally)[0] # Nombre de coordonnées pour chaque profils
        n = np.shape(ally)[1] # Nombre de profils dans la base de donnée

        with open(r"data/post_processed_data/Re_500000.pickle", "rb") as file:
            Re_500000 = pickle.load(file)

        airfoil_name = Re_500000['nom']
        A = Re_500000['aire']
        finesse = Re_500000['finesse']

        notinthisRe = []
        ally_Re_500000 = []
        i = 0
        j = 0
        for i in range(len(nom_profil)):
            if nom_profil[i]!= airfoil_name[j]:
                notinthisRe.append(i)
            else :
                j +=1

        listdes_i = [i for i in range(n)]
        for i in range(len(notinthisRe)):
            listdes_i.remove(i)

        ally_Re_500000 = np.zeros((p,n-len(notinthisRe)))
        i_temp = 0
        for i in listdes_i:
            for j in range(p):
                ally_Re_500000[j,i_temp] = ally[j,i]
            i_temp +=1

        p_1 = np.shape(ally_Re_500000)[0]
        n_1 = np.shape(ally_Re_500000)[1]
        y_model_1 = np.zeros((p_1+2,n_1))
        for i in range(p_1):
            for j in range(n_1):
                y_model_1[i,j] = ally_Re_500000[i,j]

        for j in range(n_1):
            y_model_1[-2,j] = A[j]
            y_model_1[-1,j] = finesse[j]

        return y_model_1,airfoil_name
    
    def proche_mod_2(name):
        x,ally,nom_profil,marchepas = format.coordinate(dir = r"data/Airfoil_Coordinate")
        y_model_1,airfoil_name = fcl.mat_mod_2(name,nom_profil,ally)
        try : 
            profil_proche = proche.profilWithAlly_1(name,airfoil_name,y_model_1)
        except:
            lg.error('Entrez un nom de profil présent dans la base de donnée')
        return profil_proche
    
    def plot_mod_2(name):
        x,ally,nom_profil,marchepas = format.coordinate(dir = r"data/Airfoil_Coordinate")
        y_model_1,airfoil_name = fcl.mat_mod_2(name,nom_profil,ally)
        try:
            proche.plotWithAlly(name,airfoil_name,y_model_1,dir = r"data/Airfoil_Coordinate")
        except:
            lg.error('Entrez un nom de profil présent dans la base de donnée')      

class fcn():
    # Fonction sur la classification de profils avec le clustering
    print(1)

class fg():
    # Fonction sur la géneration de profils avec le GAN
    print(1)

class fp():

    # Get all data
    all_airfoils = scrap.airfoils_name()
    scrap.airfoils_coordinate(all_airfoils)
    le.allPolar(Re_list=[50000,100000,200000,500000,1000000],M_list=0)

    '''Fonction sur les profils'''
    x,ally,nom_profil,marchepas = format.coordinate(dir = r'data/Airfoil_Coordinate')
    aire_all = lb.air_profils(x,ally)
    table=label_fin.all_data_table(ally,nom_profil,aire_all,dir = r'data/Airfoil_Polar')
    
    def all_profils():
        # On cherche le nom de tous les profils dans la base de données
        return fp.nom_profil

    def all_data():
        print(fp.table)

    def data(name):
        try :
            table_all = fp.table
            print(table_all.loc[table_all.index == name])
        except : 
            lg.error('Entrez un nom de profil présent dans la base de donnée')

    def plot(name):
        try :
            format.plot_airfoil(name)
        except :
            lg.error('Entrez un nom de profil présent dans la base de donnée')
    
    