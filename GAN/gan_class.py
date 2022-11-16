'''
------Géneration de profil avec un GAN------------
Ce module regroupe diverses fonctions et classes qui permettent 
de génerer des profils d'avion avec un GAN

Created: 09/11/2022
Updated: 09/11/2022
@Auteur: Ilyas Baktache
'''

# -------------------------------------------
# Librairies
# -------------------------------------------

# File managment 
import os

# Import from other class
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from data.pre_processing import *

# Data manipulation
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as snc
import jenkspy as jk
from itertools import combinations_with_replacement

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Reshape, GlobalAveragePooling1D, LeakyReLU, Input,concatenate, embedding
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.metrics import classification_report

# Time managment
import time

# -------------------------------------------
# Class
# -------------------------------------------

class pre_process_GAN:
    def createMainFile_GAN(mainFileName,bigfolder = 'GAN'):
            if not os.path.isdir(bigfolder):
                if os.name == 'nt':
                    os.makedirs(mainFileName) #Windows
                else:
                    os.mkdir(bigfolder) # Mac ou linux
            mainFileName = os.path.join(bigfolder,mainFileName)
            # Main folder for airfoil data
            if not os.path.isdir(mainFileName):
                if os.name == 'nt':
                    os.makedirs(mainFileName) #Windows
                else:
                    os.mkdir(mainFileName) # Mac ou linux
            return mainFileName

    def save_data_pre_process_GAN():
        x_coord_initial,ally,nom_profil,marchepas = format.coordinate()
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
                alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re)
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

        finesse_max = [finesse_max_0_50000,finesse_max_0_100000,finesse_max_0_200000,finesse_max_0_500000,finesse_max_0_1000000]
        nom_profil_tt_Re = [nom_profil_0_50000,nom_profil_0_100000,nom_profil_0_200000,nom_profil_0_500000,nom_profil_0_1000000]

       
        def split_data(dataset):

            dataset = np.matrix.tolist(dataset.T)
            # Number of data 
            num = len(dataset)

            # Number of each dataset : Train and Test.
            n_train  = int(num*0.7)
            n_test  = int(num*0.3)

            while n_train+n_test !=num:
                n_train+=1
            
            # All the index of the big dataset
            allindex = [i for i in range(num)]

            # Random list of index of the train dataset
            list_train = random.sample(allindex, n_train)
            # List of allindex without the train index
            index_notrain = list(set(allindex)-set(list_train))

            # List of random train index
            list_test = random.sample(index_notrain, n_test)

            x_train = []
            x_test = []

            y_test = []
            y_train = []
            for i in allindex:
                if i in list_train:
                    x_train.append(dataset[i])
                    y_train.append(1)
                elif i in list_test:
                    x_test.append(dataset[i])
                    y_test.append(1)
            return x_train,y_train,x_test,y_test

        def save_Re_data_GAN(dict):
            mainFileName = pre_process_GAN.createMainFile_GAN('post_processed_data_GAN')
            Re = dict['reynoldsNumber']
            name = os.path.join(mainFileName,"Re_0_{}.pickle".format(Re))
            with open(name, "wb") as tf:
                pickle.dump(dict,tf)

        x_train,y_train,x_test,y_test = split_data(ally_0_50000)
        dict_0_50000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_tt_Re[0],
                        'reynoldsNumber' : 50000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_50000)

        x_train,y_train,x_test,y_test = split_data(ally_0_100000)
        dict_0_100000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_tt_Re[1],
                        'reynoldsNumber' : 100000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_100000)

        x_train,y_train,x_test,y_test = split_data(ally_0_200000)
        dict_0_200000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_tt_Re[2],
                        'reynoldsNumber' : 200000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_200000)

        x_train,y_train,x_test,y_test = split_data(ally_0_500000)
        dict_0_500000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_tt_Re[3],
                        'reynoldsNumber' : 500000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_500000)
        
        x_train,y_train,x_test,y_test = split_data(ally_0_1000000)
        dict_0_1000000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_tt_Re[4],
                        'reynoldsNumber' : 1000000,
                        'coordonnee_x' : x_coord_initial
                        }
        
        save_Re_data_GAN(dict_0_1000000)
        
    def get_data_pre_process_GAN(M,Re):

        if not os.path.exists(r'GAN/post_processed_data_GAN/Re_{}_{}'.format(M,Re)):
            pre_process_GAN.save_data_pre_process_GAN()

        with open(r"GAN/post_processed_data_GAN/Re_{}_{}.pickle".format(M,Re), "rb") as file:
                dict_ok = pickle.load(file)
        
        return dict_ok
    def data_GAN(M,Re):
        '''
        Fonction qui permet d'avoir accés au données d'entrainenment 
        et de test pour entrainer un réseau de neurone
        '''
        dict_ok = pre_process_GAN.get_data_pre_process_GAN(M,Re)

        # Importe les données d'entrainement
        x_train = np.array(dict_ok['x_train']).astype('float32')
        y_train = np.array(dict_ok['y_train']).astype('float32')

        # Importe les données de test
        x_test = np.array(dict_ok['x_test']).astype('float32')
        y_test = np.array(dict_ok['y_test']).astype('float32')
        x_coord_ini = np.array(dict_ok['coordonnee_x']).astype('float32')
        
        return x_train,y_train,x_test,y_test,x_coord_ini

class models():
    def discriminateur(nb_coord,nb_class,nb_neurones = 128,fct_activation=LeakyReLU):
        discriminateur = Sequential()
        # hidden layer
        discriminateur.add(Dense(nb_neurones, input_shape=(nb_coord,), activation=fct_activation()))
        # output layer
        discriminateur.add(Dense(nb_class, activation='softmax'))
        
        
        
        return discriminateur
    