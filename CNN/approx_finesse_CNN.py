'''
------Approximation de la finesse maximale avec un CNN------------
Ce module regroupe diverses fonctions qui permettent d'approximer 
la finesse maximale d'un profil pour un nombre de Mach et un nombre 
de Reynolds donné 

Created: 24/10/2022
Updated: 24/10/2022
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
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Reshape, GlobalAveragePooling1D, LeakyReLU, Input,concatenate
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.metrics import classification_report

# Time managment
import time

# -------------------------------------------
# Class
# -------------------------------------------

class pre_process_CNN:
    def createMainFile_CNN(mainFileName,bigfolder = 'CNN'):
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

    def save_data_pre_process_CNN():
        x,ally,nom_profil,marchepas = format.coordinate()
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

        def discretisation_label(nom_profil_Re,finesse_max_Re,nb_class):
            Re_fin = {'nom' : nom_profil_Re, 
                            'finesse_max' : finesse_max_Re}

            df_fin = pd.DataFrame(Re_fin)

            intervalle_finesse_max = jk.jenks_breaks(df_fin['finesse_max'], n_classes=nb_class)
            df_fin['classe'] = pd.cut(df_fin['finesse_max'],
                                bins=intervalle_finesse_max,
                                labels=[i for i in range(1,nb_class+1)],
                                include_lowest=True)
            return df_fin, intervalle_finesse_max

        df_fin, intervalle_finesse_max = discretisation_label(nom_profil_0_50000,finesse_max[0],100)

        def classe2finesse_max(classe,intervalle_finesse_max):
            finesse_max_loc = (intervalle_finesse_max[classe-1] + intervalle_finesse_max[classe])/2
            return np.round(finesse_max_loc,2)

        def finesse_classe(df_fin,intervalle_finesse_max):
            # Cette fonction permet de rajouter dans le dataframe pandas
            # les données de finesse_max associée au classes
            finesse_max_fct = []
            for i in range(len(df_fin['finesse_max'])):
                classe = df_fin['classe'][i]
                finesse_max_fct.append(classe2finesse_max(classe,intervalle_finesse_max))
            
            df_fin['finesse_max_class'] = finesse_max_fct
            return df_fin

        df_fin = finesse_classe(df_fin,intervalle_finesse_max)

        def comparaison_fin_fct_Re(nom_profil_Re,finesse_max_Re,nb_class):
        
            df_fin, intervalle_finesse_max = discretisation_label(nom_profil_Re,finesse_max_Re,nb_class)
            df_fin = finesse_classe(df_fin,intervalle_finesse_max)

            list_err = []
            for i in range(len(df_fin['finesse_max'])):
                finesse_max_reelle = df_fin['finesse_max'][i]
                finesse_max_fct = df_fin['finesse_max_class'][i]
                
                if finesse_max_reelle != 0:
                    err = np.abs((finesse_max_reelle - finesse_max_fct)) / np.abs((finesse_max_reelle))
                else :
                    pass
                list_err.append(err)
            
            return list_err

        def choix_nb_classe(nom_profil_Re,finesse_max_Re,Re):
            index_class = []

            for nb_class in range(10,100):
                try:
                    list_err = comparaison_fin_fct_Re(nom_profil_Re,finesse_max_Re,nb_class)
                    err_max = (np.max(list_err)*100)
                    err_moy = (np.mean(list_err)*100)

                    if err_max <= 50 and err_moy <= 1:
                        index_class.append(nb_class)
                except:
                    pass

            #print('Pour Re = {}, il faut prendre {} classes pour respecter les critères.'.format(Re,index_class[0]))
            return index_class[0]

        # On note alors le nombres de classes nécessaire pour 
        # chaque Re dans une liste
        nb_class_list = []
        for i in range(5):
            nb_class_list.append(choix_nb_classe(nom_profil_tt_Re[i],finesse_max[i],Re_list[i]))
        
        finesse_max_classe = []

        def list_label(nom_profil_Re,finesse_max_Re,nb_class):
            df_fin, intervalle_finesse_max = discretisation_label(nom_profil_Re,finesse_max_Re,nb_class)
            df_fin = finesse_classe(df_fin,intervalle_finesse_max)
            classe_list = list(df_fin['classe'])

            return classe_list

        for i in range(5):
            finesse_max_classe.append(list_label(nom_profil_tt_Re[i],finesse_max[i],nb_class_list[i]))
        
        def split_data(dataset,finesse_max):

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
                    y_train.append(finesse_max[i])
                elif i in list_test:
                    x_test.append(dataset[i])
                    y_test.append(finesse_max[i])
            return x_train,y_train,x_test,y_test

        def save_Re_data_CNN(dict):
            mainFileName = pre_process_CNN.createMainFile_CNN('post_processed_data_CNN')
            Re = dict['reynoldsNumber']
            name = os.path.join(mainFileName,"Re_0_{}.pickle".format(Re))
            with open(name, "wb") as tf:
                pickle.dump(dict,tf)

        x_train,y_train,x_test,y_test = split_data(ally_0_50000,finesse_max_classe[0])
        dict_0_50000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_50000,
                        'nb_classe' : nb_class_list[0],
                        'reynoldsNumber' : 50000,
                        }
        save_Re_data_CNN(dict_0_50000)

        x_train,y_train,x_test,y_test = split_data(ally_0_100000,finesse_max_classe[1])
        dict_0_100000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_100000,
                        'nb_classe' : nb_class_list[2],
                        'reynoldsNumber' : 100000,
                        }
        save_Re_data_CNN(dict_0_100000)

        x_train,y_train,x_test,y_test = split_data(ally_0_200000,finesse_max_classe[2])
        dict_0_200000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_200000,
                        'nb_classe' : nb_class_list[2],
                        'reynoldsNumber' : 200000,
                        }
        save_Re_data_CNN(dict_0_200000)

        x_train,y_train,x_test,y_test = split_data(ally_0_500000,finesse_max_classe[3])
        dict_0_500000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_500000,
                        'nb_classe' : nb_class_list[3],
                        'reynoldsNumber' : 500000,
                        }
        save_Re_data_CNN(dict_0_500000)
        
        x_train,y_train,x_test,y_test = split_data(ally_0_1000000,finesse_max_classe[4])
        dict_0_1000000 = {'x_train' : x_train,
                        'y_train' : y_train,
                        'x_test' : x_test,
                        'y_test':y_test,
                        'nom_profil' : nom_profil_0_1000000,
                        'nb_classe' : nb_class_list[4],
                        'reynoldsNumber' : 1000000,
                        }
        
        save_Re_data_CNN(dict_0_1000000)
        
    def get_data_pre_process_CNN(M,Re):

        if not os.path.exists(r'CNN/post_processed_data_CNN/Re_{}_{}'.format(M,Re)):
            pre_process_CNN.save_data_pre_process_CNN()

        with open(r"CNN/post_processed_data_CNN/Re_{}_{}.pickle".format(M,Re), "rb") as file:
                dict_ok = pickle.load(file)
        
        return dict_ok
    def data_CNN(M,Re):
        '''
        Fonction qui permet d'avoir accés au données d'entrainenment 
        et de test pour entrainer un réseau de neurone
        '''
        dict_ok = pre_process_CNN.get_data_pre_process_CNN(M,Re)

        # Importe les données d'entrainement
        x_train = np.array(dict_ok['x_train']).astype('float32')
        y_train = np.array(dict_ok['y_train']).astype('float32')

        # Importe les données de test
        x_test = np.array(dict_ok['x_test']).astype('float32')
        y_test = np.array(dict_ok['y_test']).astype('float32')

        # Nombre de classes à définir
        nb_class = dict_ok['nb_classe']

        # one-hot-encoding of our labels
        y_train = y_train - 1
        y_train_hot = to_categorical(y_train, nb_class)
        y_test = y_test - 1
        y_test_hot = to_categorical(y_test, nb_class)

        nb_class = dict_ok['nb_classe']
        return x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class

class models():
    def mod_1(nb_coord,nb_class,nb_neurones = 128,fct_activation=LeakyReLU):
        model_1 = Sequential()
        # hidden layer
        model_1.add(Dense(nb_neurones, input_shape=(nb_coord,), activation=fct_activation()))
        # output layer
        model_1.add(Dense(nb_class, activation='softmax'))
        return model_1
    
    def mod_2(nb_coord,nb_class,nb_filter_1 = 64, kernel_size_1 = 3, pool_size_1 = 3,nb_filter_2 = 100, kernel_size_2 = 3,fct_activation = 'relu',nb_neurone = 128):
        model_2 = Sequential()
        model_2.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        model_2.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(MaxPooling1D(pool_size=pool_size_1))
        model_2.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_2.add(GlobalAveragePooling1D())
        model_2.add(Dropout(0.5))
        model_2.add(Dense(nb_neurone, activation=fct_activation))
        model_2.add(Dense(nb_class, activation='softmax'))
        return model_2
    
    def mod_3(nb_coord,nb_class,nb_filter_1 = 128, kernel_size_1 = 3, pool_size_1 = 3,drop1  =0.1, nb_filter_2 = 256, kernel_size_2 = 3, pool_size_2 = 3,drop2 = 0.25, nb_filter_3 = 512, kernel_size_3 = 3, drop3 = 0.5,drop4 = 0.5,fct_activation = 'relu',nb_neurone = 1024):
        model_3 = Sequential()
        model_3.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        model_3.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_1, kernel_size=kernel_size_1, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(MaxPooling1D(pool_size=pool_size_1))
        model_3.add(Dropout(drop1))
        model_3.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(MaxPooling1D(pool_size=pool_size_2))
        model_3.add(Dropout(drop2))
        model_3.add(Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation, input_shape=(nb_coord,1)))
        model_3.add(GlobalAveragePooling1D())
        model_3.add(Dropout(drop3))
        model_3.add(Dense(nb_neurone, activation=fct_activation))
        model_3.add(Dropout(drop4))
        model_3.add(Dense(nb_class, activation='softmax'))
        return model_3

    def mod_4(nb_coord,nb_class,nb_filter_1 = 64, kernel_size_1 = 3, pool_size_1 = 3, nb_drop1 =0.5,nb_filter_2 = 64, kernel_size_2 = 3, pool_size_2 = 3, nb_drop2 =0.5,nb_filter_3 = 64, kernel_size_3 = 3, pool_size_3 = 3, nb_drop3 =0.5,fct_activation = 'relu',nb_neurone = 126):
        model_4 = Sequential()
        model_4.add(Reshape((nb_coord, 1), input_shape=(nb_coord,)))
        # Head 1
        inputs1 = Input(shape = (nb_coord,1))
        conv1 = Conv1D(filters = nb_filter_1, kernel_size = kernel_size_1, activation=fct_activation)(inputs1)
        drop1 = Dropout(nb_drop1)(conv1)
        pool1 = MaxPooling1D(pool_size=pool_size_1)(drop1)
        flat1 = Flatten()(pool1)
        # Head 2
        inputs2 = Input(shape = (nb_coord,1))
        conv2 = Conv1D(filters=nb_filter_2, kernel_size=kernel_size_2, activation=fct_activation)(inputs2)
        drop2 = Dropout(nb_drop2)(conv2)
        pool2 = MaxPooling1D(pool_size=pool_size_2)(drop2)
        flat2 = Flatten()(pool2)
        # Head 3
        inputs3 = Input(shape = (nb_coord,1))
        conv3 = Conv1D(filters=nb_filter_3, kernel_size=kernel_size_3, activation=fct_activation)(inputs3)
        drop3 = Dropout(nb_drop3)(conv3)
        pool3 = MaxPooling1D(pool_size=pool_size_3)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(nb_neurone, activation=fct_activation)(merged)
        outputs = Dense(nb_class, activation='softmax')(dense1)
        model_4 = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        return model_4

    def train_model(nb_model,M,Re,BATCH_SIZE = 50,EPOCHS = 1000,plot = False):
        x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = data_CNN(M,Re)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]
        
        def link_model(nb_model):
            if nb_model == 1:
                model = models.mod_1
            elif nb_model == 2:
                model = models.mod_2
            elif nb_model == 3:
                model = models.mod_3
            elif nb_model == 4:
                model = models.mod_4
            return model 

        model = link_model(nb_model)
        modele= model(nb_coord,nb_class)
        modele.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        if nb_model == 4 :
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                verbose=1)
            # Print confusion matrix for training data
            y_pred_train = modele.predict([x_train,x_train,x_train])
        else : 
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                verbose=1)
            # Print confusion matrix for training data
            y_pred_train = modele.predict(x_train)
        # Take the class with the highest probability from the train predictions
        max_y_pred_train = np.argmax(y_pred_train, axis=1)
        print(classification_report(y_train, max_y_pred_train))

        if plot == 1 or plot == True:
            plt.figure(figsize=(12, 8))
            plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
            plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
            plt.plot(history.history['loss'], 'r--', label='Loss of training data')
            plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
            plt.title('Model Accuracy and Loss')
            plt.ylabel('Accuracy and Loss')
            plt.xlabel('Training Epoch')
            plt.ylim(0)
            plt.legend()
            plt.show()

class test_param_CNN(): 

    def mod_1(M,Re,number_of_epochs_test = 1000):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]

        report_file_path = os.path.join(mainFileName, 'mod_1_M_{}_Re_{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nouveau set d'experience \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de neurones \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de neurones sur 
        # la couche entierement connectée
        nb_neurone_list = [4,16,64,128,256,512,1024,2048,4096] 
        number_of_epochs = number_of_epochs_test
        batch_size = 50

        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for nb_neurone in nb_neurone_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {}\n ".format(nb_class,nb_neurone,number_of_epochs,batch_size)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=LeakyReLU)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            report_file.write(results)
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[0])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de neurones \n"
        report_file.write(text_start)
        report_file.write(str(nb_neurone_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        
        # Test de différents nombre d'époque
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre d'époque \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        nb_neurone =128
        if number_of_epochs_test == 1000:
            number_of_epochs_list = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
        else : 
            number_of_epochs_list = [number_of_epochs_test]
        batch_size = 50
        for number_of_epochs in number_of_epochs_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {}\n ".format(nb_class,nb_neurone,number_of_epochs,batch_size)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=LeakyReLU)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])

            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])

            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre d'epoque \n"
        report_file.write(text_start)
        report_file.write(str(number_of_epochs_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        
        # Test de différents nombre d'époque

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de paquet \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        nb_neurone =128
        number_of_epochs = number_of_epochs_test
        batch_size_list = [5,10,20,50,100,200,400,500,1000]
        for batch_size in batch_size_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {}\n ".format(nb_class,nb_neurone,number_of_epochs,batch_size)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_1(nb_coord,nb_class,nb_neurones = nb_neurone,fct_activation=LeakyReLU)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])

            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de paquet \n"
        report_file.write(text_start)
        report_file.write(str(batch_size_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        report_file.close()

    def mod_2(M,Re,number_of_epochs_test = 1000):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]

        report_file_path = os.path.join(mainFileName, 'mod_2_M_{}_Re_{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "]  Début des tests sur le mod2 \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de neurones \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de neurones sur 
        # la couche entierement connectée
        nb_neurone_list = [4,16,64,128,256,512,1024,2048,4096] 
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        nb_filter_2 = 100 
        kernel_size_2 = 3

        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for nb_neurone in nb_neurone_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de neurones \n"
        report_file.write(text_start)
        report_file.write(str(nb_neurone_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres d'epoques \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre d'époque 
        nb_neurone =128
        if number_of_epochs_test == 1000:
            number_of_epochs_list = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
        else : 
            number_of_epochs_list = [number_of_epochs_test]
        batch_size = 50
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        nb_filter_2 = 100 
        kernel_size_2 = 3
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for number_of_epochs in number_of_epochs_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre d'epoque \n"
        report_file.write(text_start)
        report_file.write(str(number_of_epochs_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de paquet \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de batch 
        nb_neurone =128
        number_of_epochs = number_of_epochs_test
        batch_size_list = [5,10,20,50,100,200,400,500,1000]
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        nb_filter_2 = 100 
        kernel_size_2 = 3
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for batch_size in batch_size_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de paquet \n"
        report_file.write(text_start)
        report_file.write(str(batch_size_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de filtres \n-----------------------------------------------------\n \n"
        report_file.write(text_start)

        # Test de différents nombre de filtre
        nb_neurone =128
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        filtr_list = [2,4,8,16,32,64,128,256,512]
        filtr_comb = combinations_with_replacement(filtr_list,2)
        kernel_size_1 = 3
        pool_size_1 = 3
        kernel_size_2 = 3

        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for filtrs in filtr_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,filtrs[0],kernel_size_1,pool_size_1,filtrs[1],kernel_size_2)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = filtrs[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = filtrs[1], kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de filtres \n"
        report_file.write(text_start)
        report_file.write(str(filtr_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de noyaux \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de noyau
        nb_neurone =128
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 64
        kernel_list = [1,2,3,4,5,6]
        kernel_comb = combinations_with_replacement(kernel_list,2)
        pool_size_1 = 3
        nb_filter_2 = 100 
        kernel_size_2 = 3
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for kernels in kernel_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernels[0],pool_size_1,nb_filter_2,kernels[1])
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1],fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de noyaux \n"
        report_file.write(text_start)
        report_file.write(str(kernel_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Pool size \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de pool size
        nb_neurone =128
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1_list = [1,2,3,4,5,6]
        nb_filter_2 = 100 
        kernel_size_2 = 3
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for pool_size_1 in pool_size_1_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,nb_filter_2,kernel_size_2)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_2(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])

            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de pool \n"
        report_file.write(text_start)
        report_file.write(str(pool_size_1_list)+ '\n' )
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        report_file.close()

    def mod_3(M,Re,number_of_epochs_test = 500):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]

        report_file_path = os.path.join(mainFileName, 'mod_3_M_{}_Re_{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nouveau set d'experience \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de neurones \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de neurones sur 
        # la couche entierement connectée
        nb_neurone_list = [4,16,64,128,256,512,1024,2048,4096] 
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 128
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.1
        nb_filter_2 = 256
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.25
        nb_filter_3 = 512
        kernel_size_3 = 3
        drop3 = 0.5
        drop4 = 0.5

        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for nb_neurone in nb_neurone_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de neurones \n"
        report_file.write(text_start)
        report_file.write(str(nb_neurone_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre d'epoque \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de epoques 
        nb_neurone = 1024
        if number_of_epochs_test==500:
            number_of_epochs_list = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
        else : 
            number_of_epochs_list = [number_of_epochs_test]
        batch_size = 50
        nb_filter_1 = 128
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.1
        nb_filter_2 = 256
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.25
        nb_filter_3 = 512
        kernel_size_3 = 3
        drop3 = 0.5
        drop4 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for number_of_epochs in number_of_epochs_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre d'epoque \n"
        report_file.write(text_start)
        report_file.write(str(number_of_epochs_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de paquet \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de paquet 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size_list = [5,10,20,50,100,200,400,500,1000]
        nb_filter_1 = 128
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.1
        nb_filter_2 = 256
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.25
        nb_filter_3 = 512
        kernel_size_3 = 3
        drop3 = 0.5
        drop4 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for batch_size in batch_size_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,drop3,drop4)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de paquet \n"
        report_file.write(text_start)
        report_file.write(str(batch_size_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de filtres \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de filtres 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        filtr_list = [64,128,256,512,1024]
        filtr_comb = combinations_with_replacement(filtr_list,3)
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.1
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.25
        kernel_size_3 = 3
        drop3 = 0.5
        drop4 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for filters in filtr_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,filters[0],kernel_size_1,pool_size_1,drop1,filters[1],kernel_size_2,pool_size_2,drop2,filters[2],kernel_size_3,drop3,drop4)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = filters[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = filters[1], kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = filters[2], kernel_size_3 = kernel_size_3, drop3 = drop3,drop4 = drop4,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de filtres \n"
        report_file.write(text_start)
        report_file.write(str(filtr_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de noyaux \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de noyaux 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 128
        kernel_size_list = [1,2,3,4]
        kernel_comb = combinations_with_replacement(kernel_size_list,3)
        pool_size_1 = 3
        drop1 = 0.1
        nb_filter_2 = 256
        pool_size_2 = 3
        drop2 = 0.25
        nb_filter_3 = 512
        drop3 = 0.5
        drop4 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for kernels in kernel_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernels[0],pool_size_1,drop1,nb_filter_2,kernels[1],pool_size_2,drop2,nb_filter_3,kernels[2],drop3,drop4)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1,drop1  =drop1, nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1], pool_size_2 = pool_size_2,drop2 = drop2, nb_filter_3 = nb_filter_3, kernel_size_3 = kernels[2], drop3 = drop3,drop4 = drop4,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de noyaux \n"
        report_file.write(text_start)
        report_file.write(str(kernel_size_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de drop \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        # Test de différents nombre de drop 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 128
        kernel_size_1 = 3
        pool_size_1 =3
        drop_list = [0.05,0.1,0.2,0.25,0.3,0.5]
        drop_comb = combinations_with_replacement(drop_list,4)
        nb_filter_2 = 256
        kernel_size_2 = 3
        pool_size_2 = 3
        nb_filter_3 = 512
        kernel_size_3 = 3

        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for drops in drop_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* drop3 : {} \n* drop4 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drops[0],nb_filter_2,kernel_size_2,pool_size_2,drops[1],nb_filter_3,kernel_size_3,drops[2],drops[3])
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_3(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1,drop1  =drops[0], nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2,drop2 = drops[1], nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, drop3 = drops[2],drop4 = drops[3],fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit(x_train,
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate(x_test,y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de drop \n"
        report_file.write(text_start)
        report_file.write(str(drop_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        report_file.close()

    def mod_4(M,Re,number_of_epochs_test = 500):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
        # Nombre de coordonnées et de profils
        nb_coord = np.shape(x_train)[1]

        report_file_path = os.path.join(mainFileName, 'mod_4_M_{}_Re_{}.txt'.format(M,Re))
        report_file = open(report_file_path, "a")

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nouveau set d'experience \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de neurones \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de neurones sur 
        # la couche entierement connectée
        nb_neurone_list = [4,16,64,128,256,512,1024,2048,4096] 
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.5
        nb_filter_2 = 64
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.5
        nb_filter_3 = 64
        kernel_size_3 = 3
        pool_size_3 = 3
        drop3 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for nb_neurone in nb_neurone_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de neurones \n"
        report_file.write(text_start)
        report_file.write(str(nb_neurone_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre d'epoque \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de epoques 
        nb_neurone = 1024
        if number_of_epochs_test ==500:
            number_of_epochs_list = [1,10,100,200,500,1000,1500,2000,3000,5000,10000]
        else: 
            number_of_epochs_list = [number_of_epochs_test]
        batch_size = 50
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.5
        nb_filter_2 = 64
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.5
        nb_filter_3 = 64
        kernel_size_3 = 3
        pool_size_3 = 3
        drop3 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for number_of_epochs in number_of_epochs_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre d'epoque \n"
        report_file.write(text_start)
        report_file.write(str(number_of_epochs_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombres de paquet \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de paquet 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size_list = [5,10,20,50,100,200,400,500,1000]
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.5
        nb_filter_2 = 64
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.5
        nb_filter_3 = 64
        kernel_size_3 = 3
        pool_size_3 = 3
        drop3 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for batch_size in batch_size_list:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drop1,nb_filter_2,kernel_size_2,pool_size_2,drop2,nb_filter_3,kernel_size_3,pool_size_3,drop3)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de paquet \n"
        report_file.write(text_start)
        report_file.write(str(batch_size_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de filtres \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de filtres 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        filtr_list = [64,128,256,512,1024]
        filtr_comb = combinations_with_replacement(filtr_list,3)
        kernel_size_1 = 3
        pool_size_1 = 3
        drop1 = 0.5
        kernel_size_2 = 3
        pool_size_2 = 3
        drop2 = 0.5
        kernel_size_3 = 3
        pool_size_3 = 3
        drop3 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for filters in filtr_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,filters[0],kernel_size_1,pool_size_1,drop1,filters[1],kernel_size_2,pool_size_2,drop2,filters[2],kernel_size_3,pool_size_3,drop3)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = filters[0], kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = filters[1], kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = filters[2], kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de filtres \n"
        report_file.write(text_start)
        report_file.write(str(filtr_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')

        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de noyaux \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de noyaux 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        kernel_size_list = [1,2,3,4,5,6]
        kernel_comb = combinations_with_replacement(kernel_size_list,3)
        nb_filter_1 = 64
        pool_size_1 = 3
        drop1 = 0.5
        nb_filter_2 = 64
        pool_size_2 = 3
        drop2 = 0.5
        nb_filter_3 = 64
        pool_size_3 = 3
        drop3 = 0.5
        # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []
        for kernels in kernel_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernels[0],pool_size_1,drop1,nb_filter_2,kernels[1],pool_size_2,drop2,nb_filter_3,kernels[2],pool_size_3,drop3)
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernels[0], pool_size_1 = pool_size_1, nb_drop1 =drop1,nb_filter_2 = nb_filter_2, kernel_size_2 = kernels[1], pool_size_2 = pool_size_2, nb_drop2 =drop2,nb_filter_3 = nb_filter_3, kernel_size_3 = kernels[2], pool_size_3 = pool_size_3, nb_drop3 =drop3,fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de noyaux \n"
        report_file.write(text_start)
        report_file.write(str(kernel_size_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Nombre de drop \n-----------------------------------------------------\n \n"
        report_file.write(text_start)
        print(text_start)
        # Test de différents nombre de drop 
        nb_neurone = 1024
        number_of_epochs = number_of_epochs_test
        batch_size = 50
        drop_list = [0.05,0.1,0.2,0.25,0.3,0.5]
        drop_comb = combinations_with_replacement(drop_list,3)
        nb_filter_1 = 64
        kernel_size_1 = 3
        pool_size_1 = 3
        nb_filter_2 = 64
        kernel_size_2 = 3
        pool_size_2 = 3
        nb_filter_3 = 64
        kernel_size_3 = 3
        pool_size_3 = 3
         # Définition des listes de resultats
        accurancy_train = []
        loss_train = []
        accurancy_val_train = []
        loss_val_train = []
        accurancy_test = []
        loss_test = []

        for drops in drop_comb:
            text_start = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'expérience \n-----------------------------------------------------\n \nLes paramètres de cette experience sont: \n* {} classes \n* {} neurones \n* {} epoque(s) \n* batch_size : {} \n* nb_filter_1 : {} \n* kernel_size_1 : {} \n* pool_size_1 : {} \n* drop1 : {} \n* nb_filter_2 : {} \n* kernel_size_2 : {} \n* pool_size_2 : {} \n* drop2 : {} \n* nb_filter_3 : {} \n* kernel_size_3 : {} \n* pool_size_3 : {} \n* drop3 : {} \n ".format(nb_class,nb_neurone,number_of_epochs,batch_size,nb_filter_1,kernel_size_1,pool_size_1,drops[0],nb_filter_2,kernel_size_2,pool_size_2,drops[1],nb_filter_3,kernel_size_3,pool_size_3,drops[2])
            report_file.write(text_start)
            start = time.perf_counter() # temps de début de l'entrainement 
            modele= models.mod_4(nb_coord,nb_class,nb_filter_1 = nb_filter_1, kernel_size_1 = kernel_size_1, pool_size_1 = pool_size_1, nb_drop1 =drops[0],nb_filter_2 = nb_filter_2, kernel_size_2 = kernel_size_2, pool_size_2 = pool_size_2, nb_drop2 =drops[1],nb_filter_3 = nb_filter_3, kernel_size_3 = kernel_size_3, pool_size_3 = pool_size_3, nb_drop3 =drops[2],fct_activation = 'relu',nb_neurone = nb_neurone)
            modele.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
            history = modele.fit([x_train,x_train,x_train],
                                y_train_hot,
                                batch_size=batch_size,
                                epochs=number_of_epochs,
                                validation_split=0.2,
                                verbose=0)
            report_file.write("\nEntrainement terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n* validation_accurancy = {}\n* validation_loss = {}\n".format(history.history['accuracy'][-1],history.history['loss'][-1],history.history['val_accuracy'][-1],history.history['val_loss'][-1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_train.append(history.history['accuracy'][-1])
            loss_train.append(history.history['loss'][-1])
            accurancy_val_train.append(history.history['val_accuracy'][-1])
            loss_val_train.append(history.history['val_loss'][-1])
            # -------------
            history_test = modele.evaluate([x_test,x_test,x_test],y_test_hot)
            # Temps ecoulé
            end = time.perf_counter()  # temps de fin
            report_file.write("\nTest terminé\n")
            results = "Les resultats sont les suivants:\n* Accurancy = {}\n* Loss = {}\n".format(history_test[1],history_test[1])
            report_file.write(results)
            # -------------
            # Ajout au listes
            # -------------
            accurancy_test.append(history_test[1])
            loss_test.append(history_test[0])
            # -------------
            report_file.write("----------------------------------------------------------------------\n")
            minute = round(end-start) // 60
            secondes = round(end-start) % 60
            report_file.write(f"Temps totale écoulé durant le test: {minute} m et {secondes} s.\n \n")
            text_start = "\n-----------------------------------------------------\n \n[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Resultat de l'experience sur le nombre de drop \n"
        report_file.write(text_start)
        report_file.write(str(drop_list)+ '\n')
        report_file.write(str(accurancy_train) + '\n')
        report_file.write(str(loss_train)+ '\n')
        report_file.write(str(accurancy_val_train)+ '\n')
        report_file.write(str(loss_val_train)+ '\n')
        report_file.write(str(accurancy_test)+ '\n')
        report_file.write(str(loss_test)+ '\n')
        
        report_file.close()

class read_test():

    def txt2list(txt):
            list = txt.split(', ')
            list[0] =list[0].replace('[','')
            list[-1] =list[-1].replace(']','')
            list = [float(elem) for elem in list]  
            return list

    def nb_neurone(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de neurones" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_epoque(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre d'epoque" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_paquet(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de paquet" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_filtre(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de filtres" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_noyaux(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de noyaux" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test

    def nb_pool(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de pool" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test

    def nb_drop(nb_model,M,Re):
        mainFileName = pre_process_CNN.createMainFile_CNN('results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de drop" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]
            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
        
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test





