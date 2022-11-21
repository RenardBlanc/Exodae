from CNN.approx_finesse_CNN import *
import os
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

def test_fin(M,Re,number_of_epochs_test = 500):

    x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
    
    report_file_path = os.path.join(r'CNN\results', 'mod_4_M_{}_Re_{}.txt'.format(M,Re))
    print('ok')
    report_file = open(report_file_path, "a")
    # Nombre de coordonnées et de profils
    nb_coord = np.shape(x_train)[1]
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

test_fin(0,50000,number_of_epochs_test = 500)