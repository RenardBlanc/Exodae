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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Reshape, GlobalAveragePooling1D, LeakyReLU, Input,concatenate, Embedding,LeakyReLU,Conv1DTranspose
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
        x_coord_initial,ally,nom_profil,marchepas = format.coordinate(nb_point = 31, nb_LE = 20, nb_TE = 10)
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
                # Ici on choisit alpha = 0
                try :
                    alpha,cL,cD,cDp,cM = utils.polarFile2list(name,M,Re)
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
            classe_list = list(np.array(list(df_fin['classe'])) - 1)

            return classe_list

        for i in range(len(Re_list)):
            finesse_max_classe.append(list_label(nom_profil_tt_Re[i],finesse_max[i],nb_class_list[i]))
        
        def save_Re_data_GAN(dict):
            mainFileName = pre_process_GAN.createMainFile_GAN('post_processed_data_GAN')
            Re = dict['reynoldsNumber']
            name = os.path.join(mainFileName,"Re_{}_{}.pickle".format(M,Re))
            with open(name, "wb") as tf:
                pickle.dump(dict,tf)

        dict_0_50000 = {'x_train' : ally_0_50000,
                        'y_train' : finesse_max_classe[0],
                        'nb_classe' : nb_class_list[0],
                        'nom_profil' : nom_profil_tt_Re[0],
                        'reynoldsNumber' : 50000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_50000)

        dict_0_100000 = {'x_train' : ally_0_100000,
                        'y_train' :finesse_max_classe[1],
                        'nb_classe' : nb_class_list[1],
                        'nom_profil' : nom_profil_tt_Re[1],
                        'reynoldsNumber' : 100000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_100000)

        dict_0_200000 = {'x_train' : ally_0_200000,
                        'y_train' :finesse_max_classe[2],
                        'nb_classe' : nb_class_list[2],
                        'nom_profil' : nom_profil_tt_Re[2],
                        'reynoldsNumber' : 200000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_200000)

        dict_0_500000 = {'x_train' : ally_0_500000,
                        'y_train' :finesse_max_classe[3],
                        'nb_classe' : nb_class_list[3],
                        'nom_profil' : nom_profil_tt_Re[3],
                        'reynoldsNumber' : 500000,
                        'coordonnee_x' : x_coord_initial
                        }
        save_Re_data_GAN(dict_0_500000)
        
        dict_0_1000000 = {'x_train' : ally_0_1000000,
                        'y_train' : finesse_max_classe[4],
                        'nb_classe' : nb_class_list[4],
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
        y_train = np.array(dict_ok['y_train']).astype('int')
        nb_class =  np.array(dict_ok['nb_classe'])
        x_coord_ini = np.array(dict_ok['coordonnee_x']).astype('float32')
        return x_train,y_train,nb_class,x_coord_ini

    def generate_real_samples(x_train,y_train,n_samples):
        nb_profil = x_train.shape[1]
        # choose random instances
        ix = np.random.randint(0, nb_profil, n_samples)
        # select images and labels
        X, labels = x_train[:,ix], y_train[ix]
        # generate class labels
        y = np.ones((n_samples, 1))
        return [X, labels], y
    
    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples, n_classes=10):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate labels
        labels = np.random.randint(0, n_classes, n_samples)
        return [z_input, labels]
        

class model():

    def discriminateur(nb_coord, nb_class):
        quart = int(nb_coord/4)
        discriminateur = Sequential()
        #Head 1
        in_coord = Input(shape = (nb_coord,))
        resh1 = Reshape((nb_coord,1))(in_coord)
        #Head 2
        in_label = Input(shape = (1,))
        emb2 = Embedding(1,nb_class)(in_label)
        dense2 = Dense(nb_coord, activation='relu')(emb2)
        resh2 = Reshape((nb_coord,1))(dense2)
        # merge
        conc1 = concatenate([resh1, resh2])
        conv1 = Conv1D(filters=128, kernel_size=quart+1, activation='relu')(conc1)
        fct1 = LeakyReLU(alpha = 0.2)(conv1)
        conv2 = Conv1D(filters=128, kernel_size=quart*2+1, activation='relu')(fct1)
        fct2 = LeakyReLU(alpha = 0.2)(conv2)
        flat3 = Flatten()(fct2)
        drop3 = Dropout(0.5)(flat3)
        output = Dense(1, activation='relu')(drop3)

        discriminateur = Model([in_coord, in_label], output)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        discriminateur.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return discriminateur

    def generateur(nb_coord,latent_dim,nb_class):
        quart = int(nb_coord/4)
        generateur = Sequential()
        #Head 1
        in_lat = Input(shape = (latent_dim,))
        dense1 = Dense(32*nb_coord, activation='relu')(in_lat)
        fct1 = LeakyReLU(alpha = 0.2)(dense1)
        resh1 = Reshape((quart,128))(fct1)
        #Head 2
        in_lab = Input(shape = (1,))
        emb2 = Embedding(1,nb_class)(in_lab)
        dense2 = Dense(quart, activation='relu')(emb2)
        resh2 = Reshape((quart,1))(dense2)

        # merge
        conc1 = concatenate([resh1, resh2])
        convt1 = Conv1DTranspose(filters=128, kernel_size=quart+1)(conc1)
        fct2 = LeakyReLU(alpha = 0.2)(convt1)
        convt2 = Conv1DTranspose(filters=128, kernel_size=2*quart+1)(fct2)
        fct3 = LeakyReLU(alpha = 0.2)(convt2)
        output = Conv1D(filters=1, kernel_size=1,activation='tanh', padding='same')(fct3)

        generateur = Model(inputs = [in_lat,in_lab],outputs = output)
        return generateur

    
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(generator, latent_dim, n_samples):
        # generate points in latent space
        z_input, labels_input = pre_process_GAN.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict([z_input, labels_input])
        # create class labels
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y
    

    def gan(d_model,g_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get coord output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
        
    def train_model(Mach,Re,x_train,y_train,latent_dim,g_model,d_model,gan_model, nb_epoch = 100, nb_batch = 50):
        
        # Import des données de profils 
        nb_coord =  np.shape(x_train)[0]
        nb_profil = np.shape(x_train)[1]

        nb_batch_per_epoch = int(nb_profil/nb_batch)
        half_batch = int(nb_batch / 2)
        # manually enumerate epochs
        for i in range(nb_epoch):
            # enumerate batches over the training set
            for j in range(nb_batch_per_epoch):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = pre_process_GAN.generate_real_samples(x_train,y_train,half_batch)
                # update discriminator model weights
                d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake =  model.generate_fake_samples(g_model, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] =  pre_process_GAN.generate_latent_points(latent_dim, nb_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((nb_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, nb_batch_per_epoch, d_loss1, d_loss2, g_loss))
        # save the generator model
        name = 'GAN/cgan_generator_{}_{}_{}.h5'.format(Mach,Re,latent_dim)
        if os.path.exists(name):
            os.remove(name)
        g_model.save(name)

if __name__ == "__main__":
    # ----------
    # Paramètres 
    # ----------
    if len(sys.argv) != 6:
        raise Exception(
            'Entrer <Nombre de Mach> <Nombre de Reynolds><Dimension latente> <BATCH_SIZE> <EPOCHS> ')
    else: 
        Mach = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        latent_dim = int(sys.argv[3]) 
        batch_size = int(sys.argv[4])
        epoch = int(sys.argv[5]) 
    
    # Import des données de profils 
    x_train,y_train,nb_class,x_coord_ini = pre_process_GAN.data_GAN(Mach,Re) # Nombre de coordonnées et de profils
    nb_coord = np.shape(x_train)[0]
    # create the discriminator
    d_model = model.discriminateur(nb_coord, nb_class)
    # create the generator
    g_model = model.generateur(nb_coord,latent_dim,nb_class)
    # create the gan
    gan_model = model.gan(d_model, g_model)
    model.train_model(Mach,Re,x_train,y_train,latent_dim,g_model,d_model,gan_model, nb_epoch = epoch, nb_batch = batch_size)
