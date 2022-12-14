
'''
------Géneration de profil avec un GAN------------
Ce module regroupe diverses fonctions et classes qui permettent 
de génerer des profils d'avion avec un GAN

Created: 09/11/2022
Updated: 09/11/2022
@Auteur: Ilyas Baktache
'''


# File managment 
import os

# Import from other class
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from GAN.train_gan import *


class label_gan():
    def classe2data_predict(classe,M,Re,mod):
        dict_ok = pre_process_GAN.get_data_pre_process_GAN(M,Re,mod)

        nom_profil = np.array(dict_ok['nom_profil']).astype('str')

        if mod == 'fin' and type == 1:
            donnee =  np.array(dict_ok['finesse_max']).astype('float')
            nb_class  =  int(dict_ok['nb_classe_fin'])

        elif mod == 'fin' and type == 2:
            finesse =  np.array(dict_ok['finesse_max']).astype('float')
            aire = np.array(dict_ok['aire']).astype('float')
            score = np.sqrt(np.power(aire,2) + np.power(finesse,2))
            donnee = pre_process_GAN.normal_score(score)
            nb_class  =   int(dict_ok['nb_classe_fin_aire'])

        elif mod == 'aire':
            donnee =  np.array(dict_ok['aire']).astype('float')
            nb_class  =  int(dict_ok['nb_classe'])

        df_fin, intervalle = pre_process_GAN.discretisation_label(nom_profil,donnee,nb_class)
        return intervalle[classe]
        
class predict():
    def generateur_prediction(Mach,Re,epoch,classe,latent_dim = 100):
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Prédiction d'un profils avec le modèle enregistré pour une classe de {}".format(classe))

        model  = tf.keras.models.load_model('GAN/cgan_generator_{}_{}_{}_{}.h5'.format(Mach,Re,epoch,latent_dim))
        # generate images
        latent_points, labels = pre_process_GAN.generate_latent_points(latent_dim, 1)
        # specify labels
        labels = np.zeros((1,1))
        
        labels[0,0] = classe
        # generate images
        X  = model.predict([latent_points, labels])
        return X[0]

    def generateur_all_prediction(Mach,Re,nb_class,epoch = 1000,latent_dim = 100):
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Prédiction d'un profils avec le modèle enregistré pour toutes les classe")

        model  = tf.keras.models.load_model('GAN/cgan_generator_{}_{}_{}_{}.h5'.format(Mach,Re,epoch,latent_dim))
        # generate images
        latent_points, labels = pre_process_GAN.generate_latent_points(latent_dim, nb_class)
        # specify labels
        labels = np.zeros((nb_class,1))
        for i in range(nb_class):
            labels[i,0] = i
        # generate images
        X  = model.predict([latent_points, labels])
        return X

    def rolling_mean(data, window_size):
        data_mean = np.empty(data.size)
        for i in range(data.size):
            # Calculate the mean of the surrounding data points
            data_mean[i] = np.mean(data[i:i+window_size])
        return data_mean

    def plot_profil(coord_y,M,Re,classe,mod,type,etat):
        x_train,y_train,nb_class,x_coord_ini = pre_process_GAN.data_GAN(M,Re,mod,type) # Nombre de coordonnées et de profils
        mainFileName = pre_process_GAN.createMainFile_GAN('figures')
        nom_figure = os.path.join(mainFileName, etat + '_{}_{}_{}'.format(M,Re,classe))
        plt.figure(figsize = (12,8))
        plt.plot(x_coord_ini,coord_y)
        plt.title("Generated airfoil with GAN")
        plt.savefig(nom_figure)

    def plot_subplots(n_class,x_coord,all_y_coord,Mach,Re,mod,ncols = 10):
        # Calculer le nombre de lignes en divisant le nombre total de sous-figures par le nombre de colonnes
        nrows = math.ceil(n_class / ncols)
        # Créer une figure et un sous-plot pour chaque entrée de données
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize = (12,8))
        for i, ax in enumerate(axs.flat):
            try:
                # Tracer les données sur le sous-plot
                ax.plot(x_coord,all_y_coord[i])
            except:
                pass    
        mainFileName = pre_process_GAN.createMainFile_GAN('figures_mois/')
        nom_figure = os.path.join(mainFileName, '{}_{}_{}'.format(mod,Mach,Re))
        if os.path.exists(nom_figure):
            os.remove(nom_figure)
        # Afficher la figure
        plt.savefig(nom_figure)

    def plot_mosaique(M,Re,mod,type):
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début du tracé du plot mosaique avec Re = et M = pour le mod = {}".format(Re,M,mod))
        all_y = []
        x_train,y_train,nb_class,x_coord_ini = pre_process_GAN.data_GAN(M,Re,mod,type) # Nombre de coordonnées et de profils
        coord_y_generated = predict.generateur_all_prediction(M,Re,nb_class,latent_dim = 100)
        for i in range(nb_class):
            all_y.append(predict.rolling_mean(coord_y_generated[i],10))
        predict.plot_subplots(nb_class,x_coord_ini,all_y,M,Re,mod,ncols = 10)
        lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] La figure a été enregistré ")


    def generate_profil(classe,M,Re,mod,type,lissage = 10):
        coord_y_generated = predict.generateur_prediction(M,Re,classe,latent_dim = 100)
        coord_y_liss = predict.rolling_mean(coord_y_generated, lissage)
        predict.plot_profil(coord_y_liss,M,Re,classe,mod,type,'liss_{}'.format(lissage))


if __name__ == '__main__':
    if len(sys.argv) == 4:
        M = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        classe = int(sys.argv[3])
        predict.generate_profil(classe,M,Re,'fin',1)

    elif len(sys.argv) == 3:
        M = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        predict.plot_mosaique(M,Re,'aire',1) 
    else:
        raise Exception(
            'Entrer <Nb_Mach> <Nb_Re> <Nb_class>')