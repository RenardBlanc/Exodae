
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


def generateur_prediction(Mach,Re,classe,latent_dim = 100):
    model  = tf.keras.models.load_model('GAN/cgan_generator_{}_{}_{}.h5'.format(Mach,Re,latent_dim))
    # generate images
    latent_points, labels = pre_process_GAN.generate_latent_points(latent_dim, 1)
    # specify labels
    labels = np.zeros((1,1))
    labels[0,0] = classe
    # generate images
    X  = model.predict([latent_points, labels])
    return X[0]

def rolling_mean(data, window_size):
  data_mean = np.empty(data.size)
  for i in range(data.size):
    # Calculate the mean of the surrounding data points
    data_mean[i] = np.mean(data[i:i+window_size])
  return data_mean

def plot_profil(coord_y,M,Re,etat):
    x_train,y_train,nb_class,x_coord_ini = pre_process_GAN.data_GAN(M,Re) # Nombre de coordonnées et de profils
    mainFileName = pre_process_GAN.createMainFile_GAN('figure')
    nom_figure = os.path.join(mainFileName, 'generated{}_M{}_Re{}'.format(etat,M,Re))
    plt.figure(figsize = (12,8))
    plt.plot(x_coord_ini,coord_y)
    plt.title("Generated airfoil with GAN")
    plt.savefig(nom_figure)

def generate_profil(classe,M,Re):
    
    coord_y_generated = generateur_prediction(M,Re,classe,latent_dim = 100)
    plot_profil(coord_y_generated,M,Re)

    coord_y_liss = rolling_mean(coord_y_generated, 3)
    plot_profil(coord_y_liss,M,Re)




if __name__ == '__main__':
    if len(sys.argv) == 3:
        M = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        classe = int(sys.argv[3]) 
        generate_profil(classe,M,Re)
    else:
        raise Exception(
            'Entrer <Nb_Mach> <Nb_Re> <Nb_class>')