
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


def generate_profil(Mach,Re,classe,latent_dim = 100):
    model  = tf.keras.models.load_model('GAN/cgan_generator_{}_{}_{}.h5'.format(Mach,Re,latent_dim))
    # generate images
    latent_points, labels = pre_process_GAN.generate_latent_points(100, 1)
    # specify labels
    labels = np.zeros((1,1))
    labels[0,0] = classe
    # generate images
    X  = model.predict([latent_points, labels])
    return X[0]

def plot_profil(coord_y,M,Re):
    x_train,y_train,nb_class,x_coord_ini = pre_process_GAN.data_GAN(M,Re) # Nombre de coordonnées et de profils
    mainFileName = pre_process_GAN.createMainFile_GAN('figure')
    nom_figure = os.path.join(mainFileName, 'generated_M{}_Re{}'.format(M,Re))
    plt.figure(figsize = (12,8))
    plt.plot(x_coord_ini,coord_y)
    plt.title("Generated airfoil with GAN")
    plt.savefig(nom_figure)

classe = 12
M = 0
Re = 50000
coord_y = generate_profil(M,Re,classe,latent_dim = 100)
plot_profil(coord_y,M,Re)