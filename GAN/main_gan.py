
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
    return X

classe = 12
print(generate_profil(Mach,Re,classe,latent_dim = 100))