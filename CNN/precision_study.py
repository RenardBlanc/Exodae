'''
------Etude de précision des modèles------------
Les modèles ont été entrainé et nous avons un résultat de 
précision par rapport à la classe. Cependant, il est interessant de
voir l'ecart avec la finesse réelle.
Created: 23/11/2022
Updated: 23/11/2022
@Auteur: Ilyas Baktache
'''

# ***********
# Librairies
# ***********

# File managment 
import os

# Import from other class
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from approx_finesse_CNN import *

def mod_1(M,Re):
    nb_mod = 1
    x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_class = pre_process_CNN.data_CNN(M,Re)
    # Charger le modèle enregistré dans le fichier h5
    model = tf.keras.models.load_model('CNN/model/' + 'mod_{}_{}_{}.h5'.format(nb_mod,M,Re))
    # Utiliser le modèle pour faire une prédiction
    predictions = model.predict(x_test[0])

    print(predictions)

mod_1(0,50000)