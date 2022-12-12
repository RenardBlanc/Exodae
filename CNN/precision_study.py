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


def predicted_class(nb_mod,M,Re):
    x_train,y_train,y_train_hot,x_test,y_test,y_test_hot,nb_classes = pre_process_CNN.data_CNN(M,Re)
    # Charger le modèle enregistré dans le fichier h5
    model = tf.keras.models.load_model('CNN/model/' + 'mod_{}_{}_{}.h5'.format(nb_mod,M,Re))
    # Fonction pour prédire la classe d'un exemple
    prediction = model.predict(x_test)
    # Récupérer la classe avec la plus grande probabilité
    predicted_class = tf.argmax(prediction, axis=1).numpy()
    print(min(y_train),max(y_train),min(predicted_class),max(predicted_class))
    print(np.abs(y_train-predicted_class))

predicted_class(1,0,50000)