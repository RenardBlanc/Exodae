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
    print(min(y_test),max(y_test),min(predicted_class),max(predicted_class))
    ecart = np.abs(y_test-predicted_class)
    
    dossierparent = os.path.join('CNN','results')
    mainFileName = pre_process_CNN.createMainFile_CNN('figure',bigfolder = dossierparent)
    nom_figure = os.path.join(mainFileName, 'predict_mod{}_M{}_Re{}'.format(nb_mod,M,Re))
    plt.figure(figsize = (12,8))
    plt.hist(ecart)
    plt.savefig(nom_figure)
    plt.close()
    necorrespondpas = 0
    for i in range(len(ecart)):
        if ecart[i]>7:
            necorrespondpas +=1
    
    print(int(necorrespondpas/len(ecart)*100))


predicted_class(1,0,50000)