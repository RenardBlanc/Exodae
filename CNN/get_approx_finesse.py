'''
------Obtenir l'approximation de la finesse d'un profil------------
Ce code Obtenir l'approximation de la finesse d'un profil
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
from precision_study import *



class predict_cnn():

    def interpole_profil(file_name,dir=r"data/Airfoil_Coordinate"):
        '''
        Cette fonction permet d'interpoler les fichier de 
        coordonnées des profils.
        '''
        x_inter,y_inter = format.rawtointer(file_name, dir=dir,nb_point = 30, nb_LE = 20, nb_TE = 10,x_rang_LE = 0.15, x_rang_TE = 0.75)
        return x_inter,y_inter

    def class2fin(classe,nb_class,M,Re):
        '''
        Cette fonction permet de passer d'une classe 
        à une finesse maximale avec l'incertitude résulante
        '''
        intervalle_finesse_max = get_ecart(M,Re,nb_class)

        return intervalle_finesse_max[nb_class-1]


    def predict_class(file_name,nb_mod,M,Re,nb_class,dir=r"data/Airfoil_Coordinate"):
        x_inter,y_inter = predict_cnn.interpole_profil(file_name,dir=r"data/Airfoil_Coordinate")
        model = tf.keras.models.load_model('CNN/model/' + 'mod_{}_{}_{}.h5'.format(nb_mod,M,Re))

        # Fonction pour prédire la classe d'un exemple
        if nb_mod==4:
            prediction = model.predict([y_inter,y_inter,y_inter])
        else : 
            prediction = model.predict(y_inter)
        # Récupérer la classe avec la plus grande probabilité
        predicted_class = tf.argmax(prediction, axis=1).numpy()
        fin_max_pred = predict_cnn.class2fin(predicted_class,nb_class,M,Re)
        incertitude = max_ecart_pos(M,Re,nb_class,3)
        
        return fin_max_pred,incertitude
    