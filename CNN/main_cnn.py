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
        p  = len(y_inter)
        x_test = np.zeros((p,1))
        for i in range(len(y_inter)):
            x_test[i,1] = y_inter[i]
        # Fonction pour prédire la classe d'un exemple
        if nb_mod==4:
            prediction = model.predict([x_test,x_test,x_test])
        else : 
            prediction = model.predict(x_test)
        # Récupérer la classe avec la plus grande probabilité
        predicted_class = tf.argmax(prediction, axis=1).numpy()
        fin_max_pred = predict_cnn.class2fin(predicted_class,nb_class,M,Re)
        incertitude = max_ecart_pos(M,Re,nb_class,3)
        
        return fin_max_pred,incertitude
    

if __name__ == '__main__':
    if len(sys.argv) == 3:
        M = 0
        Re = 50000
        nb_class = 87
        file_name = str(sys.argv[1]) 
        nb_mod = int(sys.argv[2]) 
        fin_max_pred,incertitude = predict_cnn.predict_class(file_name,nb_mod,M,Re,nb_class,dir=r"data/Airfoil_Coordinate")
        fin_d = round(fin_max_pred+incertitude,1)
        fin_g = round(fin_max_pred-incertitude,1)
        print("On retrouve avec le modèle {}, une finesse entre [{},{}]".format(nb_mod,fin_g,fin_d))
    elif len(sys.argv) == 2:
        M = 0
        Re = 50000
        nb_class = 87
        file_name = str(sys.argv[1]) 
        for nb_mod in range(1,5):
            fin_max_pred,incertitude = predict_cnn.predict_class(file_name,nb_mod,M,Re,nb_class,dir=r"data/Airfoil_Coordinate")
            fin_d = round(fin_max_pred+incertitude,1)
            fin_g = round(fin_max_pred-incertitude,1)
            print("On retrouve avec le modèle {}, une finesse entre [{},{}]\n".format(nb_mod,fin_g,fin_d))
    else:
        raise Exception(
            'Entrer <File name> <Nb_Model>')