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
    predicted_class = tf.argmax(prediction, axis=1)
    return predicted_class.numpy()

def class2finesse(M,Re):
    dict_ok = pre_process_CNN.get_data_pre_process_CNN(M,Re)
    
    # Nom des profils dans cette base de donnée
    nom_profil_Re =  np.array(dict_ok['nom_profil']).astype('int')
    # Nombre de classes à définir
    nb_class = dict_ok['nb_classe']

    # Les finesses
    x,ally,nom_profil,marchepas = format.coordinate()
    # On note dans cette liste les finesses maximales
    finesse_max = np.zeros((len(nom_profil),1))
    no_data_all = [] 

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
            finesse_max[i,0] = np.max(finesse)
        except:
            no_data.append(name)
    finesse_max = finesse_max.round(1).T

    finesse_max_classe = []
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

    df_fin, intervalle_finesse_max = discretisation_label(nom_profil_Re,finesse_max,nb_class)
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

    print(df_fin)