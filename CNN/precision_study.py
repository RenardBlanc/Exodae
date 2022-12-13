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


def predicted_class(nb_mod,M,Re,ecart_class):
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
        if ecart[i]>ecart_class:
            necorrespondpas +=1
    
    print(int(1-(necorrespondpas/len(ecart))*100))


x,ally,nom_profil,marchepas = format.coordinate()
Re_list=[50000,100000,200000,500000,1000000]
nb_class = 87
finesse_max = np.zeros((len(nom_profil),len(Re_list)))
no_data_all = [] 
for j in range(len(Re_list)):
    Re = Re_list[j]
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
            finesse_max[i,j] = np.max(finesse)
        except:
            no_data.append(name)
    no_data_all.append(no_data)
finesse_max = finesse_max.round(1).T

# M = 0, Re = 50000
ally_0_50000 = ally.copy()
nom_profil_0_50000 = nom_profil.copy()
finesse_max_0_50000 = list(finesse_max[0])
z = [False for _ in range(len(nom_profil_0_50000))]
for nom in no_data_all[0]:
    index = nom_profil.index(nom)
    z[index] = True
    finesse_max_0_50000.pop(index)
    nom_profil_0_50000.pop(index)
ally_0_50000 = ally_0_50000.compress(np.logical_not(z), axis = 1)


#predicted_class(1,0,50000)

Re_fin = {'nom' : nom_profil_0_50000, 
         'finesse_max' : finesse_max_0_50000}

df_fin = pd.DataFrame(Re_fin)

intervalle_finesse_max = jk.jenks_breaks(df_fin['finesse_max'], n_classes=nb_class)
df_fin['classe'] = pd.cut(df_fin['finesse_max'],
                    bins=intervalle_finesse_max,
                    labels=[i for i in range(1,nb_class+1)],
                    include_lowest=True)

print(intervalle_finesse_max)