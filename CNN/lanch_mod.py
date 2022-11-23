'''
------Lancement des différents moodèle------------
Ce module regroupe diverses fonctions qui permettent 
    * Afficher les résultats des experiences sur les 
    hyper-paramètres
    * Lancer les modèles avec les meilleurs hyper-paramètres
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


# ***********
# Analyse des résultats
# ***********
class result:
    def get_all_best_param(nb_mod,M,Re):
        
        if nb_mod == 1:
            hyper_param = ['nb_neuron','nb_epoque','nb_paquet']
        elif nb_mod == 2:
            hyper_param = ['nb_neuron','nb_epoque','nb_paquet','nb_filtre','nb_noyau','nb_pool']
            nb_filtr = 2
            nb_noyau = 2
        elif nb_mod == 3:
            hyper_param = ['nb_neuron','nb_epoque','nb_paquet','nb_filtre','nb_noyau','nb_drop']
            nb_filtr = 3
            nb_noyau = 3
            nb_drop = 4
        elif nb_mod == 4:
            hyper_param = ['nb_neuron','nb_epoque','nb_paquet','nb_filtre','nb_noyau','nb_drop']
            nb_filtr = 3
            nb_noyau = 3
            nb_drop = 3
        else:
            error
        
        # Listes des meilleurs paramètres
        best_param = []
        
        if 'nb_neuron' in hyper_param:
            type_param = 'nb_neuron'
            #  On accéde au données des fichiers textes
            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_neurone(nb_mod,M,Re)
            index_max = accurancy_test.index(max(accurancy_test))
            best_param.append(int(data[index_max]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter(data,np.round(np.array(accurancy_test)*100,0))
            plt.xlabel('Nombre de neurone')
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_epoque' in hyper_param:
            type_param = 'nb_epoque'
            #  On accéde au données des fichiers textes
            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_epoque(nb_mod,M,Re)
            index_max = accurancy_test.index(max(accurancy_test))
            best_param.append(int(data[index_max]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter(data,np.round(np.array(accurancy_test)*100,0))
            plt.xlabel("Nombre d'époque")
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_paquet' in hyper_param:
            type_param = 'nb_paquet'
            #  On accéde au données des fichiers textes
            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_paquet(nb_mod,M,Re)
            index_max = accurancy_test.index(max(accurancy_test))
            best_param.append(int(data[index_max]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter(data,np.round(np.array(accurancy_test)*100,0))
            plt.xlabel('Nombre de paquet')
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_pool' in hyper_param:
            type_param = 'nb_pool'
            #  On accéde au données des fichiers textes
            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_pool(nb_mod,M,Re)
            index_max = accurancy_test.index(max(accurancy_test))
            best_param.append(int(data[index_max]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter(data,np.round(np.array(accurancy_test)*100,0))
            plt.xlabel('Nombre de Pool size')
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_filtre' in hyper_param:
            type_param = 'nb_filtre'
            nb_param = nb_filtr

            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_filtre(nb_mod,M,Re)
            # On crée la liste des combinaison
            filtr_comb = list(combinations_with_replacement(data,nb_param))
            index_max = accurancy_test.index(max(accurancy_test))
            for i in range(nb_param):
                best_param.append(int(filtr_comb[index_max][i]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter([i for i in range(len(accurancy_train))],np.round(np.array(accurancy_test)*100,0))
            plt.xlabel("Index de la combinaison du nombre de filtre")
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_noyau' in hyper_param:
            type_param = 'nb_noyau'
            nb_param = nb_noyau

            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_noyaux(nb_mod,M,Re)
            # On crée la liste des combinaison
            filtr_comb = list(combinations_with_replacement(data,nb_param))
            index_max = accurancy_test.index(max(accurancy_test))
            for i in range(nb_param):
                best_param.append(int(filtr_comb[index_max][i]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter([i for i in range(len(accurancy_train))],np.round(np.array(accurancy_test)*100,0))
            plt.xlabel("Index de la combinaison du nombre de noyau")
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)
        
        if 'nb_drop' in hyper_param:
            type_param = 'nb_drop'
            nb_param = nb_drop

            data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_drop(nb_mod,M,Re)
            # On crée la liste des combinaison
            filtr_comb = list(combinations_with_replacement(data,nb_param))
            index_max = accurancy_test.index(max(accurancy_test))
            for i in range(nb_param):
                best_param.append(int(filtr_comb[index_max][i]))
            # Nom des fichiers de figures
            mainFileName = pre_process_CNN.createMainFile_CNN('results')
            sub_mainFileName = os.path.join(mainFileName, 'figures')
            nom_figure = os.path.join(sub_mainFileName, 'mod_{}_{}_M_{}_Re_{}'.format(nb_mod,type_param,M,Re))
            # Tracé des figures
            plt.figure(figsize = (12,8))
            plt.scatter([i for i in range(len(accurancy_train))],np.round(np.array(accurancy_test)*100,0))
            plt.xlabel("Index de la combinaison du coefficient de drop")
            plt.ylabel('Précision (%)')
            plt.savefig(nom_figure)


if __name__ == '__main__':
    # put main folder
    result.get_all_best_param(1,0,50000)
