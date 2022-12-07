# Librairies
import os
import sys
from datetime import datetime
from itertools import combinations_with_replacement
import logging as lg



class read_test():

    def txt2list(txt):
            list = txt.split(', ')
            list[0] =list[0].replace('[','')
            list[-1] =list[-1].replace(']','')
            list = [float(elem) for elem in list]  
            return list

    def nb_neurone(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de neurones" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_epoque(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre d'epoque" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_paquet(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de paquet" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_filtre(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de filtres" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test
    
    def nb_noyaux(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de noyaux" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test

    def nb_pool(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de pool" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]

            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
            
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test

    def nb_drop(nb_model,M,Re):
        mainFileName = os.path.join('CNN','results')
        report_file_path = os.path.join(mainFileName, 'mod_{}_M_{}_Re_{}.txt'.format(nb_model,M,Re))
        indice_result = []
        try :
            with open(report_file_path) as test_file:
                datafile = test_file.readlines()
            for i in range(len(datafile)):
                if "Resultat de l'experience sur le nombre de drop" in datafile[i]:
                    indice_result.append(i)  
            
            i = indice_result[0]
            data = read_test.txt2list(datafile[i+1])
            accurancy_train = read_test.txt2list(datafile[i+2])
            loss_train = read_test.txt2list(datafile[i+3])
            accurancy_val_train = read_test.txt2list(datafile[i+4])
            loss_val_train = read_test.txt2list(datafile[i+5])
            accurancy_test = read_test.txt2list(datafile[i+6])
            loss_test = read_test.txt2list(datafile[i+7])
        except:
            lg.error(error)
        
        return data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test







# Fonction de lecture des fichiers de résultats

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
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)
        #  On accéde au données des fichiers textes
        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_neurone(nb_mod,M,Re)
        index_max = accurancy_test.index(max(accurancy_test))
        best_param.append(int(data[index_max]))

    if 'nb_epoque' in hyper_param:
        type_param = 'nb_epoque'
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)
        #  On accéde au données des fichiers textes
        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_epoque(nb_mod,M,Re)
        index_max = accurancy_test.index(max(accurancy_test))
        best_param.append(int(data[index_max]))
    if 'nb_paquet' in hyper_param:
        type_param = 'nb_paquet'
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)
        #  On accéde au données des fichiers textes
        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_paquet(nb_mod,M,Re)
        index_max = accurancy_test.index(max(accurancy_test))
        best_param.append(int(data[index_max]))
    if 'nb_pool' in hyper_param:
        type_param = 'nb_pool'
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)
        #  On accéde au données des fichiers textes
        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_pool(nb_mod,M,Re)
        index_max = accurancy_test.index(max(accurancy_test))
        best_param.append(int(data[index_max]))
    if 'nb_filtre' in hyper_param:
        type_param = 'nb_filtre'
        nb_param = nb_filtr
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)

        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_filtre(nb_mod,M,Re)
        # On crée la liste des combinaison
        filtr_comb = list(combinations_with_replacement(data,nb_param))
        index_max = accurancy_test.index(max(accurancy_test))
        for i in range(nb_param):
            best_param.append(int(filtr_comb[index_max][i]))
    if 'nb_noyau' in hyper_param:
        type_param = 'nb_noyau'
        nb_param = nb_noyau
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)

        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_noyaux(nb_mod,M,Re)
        # On crée la liste des combinaison
        filtr_comb = list(combinations_with_replacement(data,nb_param))
        index_max = accurancy_test.index(max(accurancy_test))
        for i in range(nb_param):
            best_param.append(int(filtr_comb[index_max][i]))
    if 'nb_drop' in hyper_param:
        type_param = 'nb_drop'
        nb_param = nb_drop
        text = "[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Début de l'analyse des résultats pour {} \n-----------------------------------------------------\n ".format(type_param)
        print(text)

        data,accurancy_train,loss_train, accurancy_val_train, loss_val_train,accurancy_test,loss_test = read_test.nb_drop(nb_mod,M,Re)
        # On crée la liste des combinaison
        filtr_comb = list(combinations_with_replacement(data,nb_param))
        index_max = accurancy_test.index(max(accurancy_test))
        for i in range(nb_param):
            best_param.append(float(filtr_comb[index_max][i]))

    return best_param




print(get_all_best_param(3,0,1000000))
