from approx_finesse_CNN import *

def link_model_test(nb_model):
        if nb_model == 1:
            model = test_param_CNN.mod_1
        elif nb_model == 2:
            model = test_param_CNN.mod_2
        elif nb_model == 3:
            model = test_param_CNN.mod_3
        elif nb_model == 4:
            model = test_param_CNN.mod_4
        return model 

if __name__ == "__main__":
    # ----------
    # Paramètres 
    # ----------
    if len(sys.argv) == 4:
        Mach = int(sys.argv[1]) 
        Re = int(sys.argv[2]) 
        nb_epoch_test = int(sys.argv[3]) 
        if nb_epoch_test ==0:
            test_param_CNN.mod_1(Mach,Re)
            test_param_CNN.mod_2(Mach,Re)
            test_param_CNN.mod_3(Mach,Re)
            test_param_CNN.mod_4(Mach,Re)
        else:
            test_param_CNN.mod_1(Mach,Re,number_of_epochs_test=nb_epoch_test)
            test_param_CNN.mod_2(Mach,Re,number_of_epochs_test=nb_epoch_test)
            test_param_CNN.mod_3(Mach,Re,number_of_epochs_test=nb_epoch_test)
            test_param_CNN.mod_4(Mach,Re,number_of_epochs_test=nb_epoch_test)

    elif len(sys.argv) == 5:
        nb_model = int(sys.argv[1]) 
        Mach = int(sys.argv[2]) 
        Re = int(sys.argv[3]) 
        nb_epoch_test = int(sys.argv[4]) 
        
        test_model = link_model_test(nb_model)
        if nb_epoch_test ==0:
            test_model(Mach,Re)
        else: 
            test_model(Mach,Re,number_of_epochs_test=nb_epoch_test)
    
    else:
        raise Exception(
            'Entrer <Nb_Model> <Nb_Mach> <Nb_Re> <Nb_epoch_test>')
        

    
