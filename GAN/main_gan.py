from GAN_Class import *

if __name__ == "__main__":
    # ----------
    # Paramètres 
    # ----------
    if len(sys.argv) != 7:
        raise Exception(
            'Entrer <Nb_Model> <Nombre de Mach> <Nombre de Reynolds> <BATCH_SIZE> <EPOCHS> <Plot>')
    else:
        nb_model = int(sys.argv[1]) 
        mach = int(sys.argv[2]) 
        nb_Re = int(sys.argv[3]) 
        batch_size = int(sys.argv[4])
        epoch = int(sys.argv[5]) 
        plot_bool = sys.argv[6]

    models.train_model(nb_model,mach,nb_Re,BATCH_SIZE = batch_size,EPOCHS = epoch,plot=plot_bool)
