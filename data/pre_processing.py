'''
---------------------Pre-Processing and Label extraction---------------------
This module presents different classes and functions in order 
to asses the Pre-processing:
* Scrapping of the Data
* Data set sampling and cleaning
And the Label extraction:
* Lift and Drag coefficient calculation with Xfoil
* Shape Area calculation
* Caracteritic point calculation

Created: 24/05/2022
Updated: 24/05/2022
@Auteur: Ilyas Baktache
'''
# %%
# Librairies
# -------------------------------------------------------------

# Log
import logging as lg
lg.basicConfig(level=lg.INFO)  # Debug and info
from datetime import datetime

# Folder and string
import os
import re
import shutil

# math and plot
import matplotlib.pyplot as plt
import math as m
import numpy as np
from numpy import linalg as lin

# Threading
import threading

# Import other module
from data.scrapping import *
from data.xfoil import *

#Torch
import torch

# sklearn
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

from time import time
class le():

    def polar(name,Re,M,airfoil_polar_not):
        try : 
            xfoil.get_polar(name, Re = Re, Mach = M,dir=r"data/Airfoil_Coordinate")
        except:
            airfoil_polar_not.append(name)
            lg.error("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} coordinate couldn't be scrapped from Airfoil Tools website".format(name))   

    def polar_Re_M(Re,M):

        # Create Airfoil Polar main File if it doesn't exist
        mainFileName = "Airfoil_Polar"  
        mainFileName = utils.createMainFile(mainFileName)
        # Create Airfoil Polar Mach main File if it doesn't exist 
        mainFileName = utils.createMainFile('M_{}'.format(M),bigfolder = mainFileName)
        # Create Airfoil Polar Mach-Reynolds main File if it doesn't exist 
        mainFileName = utils.createMainFile('Re_{}'.format(Re),bigfolder = mainFileName)

        # Scrap all airfoil name in AirfoilTools.com Database
        all_airfoils = scrap.airfoils_name()

        if M == 0:
            # We directly use the data of Airfoil Tools from M = 0 
            scrap.airfoils_polar(all_airfoils,Re,mainFileName)
        else :
            if not os.path.isdir(r'data/Airfoil_Coordinate'):
                # Scrap and save locally coordinate of airfoils
                scrap.airfoils_coordinate(all_airfoils)
                
            # In the case that all the airfoil name from the website 
            # couldn't be scrapped we take only the name of 
            # data that we have
            all_airfoils = os.listdir(r'data/Airfoil_Coordinate') 

            # List of coordinate airfoils that couldn't be scrapped
            airfoil_polar_not = []
            # List all the airfoils name
            n = len(all_airfoils)
            
            # Threading Data
            num_thread = 250 # number of thread
            i0 = 0 # index of the frist page of the multithreading
            i1 = num_thread  # index of the last page of the multithreading
            # Start of threadings
            while i0<n:
                threadears = []
                for i in range(i0, i1):
                    try:
                        name_thread = all_airfoils[i].replace('.dat','')
                        t = threading.Thread(target= le.polar, args=(name_thread,Re,M,airfoil_polar_not))
                        t.start()
                        threadears.append(t)
                    except Exception as err:
                        lg.debug("error:".format(err))
                for threadear in threadears:
                    threadear.join()
                i0 += num_thread
                i1 += num_thread
                if i1 > n:
                    i1 = n
            if len(airfoil_polar_not)>0:
                lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {} airfoil data couldn't be processed in Xfoil:\n {}".format(len(airfoil_polar_not),airfoil_polar_not))
            nb_polar_file = len(os.listdir(mainFileName))
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] {}({}%) airfoils polar for Re = {} were downloaded and saved locally".format(nb_polar_file,int(nb_polar_file/n*100),Re))

    def allPolar(Re_list=[50000,100000,200000,500000,100000],M_list=[0,0.25,0.5,0.75,0.9]):
        '''
        Function that return a polar of all the airfoil in the folder 
        Airfoil_Coordinate for the Re and M wanted
        '''
        if type(M_list)==list:
            for M in M_list:
                if type(Re_list)==list:
                    for Re in Re_list:
                        le.polar_Re_M(Re,M)
                else:
                    le.polar_Re_M(Re_list,M)
        else : 
            if type(Re_list)==list:
                for Re in Re_list:
                    le.polar_Re_M(Re,M_list)
            else:
                le.polar_Re_M(Re_list,M_list)
      
class format():
    '''
    Create an unique format for data and label
    '''
    def raw_format(dir=r"data/Airfoil_Coordinate" ):
        nb_coord_x = []
        nb_coord_y = []
        airfoils = os.listdir(dir)
        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            x,y = utils.coordFile2list(name)
            if x[0] != 1 :
                nb_coord_x.append(name)
            if x[-1] != 1 :
                nb_coord_y.append(name)
        print(len(nb_coord_x),nb_coord_y)
    
    def diff_extra_intra(name,dir=r"data/Airfoil_Coordinate"):

        '''
        Permet de distinguer les coordonnées de l'extrados
        et de l'intrados. 
        '''
        i_extra = 0
        x,y = utils.coordFile2list(name,mainFileName = dir)
        for i in range(len(x)-1):
            if np.abs(x[i])>np.abs(x[i+1]):
                i_extra = i+1
        x_extra = x[:i_extra+1]
        y_extra = y[:i_extra+1]
        x_intra = x[i_extra+1:]
        y_intra = y[i_extra+1:]
        return x_extra,y_extra,x_intra,y_intra
    
    def indice_leading_trailing_edges(x_extra,x_intra,x_rang_LE,x_rang_TE):

        i_extra_TE = []
        i_extra_LE = []
        for i in range(len(x_extra)):
            if x_extra[i]<=x_rang_LE:
                i_extra_LE.append(i)
            if x_extra[i]>=x_rang_TE:
                i_extra_TE.append(i)

        i_intra_TE = []
        i_intra_LE = []
        for i in range(len(x_intra)):
            if x_intra[i]<=x_rang_LE:
                i_intra_LE.append(i)
            if x_intra[i]>=x_rang_TE:
                i_intra_TE.append(i)

        return min(i_extra_LE),max(i_extra_TE),max(i_intra_LE),min(i_intra_TE)

    def rawtointer(name, dir=r"data/Airfoil_Coordinate",nb_point = 30, nb_LE = 20, nb_TE = 10,x_rang_LE = 0.15, x_rang_TE = 0.75):
        '''
        Function that take the raw airfoil data coordinate and return
        '''

        x_extra,y_extra,x_intra,y_intra = format.diff_extra_intra(name,dir=dir)
        i_LE_extra,i_TE_extra,i_LE_intra,i_TE_intra = format.indice_leading_trailing_edges(x_extra,x_intra,x_rang_LE,x_rang_TE)

        # Interpolation extrados LE
        x_extrados_LE = np.linspace(0,x_rang_LE,nb_LE,endpoint = True)[::-1][1:]
        f_extrados_LE= interpolate.interp1d(x_extra[i_LE_extra:], y_extra[i_LE_extra:], fill_value="extrapolate", kind="quadratic")
        y_extrados_LE = f_extrados_LE(x_extrados_LE)

        # Interpolation extrados body
        x_extrados_body = np.linspace(x_rang_LE,x_rang_TE,nb_point,endpoint = True)[::-1][:]
        f_extrados_body= interpolate.interp1d(x_extra[i_TE_extra:i_LE_extra+1], y_extra[i_TE_extra:i_LE_extra+1], fill_value="extrapolate", kind="quadratic")
        y_extrados_body = f_extrados_body(x_extrados_body)

        # Interpolation extrados TE
        x_extrados_TE = np.linspace(x_rang_TE,1,nb_TE,endpoint = True)[::-1]
        f_extrados_TE= interpolate.interp1d(x_extra[:i_TE_extra+1], y_extra[:i_TE_extra+1], fill_value="extrapolate", kind="quadratic")
        y_extrados_TE = f_extrados_TE(x_extrados_TE)

        # Interpolation Intrados LE
        x_intrados_LE = np.linspace(0,x_rang_LE,nb_LE,endpoint = True)[1:]
        f_intrados_LE= interpolate.interp1d(x_intra[:i_LE_intra+1], y_intra[:i_LE_intra+1], fill_value="extrapolate", kind="quadratic")
        y_intrados_LE = f_intrados_LE(x_intrados_LE)

        # Interpolation Intrados Body
        x_intrados_body = np.linspace(x_rang_LE,x_rang_TE,nb_point,endpoint = True)
        f_intrados_body= interpolate.interp1d(x_intra[i_LE_intra:i_TE_intra+1], y_intra[i_LE_intra:i_TE_intra+1], fill_value="extrapolate", kind="quadratic")
        y_intrados_body = f_intrados_body(x_intrados_body)

        # Interpolation Intrados TE
        x_intrados_TE = np.linspace(x_rang_TE,1,nb_TE,endpoint = True)
        f_intrados_TE= interpolate.interp1d(x_intra[i_TE_intra:], y_intra[i_TE_intra:], fill_value="extrapolate", kind="quadratic")
        y_intrados_TE = f_intrados_TE(x_intrados_TE)

        x_inter = list(x_extrados_TE) + list(x_extrados_body) + list(x_extrados_LE) + list(x_intrados_LE) + list(x_intrados_body) + list(x_intrados_TE)
        y_inter = list(y_extrados_TE) + list(y_extrados_body) + list(y_extrados_LE) + list(y_intrados_LE) + list(y_intrados_body) + list(y_intrados_TE)

        #x_camb = np.divide(np.addition(x_extrados_LE,x_intrados_LE),2) + np.divide(np.addition(x_extrados_body,x_intrados_body),2) + np.divide(np.addition(x_extrados_body,x_intrados_body),2)
        #y_camb = np.divide(np.addition(y_extrados_LE,y_intrados_LE),2) + np.divide(np.addition(y_extrados_body,y_intrados_body),2) + np.divide(np.addition(y_extrados_body,y_intrados_body),2)

        return x_inter,y_inter

    
    def coordinate(dir=r"data/Airfoil_Coordinate",nb_point = 30, nb_LE = 20, nb_TE = 10):
        airfoils = os.listdir(dir)
        marchepas = []
        all_y = []
        nom_profil = []
        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            try:
                x_inter,y_inter = format.rawtointer(name,dir=dir, nb_point = nb_point, nb_LE =nb_LE , nb_TE = nb_TE)
                all_y.append(np.array(y_inter).T)
                nom_profil.append(name)
            except: 
                marchepas.append(name)
        return np.array(x_inter).T, np.matrix(all_y).T,nom_profil,marchepas

    def plot_airfoil(name):
        x,y = utils.coordFile2list(name)
        x_inter,y_inter = format.rawtointer(name)

        plt.figure(figsize = (12,8))
        plt.plot(x,y,label = 'coordonnées brute')
        plt.plot(x_inter,y_inter,label = 'coordonnées interpolées')
        plt.title('Airfoil {}'.format(name))
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.legend()
        plt.show()

    
    def camber_thick_coord(dir=r"data/Airfoil_Coordinate",nb_point = 30, nb_LE = 20, nb_TE = 10):
        
        airfoils = os.listdir(dir)
        marchepas = []
        all_y_c = []
        all_y_t = []
        nom_profil = []

        for airfoil in airfoils:
            name = airfoil.replace('.dat','')
            try:
                nb_extra = int(((nb_point + nb_LE + nb_TE)*2 - 2)/2)
                x_inter,y_inter = format.rawtointer(name, nb_point = nb_point, nb_LE =nb_LE , nb_TE = nb_TE)
                y_inter_extrados = y_inter[:nb_extra]
                y_inter_intrados = y_inter[nb_extra:]
                
                y_inter_thickness = np.abs(np.subtract(y_inter_extrados,y_inter_intrados))
                y_inter_camber = np.multiply(np.add(y_inter_extrados,y_inter_intrados),1/2)


                all_y_c.append(np.array(y_inter_camber).T)
                all_y_t.append(np.array(y_inter_thickness).T)
                nom_profil.append(name)
            except: 
                marchepas.append(name)
        return np.array(x_inter[:nb_extra]).T, np.matrix(all_y_c).T ,  np.matrix(all_y_t).T, nom_profil, marchepas
    

class cp():
    def proportion_var(d,p):
        A = np.power(d,2)
        PVE = []
        for k in range(p):
            PVE.append(np.sum(A[:k+1])/np.sum(A))
        return PVE

class lb():

    '''
    Cette classe regroupe les fonctions qui permettent de tracer la 
    Répartition des profils en fonction de leurs air et de leur finesse
    '''

    def air_profils(x,ally):
        '''
        La formule de l'air gausienne
        '''
        aire = []
        for j in range(np.shape(ally)[1]):
            y = ally[:,j]
            A  = 0
            for i in range (1,len(y)-1):
                A+= x[i]*(float(y[i+1])-float(y[i-1]))
            aire.append(A)
        return aire
   


class critere():
    def point_calc(airfoils,M,Re,class1 = True,val = 'val'):
        marchepas = []
        pD = []
        pC = []
        pA = []
        pF = []
        for airfoil in airfoils:
            try:
                data =  analyse.characteristic_point(airfoil,M,Re,class1 = class1,val = val)
                pD.append(data[0])
                pC.append(data[1])
                pA.append(data[2])
                pF.append(data[3])
            except: 
                marchepas.append(airfoil) 
                pD.append('error')
                pC.append('error')
                pA.append('error')
                pF.append('error')
        if len(marchepas)>0:
            lg.info("Les points-critères de {} profils n'ont pas pu être déterminé".format(len(marchepas)))
        
        return pD,pC,pA,pF,marchepas



if __name__ == "__main__":  
    print('fum22 nocive')