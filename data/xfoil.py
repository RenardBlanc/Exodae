"""
---------------------XFOIL python interface---------------------
This module presents different classes and functions in order 
to interact with the XFOIL software from python

Created: 12/05/2022
Updated: 24/05/2022
@Auteur: Ilyas Baktache
Based on Perdro Leal code 'Aeropy'
"""
# %%

# Librairies
# --------------------------------------------------------------

# Command and os
import subprocess as sp
import os  # To check for already existing files and delete them
import shutil  # Modules necessary for saving multiple plots.

# Log
import logging as lg
lg.basicConfig(level=lg.INFO)  # Debug and info
from datetime import datetime
import time
# Math
import numpy as np
import os
import sys
# --------------------------------------------------------------


class xfoil():
    
    # --> put you xfoil path there
    dir = r"C:\Users\ilyas\OneDrive\Documents\Projet\airfoil_Optimisation_ML"
    if os.name == "nt":
        # for autorization issue
        assert os.path.isdir(dir)
        os.chdir(dir)

    
    def get_polar(name, dir = "", Re = 50000, Mach = 0, iteration = 70):
        '''
        Function that return polar of an airfoil using xfoil through python.
        It runs for alpha in [-20,20]
        Input:
            * name (str) : Name of the airfoil
            * dir  (str) : Direction where the airfoil dat file is located
            * Re (float) : Reynold number (viscous flow)
            * Mach (float) : Mach number (compressible flow)
            * iteration (int) : Number of iteration until convergence. 
        '''   
        def cmd(commande):
            '''
            Submit a command through PIPE to the command lineself.
            @author: Ilyas Baktache (Based on Hakan Tiftikci's code)
            '''
            ps.stdin.write(commande + '\n')     
        # --------------------------------------------
        # Start xfoil
        # --------------------------------------------
        
        # Check system
        if os.name == "nt":
            # Windows
            linux = False
        else: 
            # Mac and Linux
            linux = True

        if linux:
            with open(os.devnull, 'w') as fp:
                ps = sp.Popen('xfoil.exe',
                            stdin=sp.PIPE,
                            stdout=fp,
                            stderr=None,
                            encoding='utf8')

        else:
            startupinfo = sp.STARTUPINFO()
            startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
            # Random mode variable to avoid writing stuff from xfoil on the
            # console
            sout = 0
            # Calling xfoil with Popen
            ps = sp.Popen('xfoil.exe',
                            stdin=sp.PIPE,
                            stdout=sout,
                            stderr=None,
                            startupinfo=startupinfo,
                            encoding='utf8')

        # Preventing XFOIL from opening XPLOT11-Windows, therfore being able to run aeroPy in a
        #command-line only Linux 
        if linux:
            cmd('PLOP')
            cmd('G F')
            cmd('')

        # --------------------------------------------
        # Load airfoil
        # --------------------------------------------
        path = xfoil.path2name(name,dir)
       
        cmd('LOAD {}'.format(path)) # 	Load the dat file
        lg.debug("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Airfoil {} loaded in Xfoil for performance analysis".format(name))
        cmd('MDES') # Go to the MDES menu
        cmd('FILT') # Smooth any variations in the dat file data
        cmd('EXEC') # Execute the smoothing
        cmd('')     # Exit MDES menu
        cmd('PANE') # Set the number and location of the airfoil points for analysis        
        cmd('OPER')  # Opening OPER module in Xfoil
        cmd('iter {}'.format(iteration)) # Max number of iterations set to 70 for convergence
        cmd('RE {}'.format(Re)) # Set Reynolds number
        cmd('VISC {}'.format(Re)) # Set viscous calculation with Reynolds number

        if Mach > 0:
            cmd('MACH %s' % Mach) # Defining Mach number for Prandtl-Gauber correlation        
        
        cmd('PACC') # Start polar output file
        # Defining the polar file name
        filename = xfoil.nameFile(name, Re=Re)
        try:
            os.remove('{}.txt'.format(filename))
        except OSError:
            pass

        cmd('{}.txt'.format(filename)) # The output polar file name
        cmd('') # No dump file
        
        #List of alpha going from 0 to 20 with a step of 0.25
        alphas = [0]
        for i in range (60):
            alphas.append(alphas[i]+0.25)
    
        for alpha in alphas:
            try:
                cmd('ALFA {}'.format(alpha))
            except:
                cmd('INIT')
        
        for alpha in alphas:
            try:
                cmd('ALFA {}'.format(-alpha))
            except:
                cmd('INIT')
       
        cmd('PACC') # 	Close polar file
        cmd('VISC') # Reinitialise viscous calculation 
        cmd('') # Exiting From OPER mode
        cmd('QUIT') # Exit Xfoil
        # From stdin
        ps.stdin.close()
        # From popen
        ps.wait()

        lg.debug("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Performance computed for the airfoil {}".format(name))

    def nameFile(name, Re=0, M=0):
        filename = r'data\Airfoil_Polar\M_{}\Re_{}\{}'.format(str(M).replace('.',''),Re, name)
        return filename

    def path2name(name,dir):
        path = '{}\{}.dat'.format(dir,name)
        return path

    
class polar_file():
    '''
    Cette classe permet de formater les fichier de polaire issus de xfoil:
        * Les coefficients fournis par Xfoil sont valide jusqu'a cL_max.
        On retire donc les données au dessus de cette valeurs
        * On organise les données selon alpha croissant pour plus de lisibilité
    '''
        
