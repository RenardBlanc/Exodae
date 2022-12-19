'''
------Send to Drive-----
Ce code est un programme de developpement qui nous permet 
d'envoyer les fichiers de figures des modèles présents dans les VM
dans un dropbox 

Input : Token et directory

Created: 12/12/2022
Updated: 12/12/2022
@Auteur: Ilyas Baktache
'''

# Librairies
import sys
import dropbox
from dropbox import Dropbox
import os
import logging as lg
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) == 3:
        # Demande le token d'accés 
        directory = str(sys.argv[1]) 
        ACCESS_TOKEN = str(sys.argv[2])
        # On se connecte au dropbox:
        # Créez une instance de la classe DropboxClient en utilisant votre clé d'accès
        dbx = Dropbox(ACCESS_TOKEN,  scope=['files.content.write'])
        # Utilisez la fonction os.walk pour obtenir la liste des fichiers et des répertoires dans le répertoire
        for root, dirs, files in os.walk(directory):
            # Affichez les fichiers dans le répertoire courant
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Sending file : {}...".format(files))
            for file in files:
                try :
                    # Ouvrez le fichier que vous souhaitez envoyer sur Dropbox
                    with open(directory + file, "rb") as f:
                        # Envoyez le fichier sur Dropbox en utilisant la méthode files_upload
                        dbx.files_upload(f.read(), "/"+file, mode=dropbox.files.WriteMode.overwrite)
                except:
                    pass
            lg.info("[" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "] Files sent to Dropbox")