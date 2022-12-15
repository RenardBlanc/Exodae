'''
------Send to Drive-----
Ce code est un programme de developpement qui nous permet 
d'envoyer les fichiers des modèles présents dans les VM
dans un dropbox 

Created: 12/12/2022
Updated: 12/12/2022
@Auteur: Ilyas Baktache
'''
# Librairies
import sys
import dropbox
from dropbox import Dropbox
import os

# On liste dans un premier temps les modèles présents:
# ------------------------------------------------
# Nom du répertoire
directory = 'GAN/figure'


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        # Demande le token d'accés 
        ACCESS_TOKEN = str(sys.argv[1]) 
        # on se connecte au dropbox:
        # Créez une instance de la classe DropboxClient en utilisant votre clé d'accès
        dbx = Dropbox(ACCESS_TOKEN,  scope=['files.content.write'])

        # Utilisez la fonction os.walk pour obtenir la liste des fichiers et des répertoires dans le répertoire
        for root, dirs, files in os.walk(directory):
            # Affichez les fichiers dans le répertoire courant
            for file in files:
                try :
                    print("fichier {}\n".format(file))
                    # Ouvrez le fichier que vous souhaitez envoyer sur Dropbox
                    with open(directory + file, "rb") as f:
                        # Envoyez le fichier sur Dropbox en utilisant la méthode files_upload
                        dbx.files_upload(f.read(), "/"+file, mode=dropbox.files.WriteMode.overwrite)
                except:
                    pass