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
import dropbox
from dropbox import Dropbox
import os

# on se connecte au dropbox:
# Ajoutez votre clé d'accès à Dropbox ici
ACCESS_TOKEN = "sl.BU263hdGycEpLxXpM8eFbI4s8hSMrrmlhuh-tUd1QmYj-IReO1agN3y5tw-SvwAhtEDNTRjUGgWthPKZrH9LUEm2GIJXcm90O9YxH2TigdlS32p2QlCWWQIQTEH4S61mbXrj5gczEmuf"

# Créez une instance de la classe DropboxClient en utilisant votre clé d'accès
dbx = Dropbox(ACCESS_TOKEN,  scope=['files.content.write'])

# On liste dans un premier temps les modèles présents:
# ------------------------------------------------
# Nom du répertoire
directory = 'CNN/results/figure'

# Utilisez la fonction os.walk pour obtenir la liste des fichiers et des répertoires dans le répertoire
for root, dirs, files in os.walk(directory):
    # Affichez les fichiers dans le répertoire courant
    for file in files:
        print("fichier {}\n".format(file))
        # Ouvrez le fichier que vous souhaitez envoyer sur Dropbox
        with open(directory + file, "rb") as f:
            # Envoyez le fichier sur Dropbox en utilisant la méthode files_upload
            dbx.files_upload(f.read(), "/"+file, mode=dropbox.files.WriteMode.overwrite)
