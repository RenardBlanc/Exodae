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
ACCESS_TOKEN = "sl.BU3HDGLE6kcRxN9ILNm_Pp8BCUh02f2WNLVTr9sap72Hx-Oa05AuqWBzWEoZto97n6V5ivIrrJ_DU0E0ITg58SH6sh8P2keuvIpNe8T6Dh7sq7jN_hOQ48IcQd5muhV3X8az221rOMet"

# Créez une instance de la classe DropboxClient en utilisant votre clé d'accès
dbx = Dropbox(ACCESS_TOKEN,  scope=['files.content.write'])

# On liste dans un premier temps les modèles présents:
# ------------------------------------------------
# Nom du répertoire
directory = 'CNN/model/'

# Utilisez la fonction os.walk pour obtenir la liste des fichiers et des répertoires dans le répertoire
for root, dirs, files in os.walk(directory):
    # Affichez les fichiers dans le répertoire courant
    for file in files:
        print("fichier {}\n".format(file))
        # Ouvrez le fichier que vous souhaitez envoyer sur Dropbox
        with open(directory+file, "rb") as f:
            # Envoyez le fichier sur Dropbox en utilisant la méthode files_upload
            dbx.files_upload(f.read(), "/"+file, mode=dropbox.files.WriteMode.overwrite)
