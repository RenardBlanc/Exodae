sudo su
git clone https://github.com/RenardBlanc/Exodae.git
cd Exodae
apt-get update
Yes
apt-get install python3.8-venv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python3
from app.class_app import *