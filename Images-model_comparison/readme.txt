############################################ Cours read_me!:
les librairies dans le requirement sont les librairies de bases (packages complet ci-après)
Dossier images, et les quatre csv dans: app>data
Téléchargement automatique des fichiers pré-traités si force_preprocess=True ou absence des fichiers dans output_preprocessed_filename = "app/data/processed_data"
Se mettre dans le dossier app et exécuter main.py
Dé-zipper les quatre modèles (app>data>models) si vous ne voulez pas les regénérer.
Pour regénérer l'entrainement des modèles, il suffit de supprimer le dossier du/des modèle(s) dans app/data/models

######## Python 3.10.12 (avec pyenv), sous WSL2 ubuntu_24.04 

###########################################.wslconfig
[wsl2]
memory=14GB           # Réduit de 16GB pour laisser de la marge au système hôte
processors=7
swap=24GB            # Augmenté pour supporter les pics de mémoire durant l'entraînement
localhostForwarding=true
gpuSupport=true      # Ajouté pour garantir le support GPU
kernelCommandLine=nvidia_drm.modeset=1  # Optimise le support NVIDIA

# Optimisations mémoire
vmIdleTimeout=60000

##################################### Config Win10 :
 32Go RAM, CPU Intel Core i7-7700HQ @ 2.80Ghz - 8 processeurs logiques, GPU NVIDIA Quadro 1200M

######################################
Package                  Version
------------------------ ------------
absl-py                  2.1.0
asttokens                3.0.0
astunparse               1.6.3
cachetools               5.5.0
catboost                 1.2.7
certifi                  2024.8.30
charset-normalizer       3.4.0
cloudpickle              3.1.0
cmake                    3.25.0
comm                     0.2.2
contourpy                1.3.1
cupy-cuda12x             13.3.0
cycler                   0.12.1
debugpy                  1.8.9
decorator                5.1.1
docker                   7.1.0
exceptiongroup           1.2.2
executing                2.1.0
fastrlock                0.8.2
filelock                 3.13.1
flatbuffers              24.3.25
fonttools                4.55.0
fsspec                   2024.2.0
gast                     0.4.0
google-auth              2.37.0
google-auth-oauthlib     1.2.1
google-pasta             0.2.0
graphviz                 0.20.3
grpcio                   1.68.0
h5py                     3.12.1
idna                     3.10
imbalanced-learn         0.12.4
imblearn                 0.0
ipykernel                6.29.5
ipython                  8.30.0
jedi                     0.19.2
Jinja2                   3.1.3
joblib                   1.4.2
jupyter_client           8.6.3
jupyter_core             5.7.2
keras                    2.15.0
Keras-Preprocessing      1.1.2
kiwisolver               1.4.7
libclang                 18.1.1
lit                      15.0.7
llvmlite                 0.43.0
Markdown                 3.7
markdown-it-py           3.0.0
MarkupSafe               3.0.2
matplotlib               3.6.2
matplotlib-inline        0.1.7
mdurl                    0.1.2
ml-dtypes                0.2.0
mpmath                   1.3.0
namex                    0.0.8
nest-asyncio             1.6.0
networkx                 3.2.1
numba                    0.60.0
numpy                    1.26.0
nvidia-cublas-cu11       11.11.3.6
nvidia-cuda-runtime-cu11 11.8.89
nvidia-cudnn-cu11        8.6.0.163
nvidia-nccl-cu12         2.23.4
oauthlib                 3.2.2
opt_einsum               3.4.0
optree                   0.13.1
packaging                24.2
pandas                   1.5.0
parso                    0.8.4
pexpect                  4.9.0
Pillow                   9.4.0
pip                      24.3.1
platformdirs             4.3.6
plotly                   5.24.1
prompt_toolkit           3.0.48
protobuf                 4.25.5
psutil                   6.1.0
ptyprocess               0.7.0
pure_eval                0.2.3
pyasn1                   0.6.1
pyasn1_modules           0.4.1
Pygments                 2.18.0
pynndescent              0.5.13
pyparsing                3.2.0
python-dateutil          2.9.0.post0
pytz                     2024.2
PyYAML                   6.0.2
pyzmq                    26.2.0
requests                 2.32.3
requests-oauthlib        2.0.0
rich                     13.9.4
rsa                      4.9
scikit-learn             1.2.2
scipy                    1.14.1
setuptools               65.5.0
shap                     0.46.0
six                      1.16.0
slicer                   0.0.8
stack-data               0.6.3
sympy                    1.13.1
tenacity                 9.0.0
termcolor                2.5.0
threadpoolctl            3.5.0
torch                    2.0.1+cu117
torchaudio               2.0.2+cu117
torchvision              0.15.2+cu117
tornado                  6.4.2
tqdm                     4.67.0
traitlets                5.14.3
triton                   2.0.0
typing_extensions        4.12.2
umap-learn               0.5.7
urllib3                  2.2.3
wcwidth                  0.2.13
Werkzeug                 3.1.3
wheel                    0.45.0
wrapt                    1.14.1
xgboost                  2.1.3

############################### Tentatives .venv et docker:
# Pour .venv:        curl https://pyenv.run | bash
                    export PATH="$HOME/.pyenv/bin:$PATH"
                    eval "$(pyenv init --path)"
                    eval "$(pyenv virtualenv-init -)"
                    pyenv install 3.10.12
                    pyenv global 3.10.12
                    sudo add-apt-repository ppa:deadsnakes/ppa
                    sudo apt update
                    sudo apt install python3.10 python3.10-venv python3.10-distutils
                    python3.10 -m venv .venv
                    source .venv/bin/activate
                    pip install --upgrade pip setuptools
                    pip install -r requirements.txt

# Cuda et nvidia :
# ajouter les dépôts officiels NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb

sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Mettre à jour les dépôts et installer CUDA :
sudo apt-get update
sudo apt-get install cuda-libraries-12-4
sudo apt-get install cuda-runtime-12-4
sudo apt-get install cuda-drivers

#old
#sudo apt-get update
#sudo apt-get install -y cuda-11-0
#sudo apt-get install -y libcudnn8
#sudo apt-get install -y libnvinfer7
#pip install nvidia-cuda-runtime-cu11==11.8.89
#pip install nvidia-cudnn-cu11==8.6.0.163

pip install --upgrade pip
sudo apt update
sudo apt install python3-distutils

docker-compose build
docker-compose up
