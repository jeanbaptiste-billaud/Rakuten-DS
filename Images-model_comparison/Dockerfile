# Image de base TensorFlow avec support GPU
FROM tensorflow/tensorflow:2.14.0-gpu

# Copie et installation de cuDNN
COPY cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb /tmp/

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        nodejs \
        npm && \
    dpkg -i /tmp/cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y libcudnn9-cuda-12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

# Configuration du répertoire de travail
WORKDIR /workdir

# Copie des fichiers
COPY . .

# # Liste et analyse des packages installés
# RUN pip list > /tmp/pip_list.txt && \
#     cat /tmp/pip_list.txt && \
#     python3 -c "import numpy as np; \
#                 with open('/tmp/pip_list.txt', 'r') as f: \
#                     lines = f.readlines() \
#                     for line in lines[2:]: \
#                         print(line.strip())"

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt
    
# RUN pip list > /tmp/pip_list.txt && \
#     cat /tmp/pip_list.txt && \
#     python3 -c "with open('/tmp/pip_list.txt', 'r') as f: ;\
#                     lines = f.readlines()"

# Installation des serveurs de langage pour un meilleur support
RUN npm install -g \
    bash-language-server \
    dockerfile-language-server-nodejs \
    typescript-language-server \
    pyright

# Configuration de l'environnement pour GPU et Python
ENV PYTHONPATH=/workdir \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Configuration des permissions
RUN chmod -R 777 app/data

# Test final de l'installation
RUN python3 -c "import tensorflow as tf; print('Installation finale OK - TensorFlow version:', tf.__version__)"

