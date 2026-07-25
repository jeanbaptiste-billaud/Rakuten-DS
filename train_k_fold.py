import os
import sys
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from rakuten_vision.dataset import load_lmdb_partitions, data_split_preprocess
from rakuten_vision.utils import EarlyStopping

import torch
from torch.multiprocessing import cpu_count, Pool, Queue
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torchvision.models import resnet50, ResNet50_Weights


def train_fold(backup_pwd, fold_idx, gpu_idx, raw_data, model, batch_size, num_workers, queue):
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    n_epochs = 2000

    train_idx, val_idx = fold_idx[0], fold_idx[1]
    val_preprocess, train_preprocess = data_split_preprocess(raw_data, train_idx, val_idx, batch_size)

    train_loader = DataLoader(train_preprocess, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_preprocess, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    criterion = CrossEntropyLoss()  # Pour classification multi-classes
    optimizer = Adam(model.fc.parameters(), lr=0.001)  # Optimise uniquement la tête
    early_stopping = EarlyStopping(unsupervised=False)

    val_loss = []
    train_loss = []
    checkpoints_dic = dict()

    for epoch in range(n_epochs):

        model.train()
        t_loss = 0.0
        for img, labels in train_loader:
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()

        train_loss.append(t_loss / (n + 1))

        v_loss = 0.0
        model.eval()
        with torch.no_grad():
            for n, (img, labels) in enumerate(val_loader):
                img, labels = img.to(device), labels.to(device)
                outputs = model(img)
                loss = criterion(outputs, labels)
                v_loss += loss.item()

        val_loss.append(v_loss / (n + 1))

        checkpoints_dic[str(epoch)] = dict(epoch=epoch, model_state_dict=model.state_dict(),
                                           optimizer_state_dict=optimizer.state_dict(),
                                           loss_v=val_loss[-1], loss_t=train_loss[-1])

        print(f'Epoch {epoch} training done')
        sys.stdout.flush()

        fig = plt.figure()
        plt.plot(train_loss)
        plt.plot(val_loss)
        fig.savefig(backup_pwd+"_loss.png")

        if epoch % 10 == 0:
            torch.save(checkpoints_dic, backup_pwd+"_checkpoints_dic.pth")

        # EarlyStopping
        early_stopping(model=model, val_loss=val_loss[-1])
        if early_stopping.counter == 0:
            best_epoch = epoch
        if early_stopping.early_stop:
            epoch_to_del = [str(i) for i in range(best_epoch + 1, epoch + 1)]

            for i in epoch_to_del:
                checkpoints_dic.pop(i)
            print("Early stopping triggered")
            break

    queue.put(checkpoints_dic[str(best_epoch)])


def main():
    # Répertoire de sortie
    try:
        result_pwd = os.path.join(os.getcwd(), sys.argv[1])
    except IndexError:
        result_pwd = os.path.join(os.getcwd())

    project_pwd = "/projets/signal/t0301543/simu_python/Rakuten/"

    # paramètres généraux
    batch_size = 1024
    num_workers = cpu_count()
    if num_workers > 36:
        num_workers = 36
    num_gpus = torch.cuda.device_count()

    # Chargement des données d'entrainement pré-traitées
    raw_data = load_lmdb_partitions("data/train_dataset_raw")
    labels = [raw_data[i][1] for i in range(len(raw_data))]

    n_labels = pd.DataFrame(labels).nunique().item()

    # Initialisation du kfold
    k_folds = 6
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # création des lists de folds et de gpu
    folds = list(skf.split(range(len(labels)), labels))
    gpus = list(range(num_gpus)) * (len(folds) // num_gpus)
    processes = []

    # Créer une queue pour collecter les résultats
    queue = Queue()

    # Création du modèle
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = Linear(model.fc.in_features, n_labels)

    with Pool(2) as pool:
        for n, (fold, gpu) in enumerate(zip(folds, gpus)):
            temp_pwd = project_pwd + "temp/" + f"fold_{n}_temp"
            pool.apply_async(train_fold, args=(temp_pwd, fold, gpu, raw_data, model, batch_size, num_workers, queue))
        pool.close()
        pool.join()

    # Récupérer les résultats
    results = []
    for _ in range(k_folds):
        results.append(queue.get())  # Récupère les résultats dans la queue

    backup = {"model": model, "results": results}
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = result_pwd + time + "_final_kfold_models.pth"
    torch.save(backup, file_name)


if __name__ == '__main__':
    main()
