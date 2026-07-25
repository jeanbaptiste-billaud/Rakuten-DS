import os
import sys
import socket
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

from rakuten_vision.dataset import load_lmdb_partitions, data_split_preprocess
from rakuten_vision.utils import EarlyStopping

import torch
import torch.distributed as dist
from torch.multiprocessing import cpu_count, spawn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50, ResNet50_Weights


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind sur un port libre
        return s.getsockname()[1]  # Retourne le port


def setup_windows(rank, world_size):
    hostname = socket.gethostname()
    os.environ['MASTER_ADDR'] = "::1"  # socket.gethostbyname(hostname)  # Adresse du maître
    os.environ['MASTER_PORT'] = str(find_free_port())  #'12355'  # Port de communication
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def setup_slurm(rank, world_size):
    # SLURM présente les adresses hôte et port dans les variables d'environnement suivantes
    host_name = os.getenv('SLURM_NODELIST')
    if not host_name:
        raise ValueError("SLURM_NODELIST n'est pas défini dans l'environnement SLURM")

    # Résoudre le nom d'hôte en adresse IP
    os.environ['MASTER_ADDR'] = socket.gethostbyname(host_name)
    os.environ['MASTER_PORT'] = "29500"

    # Initialise le groupe de processus en utilisant nccl pour GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def synchronize_tensor(local_value, world_size, device, operation="mean"):
    """
    Synchronise une valeur locale entre tous les processus dans un contexte DDP.

    Args:
        local_value (float, torch.Tensor): Valeur locale à synchroniser (scalaire ou tenseur).
        world_size (int): Nombre total de processus (GPUs).
        device (torch.device): Appareil utilisé (e.g., 'cuda:0').
        operation (str): Opération à effectuer après synchronisation ('mean' ou 'sum').

    Returns:
        torch.Tensor: Valeur synchronisée entre les GPUs.
    """
    # Si la valeur est un scalaire, convertir en tenseur
    if isinstance(local_value, (int, float)):
        local_value = torch.tensor([local_value], device=device)

    # Synchronisation avec dist.all_reduce
    dist.all_reduce(local_value, op=dist.ReduceOp.SUM)

    # Effectuer l'opération spécifiée
    if operation == "mean":
        return local_value / world_size
    elif operation == "sum":
        return local_value
    else:
        raise ValueError("Operation not supported: choose 'mean' or 'sum'")


def data_loader(dataset, batch_size, world_size, rank, num_workers, shuffle=True):
    """
    Prépare un DistributedSampler et un DataLoader pour un dataset donné.

    Args:
        dataset (Dataset): Dataset à utiliser.
        batch_size (int): Taille du batch.
        world_size (int): Nombre total de processus DDP.
        rank (int): Rang du processus actuel.
        shuffle (bool): Indique si les données doivent être mélangées.

    Returns:
        DataLoader: DataLoader configuré avec un DistributedSampler.
    """
    # Création du DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    # Création du DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return loader


def gather_and_concat_stats(stats, world_size, device):
    """
    Synchronise et concatène les statistiques entre tous les processus dans un environnement DDP.

    Args:
    - stats (dict): Un dictionnaire contenant les statistiques pour l'epoch courant.
    - world_size (int): Le nombre total de processus dans l'environnement DDP.
    - device (torch.device): Le device sur lequel les tensors sont présents.

    Returns:
    - all_stats (dict): Dictionnaire contenant les statistiques concaténées pour tous les processus.
    """
    all_stats = {}

    # Synchronisation et concaténation des statistiques
    for label, stat in stats.items():
        stat_tensor = stat.unsqueeze(0).to(device)  # Ajouter une dimension pour correspondre à un batch

        # Liste pour collecter les stats de tous les processus
        gathered_stats = [torch.zeros_like(stat_tensor) for _ in range(world_size)]

        # Synchroniser les stats entre tous les processus
        dist.all_gather(gathered_stats, stat_tensor)

        # Concaténer les résultats entre les processus
        if label not in all_stats:
            all_stats[label] = torch.cat(gathered_stats, dim=0).cpu()
        else:
            all_stats[label] = torch.cat((all_stats[label], torch.cat(gathered_stats, dim=0).cpu()), dim=0)

    return all_stats


def update_metric(value, history, world_size, device):
    global_value = synchronize_tensor(value, world_size, device, operation="mean")
    if isinstance(value, torch.Tensor):  # and len(value.shape) > 0:  # Vérifie si c'est un tenseur non scalaire
        return torch.cat((history, global_value.cpu().unsqueeze(0)), 0)
    else:
        history.append(global_value.cpu().item())
        return history


def train(rank, world_size, train_dataset, val_dataset, batch_size, num_workers, model, lr, backup_pwd):
    # Configuration de l'environnement DDP
    if sys.platform.system() == "Windows":
        setup_windows(rank, world_size)
    elif 'SLURM_JOB_ID' in os.environ:
        setup_slurm(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    n_epochs = 1000

    train_loader = data_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = DDP(model.to(device), device_ids=[rank])
    criterion = CrossEntropyLoss()  # Pour classification multi-classes
    optimizer = Adam(model.fc.parameters(), lr=lr)  # Optimise uniquement la tête
    early_stopping = EarlyStopping(unsupervised=False, patience=10)

    metrics_t = {"loss": {"value": None, "history": []}}
    metrics_v = {"loss": {"value": None, "history": []}}
    checkpoints_dic = dict()

    epoch_dict = dict()
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

        metrics_t["loss"]["value"] = t_loss / (n + 1)
        for name, data in metrics_t.items():
            data["history"] = update_metric(data["value"], data["history"], world_size, device)

        v_loss = 0.0
        model.eval()
        with torch.no_grad():
            for n, (img, labels) in enumerate(val_loader):
                img, labels = img.to(device), labels.to(device)
                outputs = model(img)
                loss = criterion(outputs, labels)
                v_loss += loss.item()

        metrics_v["loss"]["value"] = v_loss / (n + 1)
        for name, data in metrics_v.items():
            data["history"] = update_metric(data["value"], data["history"], world_size, device)

        if rank == 0:
            epoch_dict[str(epoch)] = dict(epoch=epoch, model_state_dict=model.module.state_dict(), optimizer_state_dict=Adam.state_dict(),
                                          loss_t=metrics_t["loss"]["history"][-1], loss_v=metrics_v["loss"]["history"][-1])

            print(f'Epoch {epoch} training done', flush=True)

            fig = plt.figure()
            plt.plot(metrics_t["loss"]["history"])
            plt.plot(metrics_v["loss"]["history"])
            fig.savefig(backup_pwd + "_loss.png")

            if epoch % 10 == 0:
                torch.save(checkpoints_dic, backup_pwd + "_checkpoints_dic.pth")

        # EarlyStopping
        early_stopping(model=model, val_loss=metrics_v["loss"]["history"][-1])
        if early_stopping.counter == 0:
            best_epoch = epoch
        if early_stopping.early_stop:
            epoch_to_del = [str(i) for i in range(best_epoch + 1, epoch + 1)]

            for i in epoch_to_del:
                checkpoints_dic.pop(i)
            print("Early stopping triggered")
            break


def main():
    world_size = torch.cuda.device_count()

    # Répertoire de sortie
    try:
        result_pwd = os.path.join(os.getcwd(), sys.argv[1])
    except IndexError:
        result_pwd = os.path.join(os.getcwd())

    project_pwd = "/projets/signal/t0301543/simu_python/Rakuten/"
    backup_pwd = project_pwd + "temp/"

    # paramètres généraux
    n_epoch = 10  # 5000  # Maximal number of training iterations
    lr = 1.e-3
    batch_size = 512
    num_workers = cpu_count()
    if num_workers > 36:
        num_workers = 36

    # Import des données
    print("\nPréparation des datasets et initialisation du modèle\n")
    raw_data = load_lmdb_partitions("data/train_dataset_raw")
    labels = [raw_data[i][1] for i in range(len(raw_data))]
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.1, stratify=labels, random_state=42)
    val_dataset, train_dataset = data_split_preprocess(raw_data, train_idx, val_idx, batch_size)

    # Création du modèle
    n_labels = pd.DataFrame(labels).nunique().item()
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = Linear(model.fc.in_features, n_labels)

    print("\nLancement de l'entrainement\n")
    spawn(train, args=(world_size, train_dataset, val_dataset, batch_size, num_workers, model, lr, backup_pwd), nprocs=world_size)


if __name__ == "__main__":
    main()
