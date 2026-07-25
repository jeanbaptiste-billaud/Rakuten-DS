import lmdb
import pickle
import shutil
import numpy as np
import pandas as pd
import os

import torch
from torch.multiprocessing import cpu_count
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from torchvision.io import read_image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from rakuten_vision.utils import calculate_mean_std


class InitialDataset(Dataset):
    def __init__(self, root_path="data/"):
        self.img_info = pd.read_csv(root_path + "X_train_update.csv", index_col=["Unnamed: 0"])
        self.img_labels = pd.read_csv(root_path + "Y_train_CVw08PX.csv", index_col=["Unnamed: 0"])
        self.img_dir = root_path + "images/image_train"

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        product_id = self.img_info.loc[idx, 'productid']
        image_id = self.img_info.loc[idx, "imageid"]
        img_name = f"image_{image_id}_product_{product_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = self.img_labels.loc[idx, "prdtypecode"]
        return image, label


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)  # Appliquer les transformations
        return image, label


def train_test_dataset_transformer(train_dataset, batch_size, reshape=(256, 256), crop=224):
    mean, std = calculate_mean_std(train_dataset, batch_size=batch_size)

    # Transformation pour le test (statistiques globales)
    test_transformer = v2.Compose([
        v2.ToImage(),
        v2.Resize(reshape),
        v2.CenterCrop(size=crop),
        v2.ToDtype(torch.float16, scale=True),
        # v2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    train_transformer = v2.Compose([
        v2.ToImage(),  # Convertir en image PIL
        v2.Resize((256, 256)),  # Uniformiser la taille des images
    ])

    return test_transformer, train_transformer


def val_dataset_preprocessing(val_dataset, mean, std, crop=224):
    # mean, std = calculate_mean_std(train_dataset, batch_size=batch_size)

    # Transformation pour le test (statistiques globales)
    transformer = v2.Compose([
        v2.CenterCrop(size=crop),
        v2.ToDtype(torch.float16, scale=True),
        v2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    val_preprocess = TransformedDataset(val_dataset, transform=transformer)

    return val_preprocess


def train_dataset_preprocessing_v1(train_dataset, mean, std, crop=224):
    # mean, std = calculate_mean_std(train_dataset, batch_size=batch_size)

    # Transformation pour le set d'entrainement (statistiques globales)
    transformer = v2.Compose([
        v2.CenterCrop(size=crop),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.RandomGrayscale(p=0.1),
        v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float16, scale=True),
        v2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # Créer les datasets transformés
    dataset = TransformedDataset(train_dataset, transform=transformer)

    return dataset


def train_dataset_preprocessing_v2(train_dataset, mean, std, crop=224):
    # mean, std = calculate_mean_std(train_dataset, batch_size=batch_size)

    # Transformation pour le set d'entrainement (statistiques globales)
    transformer = v2.Compose([
        v2.CenterCrop(size=crop),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomRotation(degrees=15),
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # v2.RandomGrayscale(p=0.1),
        # v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float16, scale=True),
        v2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # Créer les datasets transformés
    dataset = TransformedDataset(train_dataset, transform=transformer)

    return dataset


def data_split_preprocess(dataset, train_idx, val_idx, batch_size):
    val_subset = Subset(dataset, val_idx)
    train_subset = Subset(dataset, train_idx)

    mean, std = calculate_mean_std(train_subset, batch_size=batch_size)

    val_preprocess = val_dataset_preprocessing(val_subset, mean, std)
    train_preprocess = train_dataset_preprocessing_v2(train_subset, mean, std)

    return val_preprocess, train_preprocess


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        """
        Classe pour charger une partition LMDB
        :param lmdb_path: Chemin vers la base LMDB
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]  # Nombre total d'entrées

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(f"{idx}".encode()))  # Charger les données
        image = torch.tensor(data["image"])  # Charger l'image
        label = data["label"]  # Charger le label
        return image, label


def load_lmdb_partitions(lmdb_base_path):
    """
    Charger toutes les partitions LMDB depuis un chemin de base.
    :param lmdb_base_path: Chemin de base des fichiers LMDB (sans suffixe "_partN")
    :return: Un ConcatDataset combinant toutes les partitions
    """
    datasets = []
    partition_idx = 1
    while True:
        partition_path = f"{lmdb_base_path}_part{partition_idx}.lmdb"
        if not os.path.exists(partition_path):  # Arrêter si aucune partition supplémentaire n'existe
            break
        datasets.append(LMDBDataset(partition_path))
        partition_idx += 1
    return ConcatDataset(datasets)


def save_to_lmdb_partitioned(dataset, lmdb_path, transform=None, batch_size=256, map_size=5 * 1024 ** 3, num_workers=4):
    """
    Sauvegarde un dataset dans plusieurs fichiers LMDB, en partitionnant dès que la taille max est atteinte.
    Chaque partition est indexée localement de 0 jusqu'à la taille de cette partition.
    """
    import os
    import pickle
    import lmdb
    from torch.utils.data import DataLoader

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    num_partitions = 1  # Compteur pour les partitions
    current_partition_path = f"{lmdb_path}_part{num_partitions}.lmdb"
    env = lmdb.open(current_partition_path, map_size=map_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    local_idx = 0  # Index local pour chaque partition

    for idx, (images, labels) in enumerate(loader):
        for i in range(len(images)):
            image = transform(images[i]) if transform else images[i]
            data = {
                "image": image.numpy(),
                "label": labels[i].item()
            }

            try:
                # Essayer de mettre les données dans la partition actuelle
                with env.begin(write=True) as txn:
                    txn.put(f"{local_idx}".encode(), pickle.dumps(data))
                local_idx += 1
            except lmdb.MapFullError:
                # Si la partition est pleine, on la finalise et on passe à la suivante
                env.close()
                print(f"Partition {num_partitions} saved at {current_partition_path}")

                num_partitions += 1
                current_partition_path = f"{lmdb_path}_part{num_partitions}.lmdb"
                env = lmdb.open(current_partition_path, map_size=map_size)

                # Réinitialiser l'index local pour la nouvelle partition
                local_idx = 0

                # Réessayer d'ajouter les données dans la nouvelle partition
                with env.begin(write=True) as txn:
                    txn.put(f"{local_idx}".encode(), pickle.dumps(data))
                local_idx += 1

        print(f"Batch {idx + 1}/{len(loader)} processed")

    # Finaliser la dernière partition
    env.close()
    print(f"All data saved. Total partitions: {num_partitions}")


def train_test_dataset_to_lmdb_with_partition(batch_size=512):
    # Charger le dataset brut
    raw_dataset = InitialDataset()
    labels = raw_dataset.img_labels.prdtypecode

    # Séparation stratifiée des données
    train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.1, stratify=labels, random_state=42)
    train_dataset = Subset(raw_dataset, train_idx)
    test_dataset = Subset(raw_dataset, test_idx)

    train_transformer, test_transformer = train_test_dataset_transformer(train_dataset, batch_size)

    # Prétraiter et sauvegarder le test dataset
    save_to_lmdb_partitioned(test_dataset, lmdb_path="data/test_dataset", transform=test_transformer, batch_size=batch_size, num_workers=cpu_count())

    # Prétraiter et sauvegarder le train dataset brut avec partition
    save_to_lmdb_partitioned(train_dataset, lmdb_path="data/train_dataset_raw", transform=train_transformer, batch_size=batch_size, num_workers=cpu_count())

    print("Train and test datasets saved to LMDB with partitions.")
