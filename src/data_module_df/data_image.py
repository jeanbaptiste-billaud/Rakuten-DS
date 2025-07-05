# data_image.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class RakutenImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image


def create_image_dataset(df, image_dir, labels=None):
    image_paths = []
    images_not_found = 0

    for _, row in df.iterrows():
        image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        image_path = os.path.join(image_dir, image_file)
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            images_not_found += 1

    if not image_paths:
        raise FileNotFoundError("Aucune image valide trouvée dans le dossier spécifié.")

    return RakutenImageDataset(image_paths, labels)


def extract_resnet_features(dataset, device, batch_size=128, num_workers=4, desc="Extraction features"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, batch_labels = batch
                labels.extend(batch_labels.numpy())
            else:
                inputs = batch

            inputs = inputs.to(device)
            batch_features = resnet(inputs)
            features.append(batch_features.cpu().numpy())

    return {
        'features': np.vstack(features),
        'labels': np.array(labels) if labels else None
    }

