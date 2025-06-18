# model_feature_extractor.py
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_from_image_paths(self, image_paths, batch_size=64):
        dataset = FeatureDataset(image_paths, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                features.append(output.cpu().numpy())
        return np.vstack(features)

    def extract_from_single_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(image)
        return feature.cpu().numpy().flatten()


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)
