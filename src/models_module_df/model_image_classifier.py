# model_image_classifier.py
import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class XGBoostImageClassifier:
    def __init__(self, model_dir):
        self.model = None
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X_train, y_train):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs

    def evaluate(self, y_true, preds, probs):
        return {
            'accuracy': round(accuracy_score(y_true, preds), 4),
            'weighted_f1': round(f1_score(y_true, preds, average='weighted'), 4),
            'macro_f1': round(f1_score(y_true, preds, average='macro'), 4),
            'mean_confidence': round(np.mean(np.max(probs, axis=1)), 4)
        }

    def save(self):
        with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)


class MLPImageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class MLPImageClassifierWrapper:
    def __init__(self, model_dir, input_dim, num_classes):
        self.model = MLPImageClassifier(input_dim, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X_train, y_train, epochs=20, batch_size=64):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        return preds, probs

    def evaluate(self, y_true, preds, probs):
        return {
            'accuracy': round(accuracy_score(y_true, preds), 4),
            'weighted_f1': round(f1_score(y_true, preds, average='weighted'), 4),
            'macro_f1': round(f1_score(y_true, preds, average='macro'), 4),
            'mean_confidence': round(np.mean(np.max(probs, axis=1)), 4)
        }

    def save(self):
        path = os.path.join(self.model_dir, 'mlp_model.pth')
        torch.save(self.model.state_dict(), path)

    def load(self, input_dim, num_classes):
        self.model = MLPImageClassifier(input_dim, num_classes)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'mlp_model.pth')))
        self.model.to(self.device)


class ImageClassifierManager:
    def __init__(self, model_type, model_dir, input_dim=2048, num_classes=27):
        self.model_type = model_type
        if model_type == 'xgboost':
            self.model = XGBoostImageClassifier(model_dir)
        elif model_type == 'mlp':
            self.model = MLPImageClassifierWrapper(model_dir, input_dim, num_classes)
        else:
            raise ValueError(f"Modèle image inconnu : {model_type}")

    def train(self, X, y):
        self.model.train(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, preds, probs):
        return self.model.evaluate(y_true, preds, probs)

    def save(self):
        self.model.save()

    def load(self):
        if self.model_type == 'mlp':
            self.model.load(input_dim=2048, num_classes=27)
        else:
            self.model.load()