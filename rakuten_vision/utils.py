import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count


class EarlyStopping:
    def __init__(self, unsupervised=False, patience=10, min_delta=0, tolerance_loss=1e-4, tolerance_weights=1e-4):
        """
        mode (str): 'supervised' ou 'unsupervised' pour définir le type d'apprentissage.
        patience (int): Pour l'apprentissage supervisé, combien d'époques consécutives la validation peut ne pas s'améliorer avant d'arrêter l'entraînement.
        min_delta (float): Pour l'apprentissage supervisé, le changement minimum dans la métrique de validation pour qu'une amélioration soit considérée.
        tolerance_loss (float): Pour l'apprentissage non supervisé, la tolérance pour l'arrêt basé sur la convergence de la perte.
        tolerance_weights (float): Pour l'apprentissage non supervisé, la tolérance pour l'arrêt basé sur la convergence des poids.
        """
        self.unsupervised = unsupervised
        self.patience = patience
        self.min_delta = min_delta
        self.tolerance_loss = tolerance_loss
        self.tolerance_weights = tolerance_weights
        self.best_score = np.inf
        self.counter = 0
        self.prev_weights = None
        self.early_stop = False

    def check_early_stop_supervised(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

    def check_early_stop_unsupervised(self, model, current_loss):
        # Check loss convergence
        loss_converged = False
        if np.abs(self.best_score - current_loss) < self.tolerance_loss:
            self.counter += 1
        else:
            self.counter = 0
            self.epoch_to_keep = model
        self.best_score = min(self.best_score, current_loss)

        # Check weight changes
        weight_converged = False
        if self.prev_weights is not None:
            weight_change = 0
            for p_old, p_new in zip(self.prev_weights, model.parameters()):
                weight_change += torch.norm(p_new - p_old).item() / (torch.norm(p_old).item() + 1e-8)
            weight_change /= len(list(model.parameters()))
            weight_converged = weight_change < self.tolerance_weights

        # Store current weights
        self.prev_weights = [p.clone() for p in model.parameters()]

        # Check if both conditions are met
        if self.counter >= self.patience and weight_converged:
            self.early_stop = True

    def __call__(self, model=None, val_loss=None, val_penalty=None):
        if not self.unsupervised:
            self.check_early_stop_supervised(val_loss)
        else:
            self.check_early_stop_unsupervised(model, val_penalty)


def calculate_mean_std(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    n_pixels = 0

    for images, _ in loader:
        n_pixels += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_squared += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std = (channel_sum_squared / n_pixels - mean ** 2).sqrt()

    return mean, std