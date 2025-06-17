import os
import logging
import numpy as np
import pandas as pd
import pickle
import shutil
from tqdm import tqdm
from typing import Dict, Any, Optional
import yaml
import time
from dataclasses import dataclass, fields

# Imports machine learning
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold

# Imports PyTorch
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
from dataclasses import dataclass

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU disponible : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

class RakutenImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize direct à la taille finale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image

class NeuralClassifier(nn.Module):
    def __init__(self, num_classes, config=None):
        super().__init__()
        self.config = config or {}
        self.dropout_rate = self.config.get('dropout_rate', 0.3)
        
        # Architecture pour features pré-extraites de dimension 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(512, num_classes)
        )
        
        # Initialisation des poids améliorée
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)

@dataclass
class PipelineConfig:
    data_path: str
    model_path: str
    image_dir: str
    batch_size: int = 128
    target_size: int = 2000
    random_state: int = 42
    num_workers: Optional[int] = 7
    early_stopping_patience: int = 5

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
            """
            Crée une instance de PipelineConfig à partir d'un fichier YAML
            
            Args:
                yaml_path (str): Chemin vers le fichier de configuration YAML
                
            Returns:
                PipelineConfig: Instance configurée
            """
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            # Filtrer les clés pour ne garder que celles définies dans la classe
            valid_keys = {field.name for field in fields(PipelineConfig)}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
                
            return cls(**filtered_config)

class ProductClassificationPipeline:
    def __init__(self, config: PipelineConfig):
        """
        Initialise le pipeline de classification avec une configuration
        
        Args:
            config (PipelineConfig): Configuration du pipeline
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.label_encoder = LabelEncoder()
        
        # Initialisation
        self._setup_logger()
        self._init_paths()
        self._init_categories()
        
        # États
        self.preprocessed_data = None
        self.model = None

    def _init_categories(self):
        """Initialise le mapping des catégories Rakuten"""
        self.category_names = {
            10: "Livres", 2280: "Jeux vidéo", 50: "Jouets & Jeux",
            1280: "Accessoires téléphones", 2705: "Accessoires console",
            2522: "Équipement bébé", 2582: "Matériel & accessoires",
            1560: "Photos", 1281: "Téléphonie fixe",
            1920: "Musique amplifiée", 2403: "Livres en langues étrangères",
            1140: "TV", 2583: "Articles sport", 1180: "Décoration",
            1300: "Jeux vidéo ancien", 2462: "Fournitures bureau",
            1160: "Électroménager", 2060: "Articles soins",
            40: "DVD & Films", 60: "Consoles", 1320: "CD",
            1302: "Jeux vidéo rétro", 2220: "Puériculture",
            2905: "Instruments musique", 2585: "Sports & Loisirs",
            1940: "Instrument musique", 1301: "Consoles rétro"
        }
        # Créer le mapping vers des indices consécutifs
        self.category_to_idx = {code: idx for idx, code in enumerate(sorted(self.category_names.keys()))}
        self.idx_to_category = {idx: code for code, idx in self.category_to_idx.items()}

    def _init_paths(self):
        """Initialise tous les chemins nécessaires"""
        # Chemins principaux
        self.train_image_dir = os.path.join(self.config.data_path, 'images/image_train')
        self.test_image_dir = os.path.join(self.config.data_path, 'images/image_test')

        # Chemins des modèles
        self.model_dir = os.path.join(self.config.model_path)
        os.makedirs(self.model_dir, exist_ok=True)

        # Chemins des métadonnées
        self.meta_path = os.path.join(self.config.data_path, 'metadata.pkl')
        
        # Chemins pour les résultats
        self.results_dir = os.path.join(self.config.data_path, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _setup_logger(self):
        """Configure le logger pour le suivi des opérations"""
        self.logger = logging.getLogger('classification_pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Évite les doublons de handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # FileHandler pour sauvegarder les logs
            os.makedirs(os.path.join(self.config.data_path, 'logs'), exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.config.data_path, 'logs', 'pipeline.log')
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _create_dataset(self, df, df_name):
        """Crée un dataset PyTorch à partir des données"""
        image_paths = []
        images_not_found = 0
        try:
            self.logger.info(f"Création dataset à partir de {len(df_name)} entrées...")
            try:
                match df_name:
                    case "X_train": df_path = self.train_image_dir
                    case "X_test": df_path = self.test_image_dir
                    case "X_test_split": df_path = self.train_image_dir
            except Exception as e:    
                self.logger.error(f"Erreur match : {str(e)}")
                            
            for _, row in df.iterrows():
                image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
                image_path = os.path.join(df_path, image_file)
                
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    images_not_found += 1

            if len(image_paths) == 0:
                self.logger.error(f"Aucune image trouvée ! {images_not_found} images manquantes")
                self.logger.error(f"Chemin recherché : {image_paths}")
                raise ValueError("Aucune image valide trouvée")
                
            self.logger.info(f"Dataset créé avec {len(image_paths)} images ({images_not_found} non trouvées)")
            return RakutenImageDataset(image_paths)

        except Exception as e:
            self.logger.error(f"Erreur _create_dataset : {str(e)}")
            raise

    def _create_balanced_dataset(self, X_train_df, Y_train_df):
        """
        Crée un dataset équilibré en considérant à la fois la distribution des classes
        et la taille des fichiers images.
        
        Args:
            X_train_df (pd.DataFrame): DataFrame contenant les métadonnées des images
            Y_train_df (pd.DataFrame): DataFrame contenant les labels
            
        Returns:
            List[int]: Liste des indices sélectionnés pour le dataset équilibré
        """
        file_info = []
        
        # Fusion des DataFrames X et Y
        df_merged = X_train_df.merge(Y_train_df, left_index=True, right_index=True)
        
        # Collecte des informations sur les fichiers
        for _, row in df_merged.iterrows():
            image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
            file_path = os.path.join(self.train_image_dir, image_file)
            
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                file_info.append({
                    'index': row.name,
                    'size_kb': size_kb,
                    'prdtypecode': row['prdtypecode'],
                    'imageid': row['imageid'],
                    'productid': row['productid']
                })
        
        df_analysis = pd.DataFrame(file_info)
        df_analysis.set_index('index', inplace=True)
        
        balanced_indices = []
        
        # Pour chaque classe
        for classe in df_analysis['prdtypecode'].unique():
            class_data = df_analysis[df_analysis['prdtypecode'] == classe].copy()
            n_samples = len(class_data)
            
            if n_samples > self.config.target_size:
                # Sous-échantillonnage stratifié par taille
                size_bins = pd.qcut(class_data['size_kb'], q=5, labels=False)
                class_data['size_bin'] = size_bins
                samples_per_bin = self.config.target_size // 5
                
                stratified_sample = []
                for bin_id in range(5):
                    bin_data = class_data[class_data['size_bin'] == bin_id]
                    if len(bin_data) > 0:
                        selected = bin_data.sample(
                            n=min(len(bin_data), samples_per_bin),
                            random_state=self.config.random_state
                        ).index.tolist()
                        stratified_sample.extend(selected)
                
                # Si on n'a pas assez d'échantillons après stratification
                remaining = self.config.target_size - len(stratified_sample)
                if remaining > 0:
                    additional = class_data[~class_data.index.isin(stratified_sample)].sample(
                        n=min(remaining, len(class_data) - len(stratified_sample)),
                        random_state=self.config.random_state
                    ).index.tolist()
                    stratified_sample.extend(additional)
                
                balanced_indices.extend(stratified_sample)
                
            else:
                # Sur-échantillonnage stratifié par taille
                current_indices = class_data.index.tolist()
                balanced_indices.extend(current_indices)  # Ajoute d'abord tous les échantillons existants
                
                if n_samples > 0:
                    # Calcul du nombre d'échantillons supplémentaires nécessaires
                    n_needed = self.config.target_size - n_samples
                    
                    # Division en bins de taille
                    size_bins = pd.qcut(class_data['size_kb'], q=min(5, n_samples), labels=False)
                    class_data['size_bin'] = size_bins
                    
                    # Sur-échantillonnage par bin
                    additional_samples = []
                    samples_needed_per_bin = n_needed // len(class_data['size_bin'].unique())
                    
                    for bin_id in class_data['size_bin'].unique():
                        bin_data = class_data[class_data['size_bin'] == bin_id]
                        if len(bin_data) > 0:
                            bin_indices = bin_data.index.tolist()
                            additional = np.random.choice(
                                bin_indices,
                                size=samples_needed_per_bin,
                                replace=True
                            ).tolist()
                            additional_samples.extend(additional)
                    
                    # Gestion du reste
                    remaining = n_needed - len(additional_samples)
                    if remaining > 0:
                        extra = np.random.choice(
                            current_indices,
                            size=remaining,
                            replace=True
                        ).tolist()
                        additional_samples.extend(extra)
                    
                    balanced_indices.extend(additional_samples)
        
        # Vérification finale
        self.logger.info(f"Indices retenus: {len(balanced_indices)} sur {len(df_analysis)} images")
        for classe in df_analysis['prdtypecode'].unique():
            n_class = sum(df_analysis.loc[balanced_indices, 'prdtypecode'] == classe)
            self.logger.info(f"Classe {classe} ({self.category_names[classe]}): {n_class} images")
        
        return balanced_indices

    def _save_processed_data(
        self,
        X_train,        # dict {'features': np.ndarray, 'labels': np.ndarray ou None}
        y_train,        # np.ndarray
        train_indices,
        X_test,         # dict idem
        X_test_split,   # dict idem
        y_test_split,
        test_split_indices,
        required_files
    ):
        """
        Sauvegarde dans des fichiers .npz,
        chacun contenant un dictionnaire (0D array) nommé 'X_train', 'X_test', etc.
        """
        os.makedirs(os.path.dirname(required_files['X_train']), exist_ok=True)

        # On enregistre le dictionnaire X_train dans un .npz
        np.savez(required_files['X_train'], X_train_=X_train)
        np.savez(required_files['y_train'], y_train_=y_train)
        np.savez(required_files['X_test'], X_test=X_test)
        np.savez(required_files['X_test_split'], X_test_split=X_test_split)
        np.savez(required_files['y_test_split'], y_test_split=y_test_split)
        np.savez(required_files['train_indices'], train_indices=train_indices)
        np.savez(required_files['test_split_indices'], test_split_indices=test_split_indices)

        self.logger.info("[_save_processed_data] Données sauvegardées dans :")
        self.logger.info(f"  - {required_files['X_train']}")
        self.logger.info(f"  - {required_files['X_test']}")
        self.logger.info(f"  - {required_files['X_test_split']}")

    def _load_existing_processed_data(self, required_files):
        """
        Charge les 7 datasets depuis des .npz.
        
        Retourne un nouveau dictionnaire self.preprocessed_data
        """
        try:
            # 1) Train
            X_train = np.load(required_files['X_train'], allow_pickle=True)['arr_0']
            y_train = np.load(required_files['y_train'], allow_pickle=True)['arr_0']
            train_indices = np.load(required_files['train_indices'], allow_pickle=True)['arr_0']
            
            # 2) Test
            X_test = np.load(required_files['X_test'], allow_pickle=True)['arr_0']
            
            # 3) Test_split
            X_test_split = np.load(required_files['X_test_split'], allow_pickle=True)['arr_0']
            y_test_split = np.load(required_files['y_test_split'], allow_pickle=True)['arr_0']
            test_split_indices = np.load(required_files['test_split_indices'], allow_pickle=True)['arr_0']
            
            # Construction du dictionnaire
            preprocessed_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'X_test_split': X_test_split,
                'y_test_split': y_test_split,
                'train_indices': train_indices,
                'test_split_indices': test_split_indices
            }
            return preprocessed_data

        except Exception as e:
            self.logger.error(f"Erreur chargement données : {str(e)}")
            raise         

    def _extract_resnet_features(self, dataset, desc="Extraction features"):
        """
        Extrait les caractéristiques des images via ResNet

        Args:
            dataset: Dataset contenant les images à traiter
            desc: Description pour la barre de progression tqdm

        Returns:
            dict: Dictionnaire contenant les features extraites et les labels si présents
        """
        # Création du dataloader        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,  # Accélère les transferts vers GPU
            prefetch_factor=2,  # Charge les données à l'avance
            persistent_workers=True  # Garde les workers en vie
        )

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Identity()  # Retire la dernière couche
        resnet = resnet.to(self.device)
        resnet.eval()
        
        features = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                if len(batch) == 2:  # Training data avec labels
                    inputs, batch_labels = batch
                    labels.extend(batch_labels.numpy())
                else:  # Test data sans labels
                    inputs = batch
                
                inputs = inputs.to(self.device)
                batch_features = resnet(inputs)
                features.append(batch_features.cpu().numpy())

        return {
            'features': np.vstack(features),
            'labels': np.array(labels) if labels else None
        }
        
    def prepare_data(self, balance_classes=True, force_preprocess=False):
        """
        Prépare les données
        
        Args:
            force_preprocess: Si True, force le prétraitement            
            balance_classes: Si True, équilibre les classes
            
        Returns:
            Dict contenant les données prétraitées
        """
        try:
            # Vérification des fichiers prétraités existants
            features_dir = os.path.join(self.config.data_path, 'processed_data')
            required_files = {
                'X_train': os.path.join(features_dir, 'X_train.npz'),
                'y_train': os.path.join(features_dir, 'y_train.npz'),
                'X_test': os.path.join(features_dir, 'X_test.npz'),
                'X_test_split': os.path.join(features_dir, 'X_test_split.npz'),
                'y_test_split': os.path.join(features_dir, 'y_test_split.npz'),
                'train_indices': os.path.join(features_dir, 'train_indices.npz'),
                'test_split_indices': os.path.join(features_dir, 'test_split_indices.npz'),
            }

            files_exist = all(os.path.exists(path) for path in required_files.values())
                
            if not force_preprocess and files_exist:
                self.logger.info("Chargement des features pré-calculées...")
                data = self._load_existing_processed_data(required_files)
                self.logger.info("Chargement effectué avec succès.")

                # Remplit self.preprocessed_data
                self.preprocessed_data = data

                return self.preprocessed_data
            
            else:
                self.logger.info(f"Prétraitement en cours...")
                if not files_exist:
                    missing_files = [name for name, path in required_files.items()  if not os.path.exists(path)]
                    self.logger.warning(f"Les fichiers suivants sont manquants : {', '.join(missing_files)}")
                
                # a) Lecture des CSV
                X_train_df = pd.read_csv('data/X_train_update.csv')
                Y_train_df = pd.read_csv('data/Y_train_CVw08PX.csv')
                X_test_df  = pd.read_csv('data/X_test_update.csv')   # Test challenge
                
                # b) Split (train / test_split) => 80/20 sur le jeu d'entraînement
                X_train, X_test_split, y_train, y_test_split = train_test_split(
                    X_train_df, Y_train_df,
                    test_size=0.2,
                    stratify=Y_train_df['prdtypecode'],
                    random_state=self.config.random_state
                )
                test_split_indices = X_test_split.index.values
                train_indices = X_train.index.values

                # c) Éventuel balancing
                if balance_classes:
                    train_indices = self._create_balanced_dataset(X_train, y_train)
                    # Si vous renvoyez la liste finale des indices 
                    X_train = X_train.loc[train_indices]
                    y_train = y_train.loc[train_indices]

                # d) Extraction features via _extract_resnet_features
                #    1) Création dataset PyTorch pour train
                train_dataset = self._create_dataset(X_train, y_train, df_name="X_train")
                X_train_features = self._extract_resnet_features(train_dataset, desc="Extraction features train")

                #    2) Création dataset PyTorch pour test "officiel"
                test_dataset = self._create_dataset(X_test_df, None, df_name="X_test")
                X_test_features = self._extract_resnet_features(test_dataset, desc="Extraction features test")

                #    3) Création dataset PyTorch pour test_split
                test_split_dataset = self._create_dataset(X_test_split, y_test_split, df_name="X_test_split")
                X_test_split_features = self._extract_resnet_features(test_split_dataset, desc="Extraction features test_split")

                self.logger.info(f"Sauvegarde des fichiers images...")
                # e) Sauvegarde via _save_processed_data
                self._save_processed_data(
                    X_train_features, 
                    y_train['prdtypecode'].values,
                    X_test_features,
                    X_test_split_features,
                    y_test_split['prdtypecode'].values,
                    train_indices,
                    test_split_indices,
                    required_files
                )
                
                self.logger.info(f"Mise à jour de l'état...")
                # Mise à jour de l'état
                self.preprocessed_data = {
                'X_train': X_train_features['features'],
                'y_train': y_train['prdtypecode'].values,
                'X_test': X_test_features['features'],
                'X_test_split': X_test_split_features['features'],
                'y_test_split': y_test_split['prdtypecode'].values,
                'train_indices': train_indices,
                'test_split_indices': test_split_indices,
                # 'train_samples': len(X_train),
                # 'test_samples': len(X_test_df),
                # 'test_split_samples': len(X_test_split),
                # 'n_features': X_train_features['features'].shape[1],
                # 'n_classes': len(np.unique(y_train['prdtypecode']))
                }

                return self.preprocessed_data
                
        except Exception as e:
            self.logger.error(f"Erreur préparation données: {str(e)}")
            self._cleanup()
            raise
        
    def save_model(self, model_type):
        try:
            model_dir = os.path.join(self.config.model_path, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            if isinstance(self.model, NeuralClassifier):
                torch.save({
                    'state_dict': self.model.state_dict(),
                    'category_mapping': self.category_names,
                    'config': self.model.config,
                }, os.path.join(model_dir, 'model.pth'))
            else:
                with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'params': self.model.get_params(),
                        'category_mapping': self.category_names
                    }, f)
            
            self.logger.info(f"Modèle {model_type} sauvegardé dans {model_dir}")
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle {model_type}: {str(e)}")
            raise

    def load_model(self, model_type: str):
        """Charge un modèle sauvegardé avec ses métadonnées"""
        try:
            model_dir = os.path.join(self.config.model_path, model_type)
            
            affichage_param = False # Pour afficher les paramètres des modèles chargés
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Dossier modèle non trouvé: {model_dir}")
            if model_type == 'neural_net':
                
                model_path = os.path.join(model_dir, 'model.pth')
                model_data = torch.load(model_path, map_location=self.device)
                if affichage_param:
                    # Affichage des paramètres du modèle neural_net
                    print("\nParamètres du modèle neural_net:")
                    print("=" * 50)
                    print("\nConfiguration:")
                    for key, value in model_data['config'].items():
                        print(f"{key}: {value}")
                    
                    print("\nArchitecture du modèle:")
                    print("-" * 30)
                
                # Recréation du modèle
                self.model = NeuralClassifier(
                    num_classes=len(model_data['category_mapping']),
                    config=model_data['config']
                ).to(self.device)
                
                self.model.load_state_dict(model_data['state_dict'])
                print(self.model) if affichage_param else None
                self.category_names = model_data['category_mapping']
            else:
                model_path = os.path.join(model_dir, 'model.pkl')
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.category_names = model_data['category_mapping']
                
                if affichage_param:
                    # Affichage des paramètres selon le type de modèle
                    print(f"\nParamètres du modèle {model_type}:")
                    print("=" * 50)
                    if hasattr(self.model, 'get_params'):
                        params = self.model.get_params()
                        for param_name, value in sorted(params.items()):
                            print(f"{param_name}: {value}")
                    
                    # Pour XGBoost, afficher les paramètres additionnels importants
                    if model_type == 'xgboost':
                        print("\nParamètres additionnels:")
                        print("-" * 30)
                        print(f"Nombre d'arbres: {self.model.n_estimators}")
                        if hasattr(self.model, 'feature_importances_'):
                            print("Feature importances disponibles: Oui")
            
            self.logger.info(f"Modèle {model_type} chargé depuis {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle {model_type}: {str(e)}")
            raise

    def cross_validate_model(self, model_type: str, **model_params) -> Dict[str, Any]:
        """
        Effectue une validation croisée avec support GPU et affiche la progression
        
        Args:
            model_type (str): Type de modèle à entraîner ('xgboost', 'lightgbm', 'catboost', 'logistic', 'neural_net')
            **model_params: Paramètres spécifiques au modèle
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant le meilleur modèle et son F1-score
        """
        try:
            start_time = time.time()
            self.logger.info(f"Début de la validation croisée pour {model_type}")
            
            X = self.preprocessed_data['X_train']
            y = self.preprocessed_data['y_train']
            
            # Conversion des labels pour tous les modèles
            y = np.array([self.category_to_idx[label] for label in y])
        
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_f1 = 0
            best_model = None
            
            for fold, (train_idx, val_idx) in enumerate(tqdm(list(skf.split(X, y)), desc="Validation croisée", total=5), 1):
                fold_start = time.time()
                self.logger.info(f"Début du fold {fold}/5")
                
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Initialisation du modèle en fonction du type
                if model_type == 'xgboost':
                    model = xgb.XGBClassifier(**model_params)
                    model.fit(X_fold_train, y_fold_train)
                    
                elif model_type == 'lightgbm':
                    model = LGBMClassifier(**model_params)
                    model.fit(X_fold_train, y_fold_train)
                    
                elif model_type == 'catboost':
                    model = CatBoostClassifier(**model_params)
                    model.fit(X_fold_train, y_fold_train)
                    
                elif model_type == 'logistic':
                    model = LogisticRegression(**model_params)
                    model.fit(X_fold_train, y_fold_train)
                    
                elif model_type == 'neural_net':
                    model = NeuralClassifier(
                        num_classes=len(self.category_names),
                        config=model_params
                    ).to(self.device)
                    
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=model_params.get('learning_rate', 0.001),
                        weight_decay=model_params.get('weight_decay', 0.01)
                    )
                    
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=model_params.get('scheduler_params', {}).get('factor', 0.1),
                        patience=model_params.get('scheduler_params', {}).get('patience', 3),
                        min_lr=model_params.get('scheduler_params', {}).get('min_lr', 1e-6)
                    )
                    
                    criterion = nn.CrossEntropyLoss()
                    best_val_loss = float('inf')
                    patience_counter = 0
                    
                    # Entraînement du modèle neuronal
                    for epoch in range(model_params.get('epochs', 30)):
                        model.train()
                        total_loss = 0
                        batch_count = 0
                        
                        # Phase d'entraînement
                        for batch_idx in range(0, len(X_fold_train), model_params.get('batch_size', 32)):
                            batch_end = min(batch_idx + model_params.get('batch_size', 32), len(X_fold_train))
                            batch_X = torch.FloatTensor(X_fold_train[batch_idx:batch_end]).to(self.device)
                            batch_y = torch.LongTensor(y_fold_train[batch_idx:batch_end]).to(self.device)
                            
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            batch_count += 1
                        
                        avg_loss = total_loss / batch_count
                        
                        # Phase de validation
                        model.eval()
                        val_predictions = []
                        with torch.no_grad():
                            for batch_idx in range(0, len(X_fold_val), model_params.get('batch_size', 32)):
                                batch_end = min(batch_idx + model_params.get('batch_size', 32), len(X_fold_val))
                                batch_X = torch.FloatTensor(X_fold_val[batch_idx:batch_end]).to(self.device)
                                outputs = model(batch_X)
                                _, preds = torch.max(outputs, 1)
                                val_predictions.extend(preds.cpu().numpy())
                        
                        y_pred = np.array(val_predictions)
                        val_f1 = f1_score(y_fold_val, y_pred, average='weighted')
                        
                        # Mise à jour du scheduler
                        scheduler.step(1 - val_f1)  # Utilise 1 - F1 comme métrique à minimiser
                        
                        # Early stopping
                        if val_f1 > best_val_loss:
                            best_val_loss = val_f1
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= model_params.get('early_stopping_patience', 5):
                            self.logger.info(f"Early stopping à l'époque {epoch + 1}")
                            break
                
                else:
                    raise ValueError(f"Type de modèle non supporté : {model_type}")
                
                # Calcul du F1-score pour le fold
                fold_f1 = f1_score(y_fold_val, y_pred, average='weighted')
                fold_time = time.time() - fold_start
                self.logger.info(f"Fold {fold}/5 terminé en {fold_time:.2f}s - F1-score: {fold_f1:.4f}")
                
                if fold_f1 > best_f1:
                    best_f1 = fold_f1
                    best_model = model
            
            total_time = time.time() - start_time
            self.logger.info(f"Validation croisée terminée en {total_time:.2f}s - Meilleur F1-score: {best_f1:.4f}")
            
            return {
                'model': best_model,
                'best_f1': best_f1
            }
                
        except Exception as e:
            self.logger.error(f"Erreur validation croisée: {str(e)}")
            raise

    def train_model(self, model_type='xgboost', use_cv=False, **model_params):
        """
        Choix du bon type d'entraînement selon le modèle
        """
        if model_type == 'neural_net':
            self.train_dl_model(use_cv=use_cv, **model_params)
        else:
            self.train_ml_model(model_type, use_cv=use_cv, **model_params)
        
        self.logger.info(f"Sauvegarde modèle {model_type} en cours...")
        self.save_model(model_type)

    def train_ml_model(self, model_type, use_cv, **model_params):
        """
        Entraîne un modèle ML avec les paramètres fournis
        
        Args:
            model_type (str): Type de modèle ('xgboost', 'lightgbm', 'catboost', 'logistic')
            use_cv (bool): Utiliser la validation croisée
            **model_params: Paramètres spécifiques au modèle
        """
        try:
            if not hasattr(self, 'preprocessed_data') or self.preprocessed_data is None:
                raise ValueError("Les données prétraitées ne sont pas disponibles")

            self.logger.info(f"Début entraînement {model_type}")
            start_time = time.time()

            # Préparation des données
            X_train = self.preprocessed_data['X_train']
            if isinstance(X_train, dict):
                X_train = X_train['features']
            if X_train.shape == ():
                X_train = X_train.item()['features']
                
            # Conversion des labels pour tous les modèles
            y_train = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_train']])

            if use_cv:
                result = self.cross_validate_model(model_type, **model_params)
                self.model = result['model']
            else:
                # Initialisation du modèle selon le type
                if model_type == 'xgboost':
                    self.model = xgb.XGBClassifier(**model_params)
                elif model_type == 'lightgbm':
                    X_train = np.asarray(X_train, dtype=np.float32)
                    y_train = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_train']], dtype=np.int32)  
                    
                    self.model = LGBMClassifier(**model_params)
                elif model_type == 'catboost':
                    self.model = CatBoostClassifier(**model_params)
                else:
                    raise ValueError(f"Type de modèle ML non supporté: {model_type}")

                # Entraînement
                self.model.fit(X_train, y_train)

            training_time = time.time() - start_time
            self.logger.info(f"Entraînement terminé en {training_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Erreur entraînement {model_type}: {str(e)}")
            raise

    def train_dl_model(self, use_cv, **model_params):
        """
        Entraîne le modèle deep learning
        
        Args:
            use_cv (bool): Utiliser la validation croisée
            **model_params: Paramètres du modèle
        """
        try:
            if not hasattr(self, 'preprocessed_data') or self.preprocessed_data is None:
                raise ValueError("Les données prétraitées ne sont pas disponibles")

            self.logger.info("Début entraînement deep learning")
            start_time = time.time()

            # Conversion des labels
            y_train = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_train']])
            # y_test = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_test_split']])

            # Préparation et conversion des données
            X_train_data = self.preprocessed_data['X_train']
            if isinstance(X_train_data, dict):
                X_train_data = X_train_data['features']
            if X_train_data.shape == ():
                X_train_data = X_train_data.item()['features']
            
            # Conversion en float32 pour assurer la compatibilité
            X_train_data = np.array(X_train_data, dtype=np.float32)
            
            # Création des tenseurs
            X_train_tensor = torch.FloatTensor(X_train_data)
            y_train_tensor = torch.LongTensor(y_train)
            
            train_idx, val_idx = train_test_split(
                np.arange(len(X_train_tensor)),
                test_size=0.1,
                stratify=y_train,
                random_state=self.config.random_state
            )
            
            # Création des datasets
            train_dataset = TensorDataset(
                X_train_tensor[train_idx], 
                y_train_tensor[train_idx]
            )
            val_dataset = TensorDataset(
                X_train_tensor[val_idx], 
                y_train_tensor[val_idx]
            )

            # Extraction des paramètres spécifiques au DataLoader
            dataloader_params = {
                'batch_size': model_params['batch_size'],
                'num_workers': model_params['dataloader_params']['num_workers'],
                'pin_memory': model_params['dataloader_params']['pin_memory'],
                'prefetch_factor': model_params['dataloader_params']['prefetch_factor'],
                'persistent_workers': model_params['dataloader_params']['persistent_workers'],
            }

            # Création des DataLoaders avec les paramètres filtrés
            train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
            val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

            if use_cv:
                result = self.cross_validate_model('neural_net', **model_params)
                self.model = result['model']
            else:
                # Initialisation du modèle
                self.model = NeuralClassifier(
                    num_classes=len(self.category_names),
                    config=model_params
                ).to(self.device)

                # Configuration de l'optimisation
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=model_params.get('learning_rate', 0.001),
                    weight_decay=model_params.get('weight_decay', 0.01)
                )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )

                criterion = nn.CrossEntropyLoss()
                best_val_f1 = 0
                patience_counter = 0
                
                # Boucle d'entraînement
                for epoch in range(model_params.get('epochs', 30)):
                    # Mode entraînement
                    self.model.train()
                    train_loss = 0
                    train_steps = 0

                    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_steps += 1

                    avg_train_loss = train_loss / train_steps

                    # Validation (pour early stopping)
                    self.model.eval()
                    val_predictions = []
                    val_targets = []
                    val_loss = 0
                    val_steps = 0

                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = self.model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                            val_steps += 1

                            _, predicted = torch.max(output.data, 1)
                            val_predictions.extend(predicted.cpu().numpy())
                            val_targets.extend(target.cpu().numpy())

                    val_f1 = f1_score(val_targets, val_predictions, average='weighted')
                    avg_val_loss = val_loss / val_steps

                    # Mise à jour du scheduler
                    scheduler.step(val_f1)

                    # Early stopping
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        # Sauvegarde du meilleur modèle
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1

                    self.logger.info(
                        f"Epoch {epoch+1}: "
                        f"Train Loss = {avg_train_loss:.4f}, "
                        f"Val Loss = {avg_val_loss:.4f}, "
                        f"Val F1 = {val_f1:.4f}"
                    )

                    if patience_counter >= model_params.get('early_stopping_patience', 5):
                        self.logger.info(f"Early stopping à l'epoch {epoch+1}")
                        break

                # Restauration du meilleur modèle
                self.model.load_state_dict(best_model_state)

            training_time = time.time() - start_time
            self.logger.info(f"Entraînement terminé en {training_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Erreur entraînement DL: {str(e)}")
            raise

    def predict(self, X):
        """
        Génère les prédictions pour de nouvelles données
        
        Args:
            X: Données d'entrée (numpy array ou torch tensor)
        
        Returns:
            tuple: (predictions, probabilités)
        """
        try:
            if self.model is None:
                raise ValueError("Le modèle n'est pas entraîné")

            if isinstance(X, np.ndarray) and X.shape == ():
                # Si X est un scalaire, on le convertit en array 2D
                X = X.item()
                if isinstance(X, dict) and 'features' in X:
                    X = X['features']
                X = np.array(X)
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
            print("Type de X:", type(X))
            print("Shape de X:", X.shape if hasattr(X, 'shape') else "pas de shape")
            if isinstance(X, np.ndarray):
                print("Type des données:", X.dtype)
                
            # Vérification du type de modèle
            is_dl_model = isinstance(self.model, NeuralClassifier)
            
            # Assurer que X est 2D
            if isinstance(X, (np.ndarray, list)):
                X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            # Debug après conversion
            print("Shape après conversion:", X.shape)

            if is_dl_model:
                self.model.eval()
                # Conversion en tensor si nécessaire
                if isinstance(X, np.ndarray):
                    X = torch.FloatTensor(X).to(self.device)

                predictions = []
                probabilities = []
                
                with torch.no_grad():
                    for i in range(0, len(X), self.config.batch_size):
                        batch = X[i:i + self.config.batch_size]
                        outputs = self.model(batch)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        
                        predictions.extend(preds.cpu().numpy())
                        probabilities.extend(probs.cpu().numpy())

                predictions = np.array(predictions)
                probabilities = np.array(probabilities)
            else:
                # Pour les modèles ML classiques
                self.logger.info("Prédiction avec modèle ML")
                if isinstance(self.model, xgb.XGBClassifier):
                    batch_size = 1000
                    predictions = []
                    probabilities = []
                    
                    for i in range(0, len(X), batch_size):
                        batch = X[i:i + batch_size]
                        batch_pred = self.model.predict(batch)
                        batch_prob = self.model.predict_proba(batch)
                        predictions.extend(batch_pred)
                        probabilities.extend(batch_prob)
                        
                    probabilities = np.array(probabilities)
                else:
                    predictions = self.model.predict(X)
                    probabilities = self.model.predict_proba(X)

                    # Conversion des indices en codes de catégorie
                    if isinstance(self.model, CatBoostClassifier):
                        predictions = predictions.astype(int)
                        
                predictions = np.array([self.idx_to_category[int(idx)] for idx in predictions])

            return predictions, probabilities

        except Exception as e:
            self.logger.error(f"Erreur prédiction: {str(e)}")
            raise
    
    def predictions_exist(self, model_name):
        """Vérifie si les prédictions existent déjà"""
        pred_path = os.path.join('data', f'predictions_{model_name}.csv')
        return os.path.exists(pred_path)

    def save_predictions(self, model_name, predictions, probabilities):
        """Sauvegarde les prédictions"""
        # 1. DataFrame pour la colonne 'prediction'
        df_pred = pd.DataFrame({'prediction': predictions})
        
        # 2. DataFrame pour les probabilités, 27 colonnes
        df_proba = pd.DataFrame(probabilities, columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])])
        
        # 3. Concaténer les deux en un seul DataFrame
        df_final = pd.concat([df_pred, df_proba], axis=1)
        df_final.to_csv(os.path.join('data/predictions', f'predictions_{model_name}.csv'), index=False)

    def load_predictions(self, model_name):
        pred_df = pd.read_csv(os.path.join('data/predictions', f'predictions_{model_name}.csv'))
        predictions = pred_df['prediction'].values
        
        # On récupère toutes les colonnes de probas
        proba_cols = [c for c in pred_df.columns if c.startswith('prob_class_')]
        probabilities = pred_df[proba_cols].values  # shape (N, 27)
        
        return predictions, probabilities

    def _compute_gradcam(self, X=None):
        """Calcule Grad-CAM pour un modèle de deep learning"""
        try:
            if X is None:
                X = self.preprocessed_data['X_test_split']
                # Gestion du cas où X est un dictionnaire
                if isinstance(X, dict):
                    X = X['features']
                # Gestion du cas où X est un tableau 0-dimensionnel
                if X.shape == ():
                    X = X.item()['features']
            
            self.model.eval()
            resultats_gradcam = []
            
            # Conversion en tensor si nécessaire
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            
            # Traitement par lots pour optimiser la mémoire
            taille_lot = 32
            for i in range(0, len(X), taille_lot):
                lot = X[i:i + taille_lot]
                lot = lot.to(self.device)
                lot.requires_grad = True
     
                # Passage avant
                sortie = self.model(lot)
                
                # Utiliser toute la sortie pour le calcul du gradient
                pertes = sortie.sum()
                pertes.backward()
                
                # S'assurer que les gradients ont la bonne forme
                gradients = lot.grad
                
                # Moyenne sur les dimensions appropriées
                poids = torch.mean(gradients, dim=[0])
                
                # Normalisation
                poids = F.relu(poids)
                poids = poids / (torch.norm(poids) + 1e-5)
                
                resultats_gradcam.append(poids.cpu().detach().numpy())
                gradcam_results = np.stack(resultats_gradcam, axis=0)
                
            return {
                'gradcam_mean': np.mean(gradcam_results),
                'gradcam_std': np.std(gradcam_results)
            }
            
        except Exception as e:
            self.logger.warning(f"Impossible de calculer GradCAM : {str(e)}")
            return {}  # Retourne un dictionnaire vide en cas d'échec

    def evaluate(self, use_gradcam=False):
        """Évalue le modèle avec métriques complètes incluant les probabilités"""
        X_test = self.preprocessed_data['X_test_split']
        X_test = X_test['features'] if isinstance(X_test, dict) else X_test
        print("X_test shape:", X_test.shape)        

        y_pred_, probas = self.predict(X_test)
        if isinstance(self.model, NeuralClassifier):
            y_test = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_test_split']])
            y_pred = y_pred_
            y_true = y_test
        elif isinstance(self.model, LGBMClassifier):
            y_test = np.array([self.category_to_idx[label] for label in self.preprocessed_data['y_test_split']], dtype=np.int32)
            y_pred = np.array([self.category_to_idx[pred] for pred in y_pred_], dtype=np.int32).ravel()
            y_true = y_test.ravel()
            
            # Création du dictionnaire inverse une seule fois
            idx_to_category = {v: k for k, v in self.category_to_idx.items()}
        else:
            # Code original pour les autres modèles
            y_test = self.preprocessed_data['y_test_split']
            y_pred = np.array(y_pred_).ravel()
            y_true = np.array(y_test).ravel()
        
        print("y_test shape:", y_test.shape)
        present_classes = np.intersect1d(np.unique(y_true), np.unique(y_pred))
        
        def _round_metric_value(value):
            """Arrondit les valeurs numériques à 0.001"""
            if isinstance(value, np.ndarray):
                return np.round(value, 3)
            elif isinstance(value, (float, np.float32, np.float64)):
                return round(value, 3)
            else: 
                print("type value:", type(value))
            return value
        
        # Métriques basées sur les prédictions dures
        metrics = {
            'accuracy': _round_metric_value(accuracy_score(y_true, y_pred)),
            'macro_precision': _round_metric_value(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_recall': _round_metric_value(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'macro_f1': _round_metric_value(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_precision': _round_metric_value(precision_score(y_true, y_pred, average='weighted', labels=present_classes, zero_division=0)),
            'weighted_recall': _round_metric_value(recall_score(y_true, y_pred, average='weighted', labels=present_classes, zero_division=0)),
            'weighted_f1': _round_metric_value(f1_score(y_true, y_pred, average='weighted', labels=present_classes, zero_division=0)),
            'precision': _round_metric_value(precision_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)),
            'recall': _round_metric_value(recall_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)),
            'f1': _round_metric_value(f1_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0))
        }

        # Métriques basées sur les probabilités
        max_probas = np.max(probas, axis=1)
        metrics.update({
            'mean_confidence': _round_metric_value(np.mean(max_probas)),
            'median_confidence': _round_metric_value(np.median(max_probas)),
            'min_confidence': _round_metric_value(np.min(max_probas)),
            'low_confidence_samples': int(np.sum(max_probas < 0.5)),
            'high_confidence_samples': int(np.sum(max_probas > 0.8)) 
        })

        # Métriques par classe
        for classe in np.unique(y_true):
            classe_mask = (y_true == classe)
            
            
            if isinstance(self.model, (NeuralClassifier, LGBMClassifier)):
                # Convertion de l'indice (0-26) en code de catégorie (10, 40, etc.)
                real_code = self.idx_to_category[classe]  # Convertit l'indice en code de catégorie
                category_name = self.category_names[real_code]  # Obtient le nom depuis le code
                proba_idx = classe  # Garde l'indice original pour les probabilités
            else:
                # Pour les autres modèles, on utilise directement le code de catégorie
                real_code = classe
                category_name = self.category_names[real_code]
                proba_idx = self.category_to_idx[real_code]
            
            # Créer le préfixe combiné code_nom
            metric_prefix = f"{real_code}_{category_name}"
            
            # Calculer les métriques
            metrics[f'{metric_prefix}_precision'] = _round_metric_value(
                precision_score(y_true, y_pred, labels=[classe], average='micro', zero_division=0)
            )
            metrics[f'{metric_prefix}_recall'] = _round_metric_value(
                recall_score(y_true, y_pred, labels=[classe], average='micro', zero_division=0)
            )
            metrics[f'{metric_prefix}_f1'] = _round_metric_value(
                f1_score(y_true, y_pred, labels=[classe], average='micro', zero_division=0)
            )
            
            if np.sum(classe_mask) > 0:
                class_probas = probas[classe_mask, proba_idx]
                metrics[f'{metric_prefix}_mean_confidence'] = _round_metric_value(np.mean(class_probas))
                correct_mask = y_pred[classe_mask] == classe
                if np.sum(correct_mask) > 0:
                    metrics[f'{metric_prefix}_correct_high_confidence'] = _round_metric_value(
                        np.mean(class_probas[correct_mask] > 0.8)
                    )
                else:
                    metrics[f'{metric_prefix}_correct_high_confidence'] = 0.0
            else:
                metrics[f'{metric_prefix}_mean_confidence'] = 0.0
                metrics[f'{metric_prefix}_correct_high_confidence'] = 0.0

        # Interprétabilité
        if use_gradcam and isinstance(self.model, NeuralClassifier):
            metrics.update(self._compute_gradcam())

        return metrics

    def _cleanup(self):
        """Nettoie les fichiers temporaires en cas d'erreur"""
        try:
            temp_dirs = ['temp_train', 'temp_test']
            for dir_name in temp_dirs:
                dir_path = os.path.join(self.config.data_path, dir_name)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        except Exception as e:
            self.logger.warning(f"Erreur nettoyage : {str(e)}")