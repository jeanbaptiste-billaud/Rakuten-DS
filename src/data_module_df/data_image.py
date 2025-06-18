# data_image.py
from src.data_module_df.data_balancing import undersample, print_class_distribution
from src.data_module_df.data_indexing import load_indices
from src.data_module_df.data_loader import load_training_data, save_processed_npz
from src.models_module_df.model_feature_extractor import FeatureExtractor


def prepare_image_data(config, force=False):
    """
    Prépare les features image (X) et labels (y) à partir des chemins image et indices stratifiés.
    """
    print("\n🔄 Préparation des données image...")
    X_df, y_df = load_training_data()
    indices = load_indices()
    y = y_df['prdtypecode'].values

    image_paths = X_df['image_path'].values
    y_all = y

    # train
    X_train_paths = image_paths[indices['train_idx']]
    y_train = y_all[indices['train_idx']]
    print_class_distribution(y_train, "Avant équilibrage (train)")
    X_train_paths, y_train = undersample(X_train_paths, y_train, max_per_class=1000)
    print_class_distribution(y_train, "Après équilibrage (train)")

    # val
    X_val_paths = image_paths[indices['val_idx']]
    y_val = y_all[indices['val_idx']]

    # test
    X_test_paths = image_paths[indices['test_idx']]
    y_test = y_all[indices['test_idx']]

    # Feature extraction
    extractor = FeatureExtractor()
    X_train = extractor.extract_from_image_paths(X_train_paths)
    X_val = extractor.extract_from_image_paths(X_val_paths)
    X_test = extractor.extract_from_image_paths(X_test_paths)

    # Save
    save_processed_npz({'features': X_train, 'labels': y_train}, 'X_train')
    save_processed_npz({'features': X_val, 'labels': y_val}, 'X_val')
    save_processed_npz({'features': X_test, 'labels': y_test}, 'X_test')

    print("✅ Données image prêtes : features extraites et sauvegardées.")
    return X_train, X_val, X_test, y_train, y_val, y_test
