# data_text.py
from src.data_module_df.data_balancing import undersample, print_class_distribution
from src.data_module_df.data_indexing import load_indices
from src.data_module_df.data_loader import load_training_data, save_processed_npz


def prepare_text_data(config, force=False):
    """
    Prépare les données texte (designation + description) selon les splits d’indices
    et applique un équilibrage sur le jeu d'entraînement.
    """
    print("\n🔄 Préparation des données texte...")
    X_df, y_df = load_training_data(config.data_path)
    indices = load_indices()

    # Fusion du texte
    X_df['text'] = X_df['designation'].fillna('') + ' ' + X_df['description'].fillna('')
    X_all = X_df['text'].values
    y_all = y_df['prdtypecode'].values

    # train
    X_train_text = X_all[indices['train_idx']]
    y_train = y_all[indices['train_idx']]
    print_class_distribution(y_train, "Avant équilibrage (train)")
    X_train_text, y_train = undersample(X_train_text, y_train, max_per_class=1000)
    print_class_distribution(y_train, "Après équilibrage (train)")

    # val
    X_val_text = X_all[indices['val_idx']]
    y_val = y_all[indices['val_idx']]

    # test
    X_test_text = X_all[indices['test_idx']]
    y_test = y_all[indices['test_idx']]

    # Save
    save_processed_npz({'text': X_train_text, 'labels': y_train}, 'text_train')
    save_processed_npz({'text': X_val_text, 'labels': y_val}, 'text_val')
    save_processed_npz({'text': X_test_text, 'labels': y_test}, 'text_test')

    print("✅ Données texte prêtes : textes alignés et équilibrés.")
    return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test
