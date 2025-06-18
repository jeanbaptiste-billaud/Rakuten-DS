# common_utils.py
import os
import zipfile
import urllib.request
import gdown


def download_dataset_if_needed(zip_url, extract_dir, expected_files):
    """
    Télécharge un fichier zip si les fichiers attendus n'existent pas, puis l'extrait.
    """
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(extract_dir, f))]
    if not missing_files:
        print("✅ Données déjà présentes.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(extract_dir, 'dataset.zip')

    print("⬇️ Téléchargement du jeu de données...")
    urllib.request.urlretrieve(zip_url, zip_path)

    print("📦 Extraction des données...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)
    print("✅ Téléchargement et extraction terminés.")


def download_and_extract_from_drive(drive_url, dest_dir):
    """
    Télécharge et extrait une archive ZIP hébergée sur Google Drive.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "preprocessed_data.zip")

    print("⬇️ Téléchargement depuis Google Drive...")
    gdown.download(drive_url, zip_path, quiet=False)

    print("📦 Extraction des données...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    os.remove(zip_path)
    print("✅ Données extraites dans", dest_dir)
