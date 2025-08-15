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


def create_version_folder(v_num, root_path):
    """
    Crée un dossier de version v{v_num + 1} dans root_path,
    et met à jour le lien symbolique 'latest' pour qu'il pointe dessus.

    :param v_num: Dernier numéro de version (int)
    :param root_path: Chemin vers le dossier racine où les versions sont stockées
    """
    new_version_folder = os.path.join(root_path, f"v{v_num + 1}")

    os.makedirs(new_version_folder, exist_ok=True)
    create_symlink_folder(v_num + 1, root_path)

    return new_version_folder


def create_symlink_folder(v_num, root_path):
    version_folder = os.path.join(root_path, f"v{v_num}")
    link_name = os.path.join(root_path, "latest")

    try:
        if os.path.islink(link_name):
            existing_target = os.readlink(link_name)
            if existing_target == version_folder:
                return  # Le lien est déjà correct
            else:
                os.unlink(link_name)  # Supprimer l'ancien lien
                os.symlink(version_folder, link_name)
        elif os.path.exists(link_name):
            raise FileExistsError(f"Le chemin '{link_name}' existe déjà et n'est pas un lien symbolique.")
        else:
            os.symlink(version_folder, link_name)
    except Exception as e:
        print(f"Erreur lors de la création du lien symbolique : {e}")
