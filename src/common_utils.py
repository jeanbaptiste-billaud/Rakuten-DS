# common_utils.py
import os


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


def get_project_root():
    """Retourne le chemin absolu vers la racine du projet (Rakuten-DS)."""
    return os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))



#todo: ajouter une fonction de logging boto pour minio, voir artifact_test