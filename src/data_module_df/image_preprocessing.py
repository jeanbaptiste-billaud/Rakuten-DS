import os
from src.common_utils import create_version_folder
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

with open("configs/config.yaml", "r") as f:
    config = yaml.load(f)

v = config["version"]
data_path = config["data"]["processed_dir"]
new_version_folder = create_version_folder(v, data_path)
os.makedirs(os.path.join(new_version_folder, "images"), exist_ok=True)