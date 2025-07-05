import gdown
import logging
import os

def main():
    logger = logging.getLogger(__name__)
    logger.info("Téléchargement et extraction des données d'origine")

    url = "https://drive.google.com/uc?id=1vEyOwpFI6Ypv_H4Akaf6bulUjNKV1eJo"
    output = "data/raw/v1.zip"
    gdown.cached_download(
        url,
        path=output,
        postprocess=gdown.extractall
    )

    if os.listdir("data/raw/v1"):
        os.remove(output)

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()