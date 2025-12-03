# src/model_serving/service.py
import os
import numpy as np
import bentoml
import pandas as pd

from bentoml.images import Image

from src.data_module_df.data_text import preprocess_dataframe
from src.data_module_df.data_balancing import Categories
from src.utils.common_utils import get_project_root


def find_model_by_metadata(name: str, key: str, value: str):
    """
    Retourne directement l'objet BentoModel correspondant,
    ou lève une exception si aucun modèle ne correspond.
    """
    for m in bentoml.models.list(name):
        if m.info.metadata.get(key) == value:
            return m
    raise RuntimeError(
        f"Aucun modèle BentoML trouvé pour {name} "
        f"avec metadata[{key}] == {value}"
    )


demo_image = (
    Image(python_version="3.12")
    .python_packages("mlflow", "scikit-learn", "numpy", "pandas", "spacy")
)


@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextClassifier:

    # On récupère l'objet BentoModel directement
    model_ref = find_model_by_metadata(
        name="rakuten_text_classifier",
        key="mlflow_run_id",
        value=os.getenv("MLFLOW_RUN_ID"),
    )

    def __init__(self):
        # On charge le modèle MLflow packagé dans Bento
        self.model = bentoml.mlflow.load_model(self.model_ref)

        self.root = os.getenv("WORKDIR", get_project_root())
        self.data_dir = os.path.join(self.root, "data")

    @staticmethod
    def preprocess(text):
        df = pd.DataFrame({"designation_description": text})
        df = preprocess_dataframe(df, "designation_description")
        return df

    @bentoml.api
    def predict(self, text: list[str]) -> dict:

        # Prétraitement
        df_preprocessed = self.preprocess(text)
        X = df_preprocessed["text_cleaned"].astype(str)

        # Prédictions brutes
        preds = self.model.predict(X)

        cats = Categories()

        # Si le modèle renvoie des indices internes → conversion
        if isinstance(preds[0], (int, np.integer)) and preds[0] in cats.idx_to_category:
            prdtypecodes = [cats.idx_to_category[p] for p in preds]
        else:
            prdtypecodes = preds.tolist()

        # Conversion en noms lisibles
        categories = [cats.category_names[code] for code in prdtypecodes]

        return {
            "input": text,
            "prdtypecode": prdtypecodes,
            "category": categories,
            "n_samples": len(text),
        }
