# src/model_serving/service.py
import os
import numpy as np
import bentoml
import pandas as pd

from bentoml.images import Image

from src.data_module_df.data_text import preprocess_dataframe
from src.data_module_df.data_balancing import Categories
from src.utils.common_utils import get_project_root


def find_model_by_metadata(name: str, keys: list[str], value: str):
    """
    Retourne l'objet BentoModel correspondant en testant plusieurs clés de metadata.
    """
    for m in bentoml.models.list(name):
        md = m.info.metadata or {}
        if any(md.get(k) == value for k in keys):
            return m
    raise RuntimeError(
        f"Aucun modèle BentoML trouvé pour {name} "
        f"avec metadata[{keys}] == {value}"
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
        keys=["mlflow.run_id", "mlflow_run_id", "mlflow_run_id".replace("_", ".")],  # safe
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

    @bentoml.api
    def healthz(self) -> dict:
        return {
            "status": "ok",
            "model_name": self.model_ref.tag.name,
            "model_version": self.model_ref.tag.version,
        }

    @bentoml.api
    def metadata(self) -> dict:
        md = self.model_ref.info.metadata or {}

        # run_id: priorités -> env, puis metadata (2 conventions)
        run_id = (
                os.getenv("MLFLOW_RUN_ID")
                or md.get("mlflow.run_id")
                or md.get("mlflow_run_id")
        )

        model_uri = (
                os.getenv("MODEL_URI")
                or md.get("mlflow.model_uri")
                or md.get("mlflow_uri")  # chez toi, build_step met "mlflow_uri"
        )

        return {
            "status": "ok",
            "service": "TextClassifier",
            "mlflow_run_id": run_id,
            "model_uri": model_uri,
            "model_id": md.get("model_id"),
            "bento_model_tag": str(self.model_ref.tag),
            "model_name": self.model_ref.tag.name,
            "model_version": self.model_ref.tag.version,
        }
