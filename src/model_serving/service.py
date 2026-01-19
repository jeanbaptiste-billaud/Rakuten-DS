import os
import json
import bentoml
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.data_module_df.data_balancing import Categories
from src.data_module_df.data_text import preprocess_dataframe
from src.utils.common_utils import get_project_root

api = FastAPI()


def load_model_ref():
    model_tag = Path("bento_model_tag.txt").read_text().strip()
    return model_tag, bentoml.models.get(model_tag)


def build_metadata_payload() -> dict:
    # source de vérité : tag du modèle packagé
    model_tag, model_ref = load_model_ref()
    md = model_ref.info.metadata or {}

    run_id = None
    p = Path("mlflow_run_id.txt")
    if p.exists():
        run_id = p.read_text().strip()
    run_id = run_id or md.get("mlflow.run_id") or md.get("mlflow_run_id")

    return {
        "status": "ok",
        "service": "TextClassifier",
        "bento_model_tag": str(model_ref.tag),
        "model_name": model_ref.tag.name,
        "model_version": model_ref.tag.version,
        "mlflow_run_id": run_id,
        "model_uri": md.get("mlflow.model_uri") or md.get("mlflow_uri"),
        "model_id": md.get("model_id"),
    }


@api.get("/metadata")
def metadata_get():
    return JSONResponse(build_metadata_payload())


@api.get("/model_perf")
def model_perf_get():
    metrics_path = Path("metrics_text.json")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return JSONResponse(metrics)


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
@bentoml.asgi_app(api)
class TextClassifier:
    def __init__(self):
        self.model_tag, self.model_ref = load_model_ref()
        self.model = bentoml.sklearn.load_model(self.model_ref)

        self.root = os.getenv("WORKDIR", get_project_root())
        self.data_dir = os.path.join(self.root, "data")

    @staticmethod
    def preprocess(text):
        df = pd.DataFrame({"designation_description": text})
        df = preprocess_dataframe(df, "designation_description")
        return df

    @bentoml.api
    def predict(self, text: list[str]) -> dict:
        df_preprocessed = self.preprocess(text)
        X = df_preprocessed["text_cleaned"].astype(str)

        preds = self.model.predict(X)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        cats = Categories()

        # Ici preds_list contient déjà des prdtypecode
        prdtypecodes = [int(p) for p in preds_list]

        categories = [cats.category_names.get(c, "UNKNOWN_CATEGORY") for c in prdtypecodes]

        return {
            "input": text,
            "prdtypecode": prdtypecodes,
            "category": categories,
        }

    @bentoml.api()
    def metadata_post(self) -> dict:
        return build_metadata_payload()
