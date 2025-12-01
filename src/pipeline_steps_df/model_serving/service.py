# src/bentoml_module/service.py
import os

import bentoml
import numpy as np
import pandas as pd

from bentoml.models import BentoModel
from bentoml.io import JSON, Text
from src.pipeline_steps_df.data import preprocessing
from pydantic import BaseModel

from src.utils.common_utils import get_project_root

demo_image = bentoml.images.PythonImage(python_version="3.12") \
    .python_packages("mlflow", "scikit-learn", "numpy", "pandas", "spacy")

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextClassifier:
    bento_model = BentoModel("rakuten_text_classifier:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)
        self.preprocess = preprocessing.main

        self.root = os.getenv("WORKDIR", get_project_root())
        self.data_dir = os.path.join(self.root, "data")

    # Define an API endpoint
    @bentoml.api()
    def predict(self, input_data: ) -> list[str]:

        self.preprocess()

        input_data_file = os.path.join(self.data_dir, "preprocessed", "preprocessed_text.csv")
        input_data = pd.read_csv(input_data_file)

        preds = self.model.predict(input_data)
        return [target_names[i] for i in preds]