# src/bentoml_module/service.py

import bentoml

from bentoml.models import BentoModel
from bentoml.io import JSON
from pydantic import BaseModel

demo_image = bentoml.images.PythonImage(python_version="3.12") \
    .python_packages("mlflow", "scikit-learn")

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextClassifier:
    bento_model = BentoModel("rakuten_text_classifier:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    # Define an API endpoint
    @bentoml.api
    def predict(self, input_data: np.ndarray) -> list[str]:

        preds = self.model.predict(input_data)
        return [target_names[i] for i in preds]