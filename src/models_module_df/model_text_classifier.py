# model_text_classifier.py
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

class TextClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def train(self, X_text, y):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=45000)),
            ('svm', SVC(C=12, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'))
        ])
        pipeline.fit(X_text, y)
        self.model = pipeline

    def predict(self, X_text):
        preds = self.model.predict(X_text)
        probs = self.model.predict_proba(X_text)
        return preds, probs

    def evaluate(self, y_true, preds, probs):
        return {
            'accuracy': round(accuracy_score(y_true, preds), 4),
            'weighted_f1': round(f1_score(y_true, preds, average='weighted'), 4),
            'macro_f1': round(f1_score(y_true, preds, average='macro'), 4),
            'mean_confidence': round(probs.max(axis=1).mean(), 4)
        }

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.model = joblib.load(self.model_path)
