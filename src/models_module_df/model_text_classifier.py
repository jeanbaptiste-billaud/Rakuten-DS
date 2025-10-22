# model_text_classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class TextClassifier:
    def __init__(self, model=None):

        if model is None:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=45000)),
                ('svm', SVC(C=12, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'))
            ])

            self.model = pipeline
        else:
            self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs

    def evaluate(self, X_test, y_true ):
        preds, probs = self.predict(X_test)

        metrics =  {
            'accuracy': round(accuracy_score(y_true, preds), 4),
            'weighted_f1': round(f1_score(y_true, preds, average='weighted'), 4),
            'macro_f1': round(f1_score(y_true, preds, average='macro'), 4),
            'mean_confidence': round(probs.max(axis=1).mean(), 4)
        }

        cm = confusion_matrix(y_true, preds).tolist()
        cr = classification_report(y_true, preds, output_dict=True)

        return metrics, cm, cr
