# classification.py — Traffic Sign Classification

import os
import pickle
import logging
from typing import Tuple
import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.feature import hog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, feature_type='hog', color_hist_bins=32, target_size=(64,64)):
        self.feature_type = feature_type
        self.color_hist_bins = color_hist_bins
        self.target_size = target_size

        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8,8),
            'cells_per_block': (2,2),
            'block_norm': 'L2-Hys',
            'feature_vector': True
        }

    def extract_hog(self, image):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, self.target_size)
        return hog(gray, **self.hog_params).astype(np.float32)

    def extract(self, image):
        if image is None:
            raise ValueError("Invalid image")
        return self.extract_hog(image)

    def extract_batch(self, images):
        return np.stack([self.extract(img) for img in tqdm(images)])

class TrafficSignClassifier:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = GaussianNB()          # Changed from SVC to GaussianNB
        self.encoder = LabelEncoder()
        self.is_trained = False

    def train(self, images, labels):
        labels = self.encoder.fit_transform(labels)
        X = self.feature_extractor.extract_batch(images)

        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_val)

        self.is_trained = True

        return {
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, average='weighted', zero_division=0),
            "recall": recall_score(y_val, preds, average='weighted', zero_division=0),
            "f1": f1_score(y_val, preds, average='weighted', zero_division=0)
        }

    def predict(self, images):
        if isinstance(images, np.ndarray):
            images = [images]

        X = self.feature_extractor.extract_batch(images)
        X = self.scaler.transform(X)

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)   # GaussianNB supports this natively

        return preds, np.max(probs, axis=1)

    def save(self, path: str, class_names: list = None):
        """Save the trained classifier (model + scaler + encoder + class names) to disk."""
        data = {
            "model":       self.model,
            "scaler":      self.scaler,
            "encoder":     self.encoder,
            "class_names": class_names or [],
            "is_trained":  self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Classifier saved → {path}")

    @classmethod
    def load(cls, path: str):
        """Load a previously saved classifier from disk.

        Returns:
            (classifier, class_names)  — ready to call .predict() on.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        clf = cls()
        clf.model      = data["model"]
        clf.scaler     = data["scaler"]
        clf.encoder    = data["encoder"]
        clf.is_trained = data["is_trained"]
        class_names    = data.get("class_names", [])
        logger.info(f"Classifier loaded ← {path}")
        return clf, class_names

def demo():
    images = []
    labels = []

    for i in range(100):
        img = np.zeros((64,64,3), dtype=np.uint8)
        label = i % 2

        if label == 0:
            cv2.circle(img,(32,32),20,(0,0,255),-1)
        else:
            cv2.rectangle(img,(10,10),(50,50),(255,0,0),-1)

        images.append(img)
        labels.append(label)

    clf = TrafficSignClassifier()
    print(clf.train(images, labels))

    preds, confs = clf.predict(images[:5])
    print(preds, confs)

if __name__ == "__main__":
    demo()
