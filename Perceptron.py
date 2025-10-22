import numpy as np
import joblib  # để lưu và tải model

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.unique_labels = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.unique_labels = np.unique(y)

        for epoch in range(self.epochs):
            for i in range(num_samples):
                prediction = self._predict_instance(X[i])
                update = self.learning_rate * (self._label_to_numeric(y[i]) - prediction)
                self.weights += update * X[i]
                self.bias += update

    def _label_to_numeric(self, label):
        return np.where(self.unique_labels == label)[0][0]

    def _predict_instance(self, instance):
        activation = np.dot(instance, self.weights) + self.bias
        return 1 if activation >= 0 else 0

    def predict(self, X):
        predictions = np.apply_along_axis(self._predict_instance, 1, X)
        return np.array([self.unique_labels[label] for label in predictions])

    # ✅ Lưu model
    def save_model(self, path="perceptron_model.pkl"):
        joblib.dump({
            "weights": self.weights,
            "bias": self.bias,
            "unique_labels": self.unique_labels
        }, path)
        print(f"✅ Model saved to {path}")

    # ✅ Load model
    def load_model(self, path="perceptron_model.pkl"):
        data = joblib.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.unique_labels = data["unique_labels"]
        print(f"✅ Model loaded from {path}")
