import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class Layer:
    def __init__(self, input_size, output_size, activation="tanh"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def activate(self, Z):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation == "tanh":
            return np.tanh(Z)
        else:
            raise ValueError("Unsupported activation function")

    def activate_derivative(self, A):
        if self.activation == "sigmoid":
            return A * (1 - A)
        elif self.activation == "tanh":
            return 1 - np.power(A, 2)
        else:
            raise ValueError("Unsupported activation function")


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, layers: list[Layer] = [], learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = layers
        self.scaler = StandardScaler()

    def _forward_propagation(self, X) -> list[np.ndarray]:
        activations = [X]
        A = X
        for layer in self.layers:
            Z = np.dot(A, layer.W) + layer.b
            A = layer.activate(Z)
            activations.append(A)
        return activations

    def _backward_propagation(
        self, X, y, activations
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        m = X.shape[0]
        dZ = activations[-1] - y
        grads: list[tuple[np.ndarray, np.ndarray]] = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = activations[i]
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            grads.insert(0, (dW, db))
            if i > 0:
                dA_prev = np.dot(dZ, layer.W.T)
                dZ = dA_prev * layer.activate_derivative(activations[i])

        return grads

    def _update_parameters(self, grads):
        for i, layer in enumerate(self.layers):
            dW, db = grads[i]
            layer.W -= self.learning_rate * dW
            layer.b -= self.learning_rate * db

    def _compute_loss(self, y, y_hat) -> float:
        m = y.shape[0]
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        loss = -np.sum(y * np.log(y_hat)) / m
        return loss

    def fit(self, X, y, log: int = 0):
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)

        self.input_size_ = X.shape[1]
        self.output_size_ = y.shape[1]

        for epoch in range(self.epochs):
            activations = self._forward_propagation(X)
            grads = self._backward_propagation(X, y, activations)
            self._update_parameters(grads)

            if log and epoch % log == 0:
                print(f"Epoch: {epoch}", self._compute_loss(y, activations[-1]))

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        activations = self._forward_propagation(X)
        return activations[-1]

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
