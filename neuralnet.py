import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def create_weight_matrix(nrows, ncols):
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    return create_weight_matrix(length, 1)

class ActivationFunction(ABC):
    @abstractmethod
    def f(self, x):
        pass
    @abstractmethod
    def df(self, x):
        pass

class LeakyReLU(ActivationFunction):
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param
    def f(self, x):
        return np.maximum(x, x*self.alpha)
    def df(self, x):
        return np.where(x > 0, 1, self.alpha)

class ReLU(ActivationFunction):
    def f(self, x):
        return np.maximum(0, x)
    def df(self, x):
        return np.where(x > 0, 1, 0)

class Tanh(ActivationFunction):
    def f(self, x):
        return np.tanh(x)
    def df(self, x):
        return 1 - np.tanh(x)**2

class Sigmoid(ActivationFunction):
    def f(self, x):
        return 1/(1 + np.exp(-x))
    def df(self, x):
        return self.f(x) * (1 - self.f(x))

class LossFunction(ABC):
    @abstractmethod
    def loss(self, values, expected):
        pass

    @abstractmethod
    def dloss(self, values, expected):
        pass

class MSELoss(LossFunction):
    def loss(self, values, expected):
        return np.mean((values - expected)**2)
    def dloss(self, values, expected):
        return 2*(values - expected)/values.size

class CrossEntropyLoss(LossFunction):
    def loss(self, values, target):
        m = target.shape[1]
        p = np.exp(values) / np.sum(np.exp(values), axis=0, keepdims=True)
        log_likelihood = -np.log(p[target, range(m)])
        loss = np.sum(log_likelihood) / m
        return loss

    def dloss(self, values, target):
        m = target.shape[1]
        p = np.exp(values) / np.sum(np.exp(values), axis=0, keepdims=True)
        p[target, range(m)] -= 1
        return p / m

class LogLoss(LossFunction):
    def loss(self, values, expected):
        return -np.sum(expected * np.log(values + 1e-15))
    def dloss(self, values, expected):
        return -(expected - values)

class Layer:
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_weight_matrix(self.outs, self.ins)
        self._b = create_bias_vector(self.outs)

    def forward_pass(self, x):
        self.last_z = np.dot(self._W, x) + self._b
        self.last_a = self.act_function.f(self.last_z)
        return self.last_a

    def backward_pass(self, dA, x, learning_rate):
        dZ = dA * self.act_function.df(self.last_z)
        dW = np.dot(dZ, x.T) / x.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / x.shape[1]
        dA_prev = np.dot(self._W.T, dZ)
        self._W -= learning_rate * dW
        self._b -= learning_rate * db
        return dA_prev

class NeuralNetwork:
    def __init__(self, layers, loss_function, learning_rate):
        self._layers = layers
        self._loss_function = loss_function
        self.lr = learning_rate
        for (from_, to_) in zip(self._layers[:-1], self._layers[1:]):
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")
                
    def forward_pass(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out

    def loss(self, values, expected):
        return self._loss_function.loss(values, expected)

    def train(self, x, t):
        activations = [x]
        for layer in self._layers:
            activations.append(layer.forward_pass(activations[-1]))
        
        dx = self._loss_function.dloss(activations[-1], t)
        for layer, a in zip(self._layers[::-1], activations[:-1][::-1]):
            dx = layer.backward_pass(dx, a, self.lr)

# if __name__ == "__main__":
#     # Load MNIST dataset
#     mnist = fetch_openml('mnist_784')
#     X = mnist.data.values / 255.0 
#     y = mnist.target.astype(int).values.reshape(-1, 1) 
    
#     enc = OneHotEncoder()
#     y = enc.fit_transform(y).toarray().T
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=42)
#     X_train, X_test = X_train.T, X_test.T
#     y_train, y_test = y_train.T, y_test.T

#     net = NeuralNetwork([
#         Layer(784, 128, LeakyReLU()),
#         Layer(128, 64, LeakyReLU()),
#         Layer(64, 10, Sigmoid()),
#     ], LogLoss(), 0.01)

#     epochs = 1000
#     batch_size = 64
#     for epoch in range(epochs):
#         for i in range(0, X_train.shape[1], batch_size):
#             x_batch = X_train[:, i:i+batch_size]
#             t_batch = y_train[:, i:i+batch_size]
#             net.train(x_batch, t_batch)
#         print(f"Epoch {epoch+1}/{epochs} completed")

#     # Testing the network
#     predictions = []
#     for i in range(X_test.shape[1]):
#         x = X_test[:, i].reshape(-1, 1)
#         out = net.forward_pass(x)
#         predictions.append(np.argmax(out))
#     print(predictions)
#     y_test_labels = np.argmax(y_test, axis=0)
#     accuracy = accuracy_score(y_test_labels, predictions)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
if __name__ == "__main__":
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784')
    X = mnist.data.values / 255.0  # Convert to numpy array
    y = mnist.target.astype(int).values.reshape(-1, 1)  # Convert to numpy array before reshaping
    
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray().T

    # Apply PCA
    # pca = PCA(n_components=100)
    # X_reduced = pca.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=42)
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T

    # Create the neural network
    net = NeuralNetwork([
        Layer(784, 500, LeakyReLU()),
        Layer(500, 100, LeakyReLU()),
        Layer(100, 20,  LeakyReLU()),
        Layer(20, 10, Sigmoid()),
    ], MSELoss(), 0.01)

    # Training the network
    epochs = 1000
    batch_size = 64
    # for epoch in range(epochs):
    #     for i in range(0, X_train.shape[1], batch_size):
    #         x_batch = X_train[:, i:i+batch_size]
    #         t_batch = y_train[:, i:i+batch_size]
    #         net.train(x_batch, t_batch)
    #     print(f"Epoch {epoch+1}/{epochs} completed")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in tqdm(range(0, X_train.shape[1], batch_size), desc="Training"):
            x_batch = X_train[:, i:i+batch_size]
            t_batch = y_train[:, i:i+batch_size]
            net.train(x_batch, t_batch)
    # Testing the network
    predictions = []
    for i in range(X_test.shape[1]):
        x = X_test[:, i].reshape(-1, 1)
        out = net.forward_pass(x)
        predictions.append(np.argmax(out))

    y_test_labels = np.argmax(y_test, axis=0)
    accuracy = accuracy_score(y_test_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")