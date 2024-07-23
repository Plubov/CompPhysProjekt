import numpy as np
from scipy.integrate import odeint

# Definiere die Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Definiere die neuronalen Netze mit 5 Neuronen im versteckten Layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, I0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.I0 = I0

        # Gewichte initialisieren
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

        # Biases initialisieren
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.output_size)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.A1 = np.maximum(self.A1, self.I0)  # Minimum Stromstärke I0
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            # Vorwärtspropagation
            output = self.forward(X)

            # Berechne Fehler
            error = y - output

            # Rückwärtspropagation
            dZ2 = error * sigmoid_derivative(self.Z2)
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0)

            dA1 = np.dot(dZ2, self.W2.T)
            dA1 = np.maximum(dA1, self.I0)  # Minimum Stromstärke I0
            dZ1 = dA1 * sigmoid_derivative(self.Z1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0)

            # Aktualisiere Gewichte und Biases
            self.W1 += learning_rate * dW1
            self.b1 += learning_rate * db1
            self.W2 += learning_rate * dW2
            self.b2 += learning_rate * db2

# Definiere das 2x2-Schachbrettmuster
X = np.array([
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 0]
])

y = np.array([
    [1],
    [0],
    [1],
    [0]
])

# Initialisiere und trainiere das neuronale Netzwerk
I0 = 0.01
nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=1, I0=I0)
nn.train(X, y)

# Teste das Netzwerk
for pattern in X:
    print(f"Input: {pattern} -> Output: {nn.forward(pattern)}")
