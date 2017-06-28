import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        # Define hyper parameters of the network
        self.inputLayerSize = 5
        self.outputLayerSize = 1
        self.hiddenLayerSize = 8

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    @staticmethod
    def sigmoid(z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def forward(self, X):
        # Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        res = self.sigmoid(self.z3)
        return res

    def cost_function(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.res = self.forward(X)
        J = 0.5 * sum((y - self.res) ** 2)
        return J

    def cost_function_prime(self, X, y):
        # Compute derivative with respect to W1 and W2 for a given X and y:
        self.res = self.forward(X)

        delta3 = np.multiply(-(y - self.res), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper Functions for interacting with other classes:
    def get_params(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        # Set W1 and W2 using single paramater vector.
        w1_start = 0
        w1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[w1_start:w1_end], (self.inputLayerSize, self.hiddenLayerSize))
        w2_end = w1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[w1_end:w2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
