from dataclasses import dataclass
import numpy as np

# Shorten names and explain with comments?
@dataclass
class hidden_neurons:
    n: int
    features: int
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, n: int, features: int):
        self.n = n
        self.features = features
        self.weights = np.random.randn(features, n)
        self.bias = np.zeros(n) + 0.01

@dataclass
class output_layer:
    n: int
    categories: int
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, n: int, categories: int):
        self.n = n
        self.categories = categories
        self.weights = np.random.randn(n, categories)
        self.bias = np.zeros(categories) + 0.01

