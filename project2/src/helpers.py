import numpy as np


#def sigmoid(x: np.ndarray) -> np.ndarray:
#    return 1 / (1 + np.exp(-x))

class sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x)
        return e/((1 + e)**2)
        #return x*(1 - x)

class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros(x.shape), x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.heaviside(x, 1)

class leakyReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(-0.01*x, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.heaviside(x, 1)*0.99 + 0.01

class tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (np.tanh(x)**2)/(np.sinh(x)**2)
#def ReLU(x: np.ndarray) -> np.ndarray:
#    return np.maximum(np.zeros(x.shape), x)

def softmax(z: np.ndarray) -> np.ndarray:
    #mx = np.amax(z)
    #adjusted_z = z - mx
    #sm =  np.sum(adjusted_z, axis=1, keepdims=True)
    #t = adjusted_z - np.log(sm)
    #print(mx)
    #return np.exp(t)

    #print(z[:,0])
    #print(np.amax(z, axis=0, keepdims=True))
    #mx = np.amax(z, axis=0)#
    #exp = np.exp(z - mx)
    exp = np.exp(z)
    #t = z - np.log(np.sum(exp, axis=1, keepdims=True))
    #return np.exp(t)
    #print(np.sum(exp, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_der(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)

def cost1(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a - t

#derivative w.r.t a
def cross_entropy(a: np.ndarray, t: np.ndarray):
    return (a - t)/(a*(1 - a))

def prediction(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


def make_one_hot(vector: np.ndarray) -> np.ndarray:
    n = len(vector)
    categories = np.max(vector) + 1
    onehot_vector = np.zeros((n, categories))
    onehot_vector[range(n), vector] = 1
    return onehot_vector


def accuracy_score(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)
