import numpy as np
from typing import Tuple, Dict
from tnet.modules.activation import ReLU, Softmax

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.weights = self._initialize_weights(input_size, output_size)
        self.activation_type = activation
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _initialize_weights(self, in_size: int, out_size: int) -> Dict[str, np.ndarray]:
        he = np.sqrt(2.0 / in_size)
        return {
            'W': np.random.randn(in_size, out_size) * he,
            'b': np.zeros((1, out_size))
        }
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        W, b = self.weights['W'], self.weights['b']
        Z = X @ W + b
        A = self.activation.forward(Z)
        return A, {'Z': Z, 'X': X}
    
    def backward(self, dA: np.ndarray, cache: Dict[str, np.ndarray], reg_lambda: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        Z, X = cache['Z'], cache['X']
        batch_size = X.shape[0]
        W = self.weights['W']
        
        dZ = self.activation.backward(dA)
        dW = X.T @ dZ / batch_size + reg_lambda * W
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dX = dZ @ W.T
        
        return dX, {'dW': dW, 'db': db}

class NeuralNetwork:
    def __init__(self, input_size: int, hidden1_size: int, hidden2_size: int, output_size: int, 
                 activation: str = 'relu'):
        self.layer1 = Layer(input_size, hidden1_size, activation=activation)
        self.layer2 = Layer(hidden1_size, hidden2_size, activation=activation)
        self.layer3 = Layer(hidden2_size, output_size, activation='softmax')
        self.weights = {
            'W1': self.layer1.weights['W'], 'b1': self.layer1.weights['b'],
            'W2': self.layer2.weights['W'], 'b2': self.layer2.weights['b'],
            'W3': self.layer3.weights['W'], 'b3': self.layer3.weights['b']
        }
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
    
    def load_weights(self, weights: Dict[str, np.ndarray]):
        self.layer1.weights['W'] = weights['W1']
        self.layer1.weights['b'] = weights['b1']
        self.layer2.weights['W'] = weights['W2']
        self.layer2.weights['b'] = weights['b2']
        self.layer3.weights['W'] = weights['W3']
        self.layer3.weights['b'] = weights['b3']
        self.weights = weights
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        A1, cache1 = self.layer1.forward(X, training)
        A2, cache2 = self.layer2.forward(A1, training)
        y_pred, cache3 = self.layer3.forward(A2, training)
        cache = {
            'layer1': cache1, 'layer2': cache2, 'layer3': cache3,
            'A1': A1, 'A2': A2, 'X': X
        }
        return y_pred, cache
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                 cache: Dict[str, np.ndarray], reg_lambda: float) -> Dict[str, np.ndarray]:
        dZ3 = output - y
        dA2, grads3 = self.layer3.backward(dZ3, cache['layer3'], reg_lambda)
        dA1, grads2 = self.layer2.backward(dA2, cache['layer2'], reg_lambda)
        _, grads1 = self.layer1.backward(dA1, cache['layer1'], reg_lambda)
        
        gradients = {
            'dW1': grads1['dW'], 'db1': grads1['db'],
            'dW2': grads2['dW'], 'db2': grads2['db'],
            'dW3': grads3['dW'], 'db3': grads3['db']
        }
        return gradients
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, reg_lambda: float) -> float:
        batch_size = y_pred.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-15)) / batch_size
        W1, W2, W3 = self.weights['W1'], self.weights['W2'], self.weights['W3']
        l2_reg = reg_lambda / 2 * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        return cross_entropy + l2_reg