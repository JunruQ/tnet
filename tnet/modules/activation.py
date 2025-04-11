import numpy as np

class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """ReLU 前向传播"""
        self.Z = x  # 缓存输入，用于反向传播
        return np.maximum(0, x)
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """ReLU 反向传播"""
        return dA * (self.Z > 0).astype(float)

class Softmax:
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Softmax 前向传播"""
        self.Z = x  # 缓存输入
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定性
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Softmax 反向传播（占位符，因交叉熵损失已处理）"""
        # 在交叉熵损失下，dZ = y_pred - y_true，无需显式计算
        return dA
