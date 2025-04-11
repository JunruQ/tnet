import numpy as np
from typing import Tuple, List
import pickle
import os

class DataLoader:
    def __init__(self, batch_size: int, augment: bool = True):
        self.batch_size = batch_size
        self.augment = augment  # 是否启用数据增强
    
    def _unpickle(self, file: str) -> dict:
        """读取 CIFAR-10 的 pickle 文件"""
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    def load_cifar10(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载 CIFAR-10 数据集
        参数:
            data_path: 数据文件夹路径，例如 'cifar-10-batches-py'
        返回:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        train_images = []
        train_labels = []
        for i in range(1, 6):
            batch_file = os.path.join(data_path, f'data_batch_{i}')
            batch = self._unpickle(batch_file)
            train_images.append(batch[b'data'])
            train_labels.append(batch[b'labels'])
        
        X_train = np.concatenate(train_images, axis=0)  # (50000, 3072)
        y_train = np.concatenate(train_labels, axis=0)  # (50000,)
        
        X_val = X_train[45000:]  # (5000, 3072)
        y_val = y_train[45000:]  # (5000,)
        X_train = X_train[:45000]  # (45000, 3072)
        y_train = y_train[:45000]  # (45000,)
        
        test_file = os.path.join(data_path, 'test_batch')
        test_batch = self._unpickle(test_file)
        X_test = test_batch[b'data']  # (10000, 3072)
        y_test = np.array(test_batch[b'labels'])  # (10000,)
        
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        X_test = self.preprocess(X_test)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        数据预处理：归一化到 [0, 1]
        参数:
            images: 输入图像数据，形状 (N, 3072)
        返回:
            预处理后的图像数据
        """
        images = images.astype(np.float32) / 255.0
        return images
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        数据增强：对单张图片应用随机变换
        参数:
            image: 单张图片，形状 (3072,)，表示 32x32x3 的扁平化数据
        返回:
            增强后的图片，形状 (3072,)
        """
        # 将扁平化的 (3072,) 数据转换为 (32, 32, 3)
        img = image.reshape(32, 32, 3)
        
        # 1. 随机水平翻转 (50% 概率)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        
        # 2. 随机裁剪 (padding 后随机裁剪回 32x32)
        pad_size = 4
        padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        h, w, _ = padded_img.shape  # (40, 40, 3)
        crop_x = np.random.randint(0, h - 32)
        crop_y = np.random.randint(0, w - 32)
        img = padded_img[crop_x:crop_x+32, crop_y:crop_y+32, :]
        
        # 3. 随机亮度调整 (±0.1 范围)
        brightness_factor = np.random.uniform(-0.1, 0.1)
        img = np.clip(img + brightness_factor, 0, 1)
        
        # 4. 随机对比度调整 (0.9 到 1.1 倍)
        contrast_factor = np.random.uniform(0.9, 1.1)
        img = np.clip(contrast_factor * (img - 0.5) + 0.5, 0, 1)
        
        # 将增强后的图片扁平化回 (3072,)
        return img.reshape(3072)
    
    def get_batches(self, images: np.ndarray, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成小批量数据，可选数据增强
        参数:
            images: 输入图像数据，形状 (N, 3072)
            labels: 标签数据，形状 (N,)
        返回:
            List of (batch_images, batch_labels)
        """
        num_samples = images.shape[0]
        indices = np.random.permutation(num_samples)
        batches = []
        
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            
            # 如果启用增强，对训练数据应用增强
            if self.augment:
                batch_images = np.array([self.augment(img) for img in batch_images])
            
            batches.append((batch_images, batch_labels))
        
        return batches