# CIFAR-10 Three-Layer Neural Network Classifier

This repository contains a from-scratch implementation of a three-layer fully-connected neural network classifier (TNet) for image classification on the CIFAR-10 dataset, built using NumPy without automatic differentiation frameworks like PyTorch or TensorFlow. The project fulfills the assignment requirements for constructing a neural network with manual backpropagation, SGD optimization, hyperparameter tuning, and visualization.

## Features
- **Model**: Three-layer neural network with customizable hidden layer sizes, ReLU activations (hidden layers), and Softmax output.
- **Training**: Stochastic Gradient Descent (SGD) with learning rate decay, cross-entropy loss, L2 regularization, and automatic model saving based on validation accuracy.
- **Hyperparameter Tuning**: Searches over learning rate, hidden layer sizes, and regularization strength, evaluating 27 combinations.
- **Testing**: Loads pre-trained weights and computes test accuracy on CIFAR-10.
- **Visualization**: Generates training/validation loss and accuracy curves, and visualizes first-layer weights as 32x32x3 images.
- **Data Augmentation**: Includes random horizontal flips, cropping, brightness, and contrast adjustments to improve generalization.


## Prerequisites
- Python 3.8+
- NumPy
- Matplotlib
- CIFAR-10 dataset (download from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html))

## Usage

```bash
python main.py
```

All features are written in the `main.py` file. The code is organized into functions for clarity and modularity. The main function orchestrates the training, validation, and testing processes.