import numpy as np
from typing import List, Dict
import pickle
import matplotlib.pyplot as plt
from dataset.cifar10 import DataLoader
from itertools import product
from tnet.neural_network import NeuralNetwork

class Trainer:
    def __init__(self, model, learning_rate: float, reg_lambda: float, 
                 lr_decay: float = 0.95, save_path: str = 'best_model.pkl',
                 load_path: str = None):
        self.model = model
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        self.lr_decay = lr_decay
        self.save_path = save_path
        
        if load_path is not None:
            with open(load_path, 'rb') as f:
                weights = pickle.load(f)
                if self._validate_weights(weights):
                    self.model.load_weights(weights)
                    print(f"Loaded weights from {load_path}")
                else:
                    raise ValueError("Loaded weights do not match model architecture")
        
        self.best_val_acc = 0.0
    
    def _validate_weights(self, weights: Dict[str, np.ndarray]) -> bool:
        expected_keys = self.model.weights.keys()
        loaded_keys = weights.keys()
        
        if not all(key in expected_keys for key in loaded_keys):
            print(f"Unexpected keys in loaded weights: {set(loaded_keys) - set(expected_keys)}")
            return False
        
        for key in loaded_keys:
            if weights[key].shape != self.model.weights[key].shape:
                print(f"Shape mismatch for {key}: expected {self.model.weights[key].shape}, got {weights[key].shape}")
                return False
        
        return True
    
    def sgd_step(self, grads: Dict[str, np.ndarray]):
        for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            self.model.weights[key] -= self.lr * grads[f'd{key}']
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int, batch_size: int, 
              test_interval: int = 5, visualize: bool = False) -> Dict[str, List[float]]:
        num_classes = self.model.weights['b3'].shape[1]
        y_train_onehot = np.eye(num_classes)[y_train]
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }
        
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train_onehot[perm]
            
            total_loss = 0.0
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                X_batch = X_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                
                y_pred, cache = self.model.forward(X_batch)
                loss = self.model.compute_loss(y_pred, y_batch, self.reg_lambda)
                total_loss += loss * (batch_end - i)
                
                grads = self.model.backward(X_batch, y_batch, y_pred, cache, self.reg_lambda)
                self.sgd_step(grads)
            
            train_loss = total_loss / num_samples
            train_acc = self.evaluate(X_train, y_train)
            val_acc = self.evaluate(X_val, y_val)
            
            y_val_pred, _ = self.model.forward(X_val)
            val_loss = self.model.compute_loss(y_val_pred, np.eye(num_classes)[y_val], self.reg_lambda)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % test_interval == 0 or epoch == epochs - 1:
                test_acc = self.evaluate(X_test, y_test)
                history['test_acc'].append(test_acc)
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
            else:
                history['test_acc'].append(None)
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.model.weights, f)
                print(f"New best val_acc={val_acc:.4f}, model saved")
            
            self.lr *= self.lr_decay
        
        if visualize:
            self.plot_training_curves(history, epochs)
            self.visualize_weights()
        
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred, _ = self.model.forward(X)
        predicted_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predicted_labels == y)
        return accuracy
    
    def plot_training_curves(self, history: Dict[str, List[float]], epochs: int):
        epochs_range = range(1, epochs + 1)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
        plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        test_epochs = [i+1 for i, acc in enumerate(history['test_acc']) if acc is not None]
        test_accs = [acc for acc in history['test_acc'] if acc is not None]
        plt.plot(test_epochs, test_accs, 'o-', label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()
    
    def visualize_weights(self):
        W1 = self.model.weights['W1']
        num_filters = W1.shape[1]
        W1_images = W1.T.reshape(num_filters, 3, 32, 32)
        W1_images = (W1_images - W1_images.min()) / (W1_images.max() - W1_images.min())
        
        plt.figure(figsize=(12, 12))
        num_to_show = min(64, num_filters)
        rows = int(np.ceil(num_to_show / 8))
        
        for i in range(num_to_show):
            plt.subplot(rows, 8, i + 1)
            img = W1_images[i].transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Filter {i+1}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('weights_visualization.png')
        plt.show()

class Tester:
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
        
        input_size = weights['W1'].shape[0]
        hidden1_size = weights['W1'].shape[1]
        hidden2_size = weights['W2'].shape[1]
        output_size = weights['W3'].shape[1]
        
        self.model = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)
        self.model.load_weights(weights)
        
        print(f"Loaded model weights shapes: {[w.shape for w in weights.values()]}")
    
    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred, _ = self.model.forward(X_test)
        predicted_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predicted_labels == y_test)
        return accuracy

class HyperparameterSearch:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def search(self, param_grid: Dict[str, List], epochs: int, batch_size: int) -> Dict[str, float]:
        results = {}
        X_train, y_train, X_val, y_val, _, _ = self.data_loader.load_cifar10('cifar-10-batches-py')
        for params in self._generate_param_combinations(param_grid):
            model = NeuralNetwork(3072, params['h1_size'], params['h2_size'], 10)
            trainer = Trainer(
                model=model,
                learning_rate=params['lr'],
                reg_lambda=params['reg'],
                lr_decay=0.95,  # Fixed from paper's baseline
                save_path=f"model_{params['lr']}_{params['h1_size']}_{params['h2_size']}_{params['reg']}.pkl"
            )
            history = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_train,  # Using train as placeholder for test to match original code
                y_test=y_train,
                epochs=epochs,
                batch_size=batch_size,
                test_interval=5,
                visualize=False
            )
            val_acc = max(history['val_acc'])
            key = f"lr={params['lr']},h1_size={params['h1_size']},h2_size={params['h2_size']},reg={params['reg']}"
            results[key] = val_acc
            print(f"Params: {key}, Best val_acc: {val_acc:.4f}")
        return results
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]):
        keys = ['lr', 'h1_size', 'h2_size', 'reg']
        values = [
            param_grid['lr'],
            [h[0] for h in param_grid['hidden_sizes']],
            [h[1] for h in param_grid['hidden_sizes']],
            param_grid['reg']
        ]
        return [dict(zip(keys, v)) for v in product(*values)]

def main():
    data_loader = DataLoader(batch_size=64)
    param_grid = {
        'lr': [0.01, 0.005, 0.02],
        'hidden_sizes': [(256, 128), (512, 256), (128, 64)],
        'reg': [0.001, 0.01, 0.0001]
    }
    searcher = HyperparameterSearch(data_loader)
    results = searcher.search(param_grid, epochs=20, batch_size=64)
    # data_loader = DataLoader(batch_size=64)
    # X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_cifar10('cifar-10-batches-py')
    
    # print("Dataset Statistics:")
    # print(f"X_train - shape: {X_train.shape}, mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    # print(f"X_val   - shape: {X_val.shape}, mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
    # print(f"X_test  - shape: {X_test.shape}, mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
    # print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}, Test samples: {len(y_test)}")
    
    # model = NeuralNetwork(input_size=3072, hidden1_size=256, hidden2_size=128, output_size=10)

    # # Initialize trainer with pre-trained weights
    # trainer = Trainer(
    #     model=model,
    #     learning_rate=0.01, 
    #     reg_lambda=0.001,
    #     lr_decay=0.95,
    #     save_path='best_model.pkl',
    #     load_path='best_model_0409.pkl'  # Load the pre-trained model
    # )
    
    # # Train for 20- more epochs
    # print('Start training...')
    # history = trainer.train(
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_val=X_val,
    #     y_val=y_val,
    #     X_test=X_test,
    #     y_test=y_test,
    #     epochs=200,
    #     batch_size=64,
    #     test_interval=5,
    #     visualize=True
    # )
    
    # # Print final results
    # print("\nTraining completed!")
    # print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    # print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    # print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    # print(f"Best validation accuracy achieved: {trainer.best_val_acc:.4f}")
    
    # # Test the best saved model
    # tester = Tester('best_model.pkl')
    # final_test_acc = tester.test(X_test, y_test)
    # print(f"\nVerification - Test accuracy with best saved model: {final_test_acc:.4f}")

if __name__ == "__main__":
    main()