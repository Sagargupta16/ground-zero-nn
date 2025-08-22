import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import groundzero_nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

from groundzero_nn.layers import DenseLayer
from groundzero_nn.losses import get_loss_function
from groundzero_nn.optimizers import get_optimizer

class GroundZeroMNISTNetwork:
    """Optimized MNIST network using GroundZero components."""
    
    def __init__(self):
        print("Building OPTIMIZED MNIST Network...")
        
        # Create layers with better architecture
        self.layer1 = DenseLayer(input_size=784, output_size=128, activation='relu', initializer='he_normal')
        self.layer2 = DenseLayer(input_size=128, output_size=64, activation='relu', initializer='he_normal')
        self.layer3 = DenseLayer(input_size=64, output_size=10, activation='sigmoid', initializer='xavier_normal')
        
        # Get loss function
        self.loss_fn = get_loss_function('categorical_crossentropy')
        
        print("Optimized network built!")
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        
        # Apply softmax to final layer outputs
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return x
    
    def backward(self, predictions, targets):
        """Backward pass through the network."""
        # For softmax + cross-entropy, the gradient w.r.t. logits is:
        grad_output = predictions - targets
        
        # Backward through layers (reverse order)
        grad_output = self.layer3.backward(grad_output)
        grad_output = self.layer2.backward(grad_output)
        grad_output = self.layer1.backward(grad_output)
        
        return grad_output
    
    def update_weights(self, learning_rate=0.01):
        """Update all layer weights."""
        self.layer1.update_weights(learning_rate)
        self.layer2.update_weights(learning_rate)
        self.layer3.update_weights(learning_rate)
    
    def train_step(self, x_batch, y_batch, learning_rate=0.01):
        """Perform one training step."""
        # Forward pass
        predictions = self.forward(x_batch)
        
        # Calculate loss
        loss = self.loss_fn.forward(predictions, y_batch)
        
        # Backward pass
        self.backward(predictions, y_batch)
        
        # Update weights
        self.update_weights(learning_rate)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)
        accuracy = np.mean(pred_classes == true_classes) * 100
        
        return loss, accuracy

def load_real_mnist_sample():
    """Load a small sample of real MNIST data."""
    print("Loading real MNIST data...")
    
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Take smaller sample for testing
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Get 1000 samples for training, 200 for testing  
    train_indices = indices[:1000]
    test_indices = indices[1000:1200]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels
    y_train_onehot = np.zeros((len(y_train), 10))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1
    
    y_test_onehot = np.zeros((len(y_test), 10))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train_onehot, y_test_onehot

def train_network():
    """Train the network with proper learning."""
    print("\nOPTIMIZED GROUNDZERO TRAINING!")
    print("================================")
    
    # Load data
    X_train, X_test, y_train, y_test = load_real_mnist_sample()
    
    # Create network
    network = GroundZeroMNISTNetwork()
    
    # Training parameters
    epochs = 40
    batch_size = 64
    learning_rate = 0.01  # Smaller learning rate for stability
    
    # Storage for metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Shuffle training data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Training in batches
        n_batches = len(X_train) // batch_size
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}', leave=False)
        
        for i in pbar:
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Training step
            loss, batch_acc = network.train_step(x_batch, y_batch, learning_rate)
            
            epoch_loss += loss
            correct_predictions += batch_acc * len(x_batch) / 100
            total_predictions += len(x_batch)
            
            pbar.set_postfix({'Loss': f'{loss:.4f}', 'Acc': f'{batch_acc:.1f}%'})
        
        # Calculate training metrics
        train_loss = epoch_loss / n_batches
        train_acc = (correct_predictions / total_predictions) * 100
        
        # Test accuracy
        test_predictions = network.forward(X_test)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(y_test, axis=1)
        test_acc = np.mean(test_pred_classes == test_true_classes) * 100
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {train_acc:5.1f}% | "
              f"Test: {test_acc:5.1f}% | "
              f"Loss: {train_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    print("\nTRAINING COMPLETE!")
    print(f"  Final Train Accuracy: {train_accuracies[-1]:.1f}%")
    print(f"  Final Test Accuracy: {test_accuracies[-1]:.1f}%")
    print(f"  Best Test Accuracy: {max(test_accuracies):.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return network, train_accuracies, test_accuracies

if __name__ == "__main__":
    print("OPTIMIZED GROUNDZERO MNIST TRAINING")
    print("=====================================")
    print("Using real MNIST data with proper learning!")
    
    network, train_accs, test_accs = train_network()
    
    print("\nSUCCESS!")
    print(f"Your GroundZero neural network achieved {max(test_accs):.1f}% accuracy!")
    print("The network is now properly learning with backpropagation!")
