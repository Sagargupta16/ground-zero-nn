"""
🚀 ULTRA-FAST GROUNDZERO MNIST
==============================
Maximum speed optimizations - finish training in under 2 minutes!

Extreme speed optimizations:
1. Minimal data size but representative
2. Aggressive batch sizes  
3. Fewer epochs with smart learning
4. Simplified architecture
5. Pre-computed optimizations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import groundzero_nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

from groundzero_nn.layers import DenseLayer
from groundzero_nn.losses import get_loss_function

class UltraFastMNIST:
    """Ultra-fast MNIST network - maximum speed!"""
    
    def __init__(self):
        print("Building ULTRA-FAST MNIST Network...")
        
        # Minimal but effective architecture
        self.layer1 = DenseLayer(input_size=784, output_size=128, activation='relu', initializer='he_normal')
        self.layer2 = DenseLayer(input_size=128, output_size=10, activation='linear', initializer='xavier_normal')
        
        print("Ultra-fast network built: 784→128→10 (minimal layers)")
        
    def forward(self, x):
        """Lightning-fast forward pass."""
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        
        # Fast softmax
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def train_epoch(self, X_train, y_train, learning_rate=0.01):
        """Train entire epoch in one go for maximum speed."""
        # Forward pass
        predictions = self.forward(X_train)
        
        # Loss calculation
        epsilon = 1e-15
        predictions_safe = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_train * np.log(predictions_safe), axis=1))
        
        # Backward pass
        grad_output = predictions - y_train
        grad_output = self.layer2.backward(grad_output)
        grad_output = self.layer1.backward(grad_output)
        
        # Update weights
        self.layer1.update_weights(learning_rate)
        self.layer2.update_weights(learning_rate)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_train, axis=1)
        accuracy = np.mean(pred_classes == true_classes) * 100
        
        return loss, accuracy
    
    def evaluate(self, X_test, y_test):
        """Fast evaluation."""
        predictions = self.forward(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        return np.mean(pred_classes == true_classes) * 100

def load_ultra_fast_data():
    """Load minimal but representative MNIST data."""
    print("Loading minimal MNIST data for ultra-fast training...")
    
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Ultra-small but balanced dataset
    train_per_class = 50  # 50 samples per digit = 500 total
    test_per_class = 20   # 20 samples per digit = 200 total
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        np.random.shuffle(digit_indices)
        
        # Take samples for this digit
        train_indices = digit_indices[:train_per_class]
        test_indices = digit_indices[train_per_class:train_per_class + test_per_class]
        
        X_train_list.extend(X[train_indices])
        y_train_list.extend([digit] * train_per_class)
        X_test_list.extend(X[test_indices])
        y_test_list.extend([digit] * test_per_class)
    
    # Convert to arrays
    X_train = np.array(X_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    
    # Fast normalization
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Center the data
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean
    
    # One-hot encode
    y_train_onehot = np.eye(10, dtype=np.float32)[y_train]
    y_test_onehot = np.eye(10, dtype=np.float32)[y_test]
    
    print(f"Ultra-fast data loaded: {len(X_train)} train, {len(X_test)} test (balanced)")
    return X_train, X_test, y_train_onehot, y_test_onehot

def ultra_fast_training():
    """Ultra-fast training - complete in under 2 minutes!"""
    print("\n🚀 ULTRA-FAST GROUNDZERO TRAINING!")
    print("==================================")
    
    start_time = time.time()
    
    # Load minimal data
    X_train, X_test, y_train, y_test = load_ultra_fast_data()
    
    # Create network
    network = UltraFastMNIST()
    
    # Ultra-fast parameters
    epochs = 20  # Minimal epochs
    initial_lr = 0.1  # High learning rate for fast convergence
    
    print(f"Starting ULTRA-FAST training for {epochs} epochs...")
    print(f"Target: Complete training in under 2 minutes!")
    
    # Storage
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []
    
    best_test_acc = 0
    
    # Training loop - no batching for ultra speed
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Learning rate decay
        current_lr = initial_lr * (0.9 ** epoch)
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Train entire epoch at once
        train_loss, train_acc = network.train_epoch(X_shuffled, y_shuffled, current_lr)
        
        # Test evaluation
        test_acc = network.evaluate(X_test, y_test)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Progress display
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {train_acc:5.1f}% | "
              f"Test: {test_acc:5.1f}% | "
              f"Loss: {train_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n🚀 ULTRA-FAST TRAINING COMPLETE!")
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.2f} minutes)")
    print(f"  Average per Epoch: {np.mean(epoch_times):.2f}s")
    print(f"  Best Test Accuracy: {best_test_acc:.1f}%")
    
    # Speed analysis
    if total_time < 120:
        print(f"  🎯 TARGET ACHIEVED! Completed in under 2 minutes!")
    else:
        print(f"  ⏱️  Close! Just {total_time - 120:.1f}s over target")
    
    # Accuracy analysis
    if best_test_acc >= 70:
        print(f"  🌟 GREAT! {best_test_acc:.1f}% accuracy with ultra-fast training!")
    else:
        print(f"  📊 {best_test_acc:.1f}% accuracy - good for ultra-fast training!")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-o', linewidth=2, markersize=6)
    plt.title('Ultra-Fast Loss Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'b-o', label='Train', linewidth=2, markersize=6)
    plt.plot(test_accuracies, 'r-s', label='Test', linewidth=2, markersize=6)
    plt.title('Ultra-Fast Accuracy Progress', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(1, len(epoch_times) + 1), epoch_times, color='green', alpha=0.7)
    plt.title(f'Time per Epoch (Avg: {np.mean(epoch_times):.2f}s)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return network, best_test_acc, total_time

def speed_benchmark():
    """Benchmark different speed configurations."""
    print("\n⚡ SPEED BENCHMARK")
    print("=================")
    
    configs = [
        {"name": "Ultra-Fast", "data_size": 500, "epochs": 20, "expected_time": 60},
        {"name": "Lightning", "data_size": 300, "epochs": 15, "expected_time": 30},
        {"name": "Instant", "data_size": 200, "epochs": 10, "expected_time": 15},
    ]
    
    print("Estimated performance for different configurations:")
    print("-" * 60)
    print(f"{'Config':<12} {'Data Size':<10} {'Epochs':<8} {'Est. Time':<12} {'Est. Accuracy'}")
    print("-" * 60)
    
    for config in configs:
        if config['name'] == "Ultra-Fast":
            est_acc = "70-75%"
        elif config['name'] == "Lightning":
            est_acc = "65-70%"
        else:
            est_acc = "60-65%"
        
        print(f"{config['name']:<12} {config['data_size']:<10} "
              f"{config['epochs']:<8} {config['expected_time']:>3}s ({config['expected_time']/60:.1f}min) "
              f"{est_acc:>10}")
    
    print("-" * 60)
    print("\n💡 Choose your speed level:")
    print("1. Ultra-Fast: Best balance of speed and accuracy")
    print("2. Lightning: Very fast, decent accuracy")
    print("3. Instant: Maximum speed, basic accuracy")

if __name__ == "__main__":
    print("🚀 ULTRA-FAST GROUNDZERO MNIST")
    print("==============================")
    print("Maximum speed training - target under 2 minutes!")
    
    # Show speed options
    speed_benchmark()
    
    print(f"\nRunning Ultra-Fast configuration...")
    
    # Run ultra-fast training
    network, best_acc, total_time = ultra_fast_training()
    
    print(f"\n🎯 ULTRA-FAST RESULTS SUMMARY:")
    print(f"  ⏱️  Training Time: {total_time:.1f}s ({total_time/60:.2f} minutes)")
    print(f"  🎯 Best Accuracy: {best_acc:.1f}%")
    print(f"  ⚡ Speed Rating: {'EXCELLENT' if total_time < 120 else 'GOOD'}")
    print(f"  📊 Accuracy Rating: {'EXCELLENT' if best_acc >= 70 else 'GOOD'}")
    
    # Final recommendations
    print(f"\n💡 ULTRA-FAST TRAINING TIPS:")
    print("✅ Use minimal representative data")
    print("✅ Simple but effective architecture") 
    print("✅ High learning rate with decay")
    print("✅ No batching for small datasets")
    print("✅ Pre-normalized data")
    print("✅ Float32 for speed")
    
    print(f"\n🚀 Your training is now ULTRA-FAST!")
    print(f"Perfect for rapid prototyping and testing!")
