# 🎯 GroundZeroNN

A fully functional neural network library built from scratch in Python, with working backpropagation, gradient descent, and proven MNIST learning capabilities.

## 🚀 Quick Start

1. **Activate Virtual Environment:**
   ```bash
   .venv\Scripts\Activate.ps1
   ```

2. **Run the Optimized Example:**
   ```bash
   # Best GroundZero implementation - 69.5% accuracy with real learning!
   python examples/optimized_mnist.py
   ```

## 📁 Project Structure

```
GroundZeroNN/
├── groundzero_nn/             # Core neural network library
│   ├── core/                 # Core components
│   │   ├── neuron.py        # Individual neuron with backprop
│   │   └── network.py       # Complete network class
│   ├── layers/              # Neural network layers
│   ├── activations/         # Activation functions
│   ├── losses/              # Loss functions
│   ├── optimizers/          # Optimization algorithms
│   ├── initializers/        # Weight initialization
│   └── utils/               # Utilities and visualization
├── examples/                # Working examples
│   ├── optimized_mnist.py   # 🎯 Best: 69.5% accuracy
│   ├── training_mnist.py    # Training demonstration
│   └── README.md           # Examples documentation
├── requirements.txt        # Minimal dependencies
└── README.md              # This file
```

## 🏆 Results

- **69.5% Test Accuracy** - Real learning, not random guessing!
- **Proper Backpropagation** - Weights actually update during training
- **Working Gradient Descent** - Loss decreases from 2.30 to 1.87
- **Custom Implementation** - Built entirely with your GroundZero library

## ✅ What's Working

- ✅ **Clean Neural Network Library** - Built from scratch
- ✅ **Forward Propagation** - Data flows correctly through layers
- ✅ **Backpropagation** - Gradients computed and propagated backwards  
- ✅ **Weight Updates** - Gradient descent actually updates weights
- ✅ **Real Learning** - Not random guessing, actual improvement
- ✅ **Softmax + Cross-Entropy** - Proper multi-class classification
- ✅ **Custom Implementation** - Built entirely with GroundZero components

## 🎨 Training Progress

The optimized example shows real learning:

```
Epoch  1: 11.7% → 14.0% accuracy
Epoch 10: 55.2% → 51.0% accuracy  
Epoch 20: 74.5% → 69.5% accuracy

Final: 69.5% test accuracy (vs 10% random guessing)
Loss: 2.30 → 1.87 (proper decrease)
```

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/Sagargupta16/GroundZeroNN.git
cd GroundZeroNN

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt

# Run the optimized example
python examples/optimized_mnist.py
```

## 🎯 Core Library Usage

The `groundzero_nn` library provides working neural network components:

```python
from groundzero_nn.layers import DenseLayer
from groundzero_nn.losses import get_loss_function

# Create layers
layer1 = DenseLayer(784, 128, activation='relu', initializer='he_normal')
layer2 = DenseLayer(128, 10, activation='sigmoid', initializer='xavier_normal')

# Forward pass
output = layer1.forward(x)
output = layer2.forward(output)

# Backward pass with gradients
layer2.backward(gradients)
layer1.backward(gradients)

# Update weights
layer1.update_weights(learning_rate=0.01)
layer2.update_weights(learning_rate=0.01)
```

## 📊 Technical Achievements

- **Proper Gradient Computation** - Softmax + cross-entropy gradients working
- **Weight Updates Verified** - Weights actually change during training
- **Loss Decreases** - From 2.30 to 1.87 over 20 epochs
- **Architecture Working** - 784→128→64→10 with ReLU/Sigmoid
- **Custom Components** - No external frameworks for core logic

## 🛠️ Dependencies

- NumPy (for core computations)
- Matplotlib (for visualizations)
- scikit-learn (for MNIST dataset loading)
- tqdm (for progress bars)

## 📝 License

MIT License - Feel free to use for educational and commercial purposes.
