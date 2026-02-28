# GroundZeroNN

A neural network library built from scratch in Python -- no TensorFlow, no PyTorch -- with working backpropagation and 79.8% MNIST accuracy.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![MNIST](https://img.shields.io/badge/MNIST-79.8%25_Accuracy-green?style=flat)

## Overview

GroundZeroNN is an educational neural network library built entirely from scratch using only NumPy for computation. It implements forward propagation, backpropagation with gradient computation, multiple activation functions, loss functions, and optimizers -- all without any external ML frameworks. Achieves 79.8% test accuracy on MNIST in under 3 minutes.

## Results

| Example | Accuracy | Training Time | Description |
|---------|----------|---------------|-------------|
| `fast_parallel_mnist.py` | **79.8%** | 2.7 min | Best: Parallel batch processing |
| `optimized_mnist.py` | 73% | ~5 min | Baseline: Proven working |
| `ultra_fast_mnist.py` | 23% | 12 sec | Speed test: Quick validation |

### Training Progress (Best Model)

```
Epoch  1: 21.7% -> 27.6% accuracy
Epoch 10: 67.1% -> 72.0% accuracy
Epoch 25: 79.5% -> 79.8% accuracy
Loss:     2.30  -> 1.87  (20 epochs)
```

## Features

- **Forward Propagation** -- Data flows correctly through layers
- **Backpropagation** -- Gradients computed and propagated backwards
- **Weight Updates** -- Gradient descent with configurable learning rate
- **Softmax + Cross-Entropy** -- Multi-class classification support
- **Multiple Activations** -- ReLU, Sigmoid, Softmax
- **Weight Initialization** -- He Normal, Xavier Normal
- **No External ML Frameworks** -- Pure NumPy implementation

## Architecture

```
Input (784) -> Dense(128, ReLU) -> Dense(64, Sigmoid) -> Dense(10, Softmax) -> Output
```

## Project Structure

```
GroundZeroNN/
├── groundzero_nn/                 # Core library
│   ├── core/
│   │   ├── neuron.py             # Individual neuron with backprop
│   │   └── network.py           # Complete network class
│   ├── layers/                   # Dense layer implementation
│   ├── activations/              # ReLU, Sigmoid, Softmax
│   ├── losses/                   # Cross-Entropy, MSE
│   ├── optimizers/               # SGD, Gradient Descent
│   ├── initializers/             # He Normal, Xavier Normal
│   └── utils/                    # Visualization utilities
├── examples/
│   ├── fast_parallel_mnist.py    # Best: 79.8% in 2.7 min
│   ├── optimized_mnist.py        # Baseline: 73% accuracy
│   └── ultra_fast_mnist.py       # Quick test: 12 seconds
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/Sagargupta16/GroundZeroNN.git
cd GroundZeroNN

# Create and activate virtual environment
python -m venv .venv
.venv/Scripts/activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Run the best example
python examples/fast_parallel_mnist.py
```

## Usage

```python
from groundzero_nn.layers import DenseLayer
from groundzero_nn.losses import get_loss_function

# Create layers
layer1 = DenseLayer(784, 128, activation='relu', initializer='he_normal')
layer2 = DenseLayer(128, 64, activation='sigmoid', initializer='xavier_normal')
layer3 = DenseLayer(64, 10, activation='softmax', initializer='xavier_normal')

# Forward pass
h1 = layer1.forward(x)
h2 = layer2.forward(h1)
output = layer3.forward(h2)

# Backward pass
grad = layer3.backward(loss_gradient)
grad = layer2.backward(grad)
grad = layer1.backward(grad)

# Update weights
for layer in [layer1, layer2, layer3]:
    layer.update_weights(learning_rate=0.01)
```

## Dependencies

- NumPy -- Core computations
- Matplotlib -- Training visualizations
- scikit-learn -- MNIST dataset loading
- tqdm -- Progress bars

## License

MIT
