import numpy as np
import matplotlib.pyplot as plt


class ActivationFunction:
    
    def __init__(self, name):
        self.name = name
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
    def visualize(self, x_range=(-5, 5), num_points=100):
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = self.forward(x)
        dy_dx = self.backward(x)
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title(f'{self.name} Function')
        plt.xlabel('x')
        plt.ylabel(f'{self.name}(x)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(x, dy_dx, 'r-', linewidth=2)
        plt.title(f'{self.name} Derivative')
        plt.xlabel('x')
        plt.ylabel(f"d{self.name}/dx")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class ReLU(ActivationFunction):
    
    def __init__(self):
        super().__init__("ReLU")
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    
    def __init__(self):
        super().__init__("Sigmoid")
    
    def forward(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    
    def __init__(self):
        super().__init__("Tanh")
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x) ** 2


class LeakyReLU(ActivationFunction):
    
    def __init__(self, alpha=0.01):
        super().__init__("LeakyReLU")
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)


class Softmax(ActivationFunction):
    
    def __init__(self):
        super().__init__("Softmax")
    
    def forward(self, x):
        if x.ndim == 1:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        else:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x):
        s = self.forward(x)
        if x.ndim == 1:
            return s * (1 - s)
        else:
            jacobian = np.zeros((*s.shape, s.shape[1]))
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    for k in range(s.shape[1]):
                        if j == k:
                            jacobian[i, j, k] = s[i, j] * (1 - s[i, k])
                        else:
                            jacobian[i, j, k] = -s[i, j] * s[i, k]
            return jacobian


class Linear(ActivationFunction):
    
    def __init__(self):
        super().__init__("Linear")
    
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)


def get_activation(name):
    activations = {
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'leaky_relu': LeakyReLU(),
        'softmax': Softmax(),
        'linear': Linear()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name.lower()]


def compare_activations():
    activations = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'LeakyReLU': LeakyReLU(alpha=0.01)
    }
    
    x = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (name, activation) in enumerate(activations.items()):
        y = activation.forward(x)
        dy = activation.backward(x)
        
        axes[i].plot(x, y, 'b-', linewidth=2, label=f'{name} Forward')
        axes[i].plot(x, dy, 'r--', linewidth=2, label=f'{name} Derivative')
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{name} Activation Function')
        axes[i].set_xlabel('Input')
        axes[i].set_ylabel('Output')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Activation Functions Demo")
    
    relu = ReLU()
    sigmoid = Sigmoid()
    tanh = Tanh()
    
    x = np.array([-2, -1, 0, 1, 2])
    
    print(f"Input: {x}")
    print(f"ReLU: {relu.forward(x)}")
    print(f"Sigmoid: {sigmoid.forward(x)}")
    print(f"Tanh: {tanh.forward(x)}")
    
    compare_activations()
