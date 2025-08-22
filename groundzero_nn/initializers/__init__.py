import numpy as np
import matplotlib.pyplot as plt


class WeightInitializer:
    def __init__(self, name):
        self.name = name
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        raise NotImplementedError
    
    def get_info(self):
        return {'name': self.name}


class ZeroInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("Zero")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.zeros(shape)


class RandomNormalInitializer(WeightInitializer):
    def __init__(self, mean=0.0, std=0.01):
        super().__init__(f"Random Normal (μ={mean}, σ={std})")
        self.mean = mean
        self.std = std
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.random.normal(self.mean, self.std, shape)


class RandomUniformInitializer(WeightInitializer):
    def __init__(self, low=-0.05, high=0.05):
        super().__init__(f"Random Uniform ({low}, {high})")
        self.low = low
        self.high = high
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        return np.random.uniform(self.low, self.high, shape)


class XavierUniformInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("Xavier Uniform")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        if fan_out is None:
            fan_out = shape[1] if len(shape) > 1 else shape[0]
        
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)


class XavierNormalInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("Xavier Normal")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        if fan_out is None:
            fan_out = shape[1] if len(shape) > 1 else shape[0]
        
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)


class HeUniformInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("He Uniform")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)


class HeNormalInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("He Normal")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)


class LeCunUniformInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("LeCun Uniform")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        
        limit = np.sqrt(3.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)


class LeCunNormalInitializer(WeightInitializer):
    def __init__(self):
        super().__init__("LeCun Normal")
    
    def initialize(self, shape, fan_in=None, fan_out=None):
        if fan_in is None:
            fan_in = shape[0] if len(shape) > 1 else shape[0]
        
        std = np.sqrt(1.0 / fan_in)
        return np.random.normal(0, std, shape)


def get_initializer(name, **kwargs):
    initializers = {
        'zero': ZeroInitializer,
        'zeros': ZeroInitializer,
        'random_normal': RandomNormalInitializer,
        'normal': RandomNormalInitializer,
        'random_uniform': RandomUniformInitializer,
        'uniform': RandomUniformInitializer,
        'xavier_uniform': XavierUniformInitializer,
        'glorot_uniform': XavierUniformInitializer,
        'xavier_normal': XavierNormalInitializer,
        'glorot_normal': XavierNormalInitializer,
        'he_uniform': HeUniformInitializer,
        'he_normal': HeNormalInitializer,
        'lecun_uniform': LeCunUniformInitializer,
        'lecun_normal': LeCunNormalInitializer
    }
    
    if name.lower() not in initializers:
        raise ValueError(f"Unknown initializer: {name}")
    
    return initializers[name.lower()](**kwargs)


def compare_initializers():
    np.random.seed(42)
    
    shape = (1000, 500)
    fan_in, fan_out = shape
    
    initializers = [
        ZeroInitializer(),
        RandomNormalInitializer(std=0.01),
        RandomNormalInitializer(std=1.0),
        XavierUniformInitializer(),
        XavierNormalInitializer(),
        HeUniformInitializer(),
        HeNormalInitializer()
    ]
    
    plt.figure(figsize=(20, 12))
    
    for i, init in enumerate(initializers):
        weights = init.initialize(shape, fan_in, fan_out)
        
        plt.subplot(3, len(initializers), i + 1)
        plt.hist(weights.flatten(), bins=50, alpha=0.7, density=True)
        plt.title(f'{init.name}\nMean: {weights.mean():.4f}\nStd: {weights.std():.4f}')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, len(initializers), i + 1 + len(initializers))
        std_per_neuron = np.std(weights, axis=0)
        plt.hist(std_per_neuron, bins=30, alpha=0.7)
        plt.title(f'Output Variance Distribution\nMean Std: {std_per_neuron.mean():.4f}')
        plt.xlabel('Standard Deviation per Neuron')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        forward_pass = np.tanh(np.dot(np.random.randn(100, fan_in), weights))
        
        plt.subplot(3, len(initializers), i + 1 + 2*len(initializers))
        plt.hist(forward_pass.flatten(), bins=50, alpha=0.7, density=True)
        plt.title(f'Forward Pass Output\nMean: {forward_pass.mean():.4f}\nStd: {forward_pass.std():.4f}')
        plt.xlabel('Activation Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def gradient_flow_comparison():
    np.random.seed(42)
    
    layer_sizes = [784, 512, 256, 128, 64, 10]
    num_layers = len(layer_sizes) - 1
    
    initializers = [
        ('Xavier Normal', XavierNormalInitializer()),
        ('He Normal', HeNormalInitializer()),
        ('Random Normal (0.01)', RandomNormalInitializer(std=0.01)),
        ('Random Normal (1.0)', RandomNormalInitializer(std=1.0))
    ]
    
    plt.figure(figsize=(15, 10))
    
    for idx, (name, init) in enumerate(initializers):
        weights = []
        for i in range(num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = init.initialize((fan_in, fan_out), fan_in, fan_out)
            weights.append(w)
        
        x = np.random.randn(100, layer_sizes[0])
        
        activations = [x]
        for w in weights:
            x = np.tanh(np.dot(x, w))
            activations.append(x.copy())
        
        gradients = [np.ones_like(activations[-1])]
        for i in range(len(weights) - 1, -1, -1):
            grad = gradients[-1]
            grad = grad * (1 - activations[i + 1]**2)
            grad = np.dot(grad, weights[i].T)
            gradients.append(grad)
        
        gradients.reverse()
        
        plt.subplot(2, 2, idx + 1)
        
        layer_indices = range(len(activations))
        activation_means = [np.mean(np.abs(a)) for a in activations]
        gradient_means = [np.mean(np.abs(g)) for g in gradients]
        
        plt.semilogy(layer_indices, activation_means, 'b-o', label='Activations', linewidth=2)
        plt.semilogy(layer_indices, gradient_means, 'r-s', label='Gradients', linewidth=2)
        
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Absolute Value (log scale)')
        plt.title(f'{name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if activation_means[-1] > 1e-5:
            plt.text(0.5, 0.95, '✓ Good signal flow', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
                    verticalalignment='top', horizontalalignment='center')
        else:
            plt.text(0.5, 0.95, '✗ Vanishing gradients', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
                    verticalalignment='top', horizontalalignment='center')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🎯 Weight Initializers Demonstration")
    print("=" * 50)
    
    shape = (100, 50)
    fan_in, fan_out = shape
    
    initializers = [
        ZeroInitializer(),
        RandomNormalInitializer(),
        XavierUniformInitializer(),
        XavierNormalInitializer(),
        HeUniformInitializer(),
        HeNormalInitializer()
    ]
    
    print("\n📊 Initializer Statistics:")
    for init in initializers:
        weights = init.initialize(shape, fan_in, fan_out)
        print(f"  {init.name}:")
        print(f"    Mean: {weights.mean():.6f}")
        print(f"    Std:  {weights.std():.6f}")
        print(f"    Min:  {weights.min():.6f}")
        print(f"    Max:  {weights.max():.6f}")
    
    print("\n🎨 Generating initializer comparison...")
    compare_initializers()
    
    print("\n🌊 Generating gradient flow comparison...")
    gradient_flow_comparison()
    
    print("\n✅ Weight initializers demonstration complete!")
