import numpy as np
from typing import List, Optional, Union
from ..core.neuron import Neuron
from ..activations import get_activation
from ..initializers import get_initializer


class DenseLayer:
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'relu', initializer: str = 'xavier_uniform',
                 learning_rate: float = 0.01, name: Optional[str] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.initializer_name = initializer
        self.learning_rate = learning_rate
        self.name = name or f"Dense_{output_size}"
        
        self.neurons = []
        for i in range(output_size):
            neuron = Neuron(
                num_inputs=input_size,
                activation=activation,
                initializer=initializer,
                name=f"{self.name}_neuron_{i}"
            )
            self.neurons.append(neuron)
        
        self.last_input = None
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        self.last_input = x.copy()
        
        outputs = np.zeros((batch_size, self.output_size))
        
        for i, neuron in enumerate(self.neurons):
            neuron_outputs = []
            for j in range(batch_size):
                output = neuron.forward(x[j])
                neuron_outputs.append(output)
            outputs[:, i] = neuron_outputs
        
        self.last_output = outputs.copy()
        return outputs
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.last_input is None:
            raise ValueError("Must call forward() before backward()")
        
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
        
        batch_size = grad_output.shape[0]
        grad_input = np.zeros_like(self.last_input)
        
        # Backward pass for each neuron
        for i, neuron in enumerate(self.neurons):
            for j in range(batch_size):
                # Use the new gradient-based backward method
                neuron_grad_input = neuron.backward_gradient(grad_output[j, i])
                grad_input[j] += neuron_grad_input
        
        return grad_input
    
    def update_weights(self, learning_rate=0.01):
        """Update weights using the provided learning rate and gradients."""
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)
        for neuron in self.neurons:
            if hasattr(neuron, 'update_weights'):
                neuron.update_weights(learning_rate)
    
    def get_biases(self) -> np.ndarray:
        return np.array([neuron.bias for neuron in self.neurons])
    
    def set_biases(self, biases: np.ndarray):
        if len(biases) != len(self.neurons):
            raise ValueError(f"Expected {len(self.neurons)} biases, got {len(biases)}")
        
        for neuron, bias in zip(self.neurons, biases):
            neuron.bias = bias
    
    def get_parameters(self):
        weights = [neuron.weights for neuron in self.neurons]
        biases = [neuron.bias for neuron in self.neurons]
        return {'weights': weights, 'biases': biases}
    
    def set_parameters(self, params):
        weights = params['weights']
        biases = params['biases']
        
        for i, neuron in enumerate(self.neurons):
            neuron.weights = weights[i]
            neuron.bias = biases[i]
    
    def __str__(self):
        return f"DenseLayer(input_size={self.input_size}, output_size={self.output_size}, activation={self.activation_name})"
    
    def __repr__(self):
        return self.__str__()


class InputLayer:
    def __init__(self, input_size: int, name: Optional[str] = None):
        self.input_size = input_size
        self.output_size = input_size
        self.name = name or "Input"
        self.last_input = None
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.last_input = x.copy()
        self.last_output = x.copy()
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output
    
    def get_parameters(self):
        return {}
    
    def set_parameters(self, params):
        pass
    
    def __str__(self):
        return f"InputLayer(input_size={self.input_size})"
    
    def __repr__(self):
        return self.__str__()


class OutputLayer:
    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'linear', initializer: str = 'xavier',
                 learning_rate: float = 0.01, name: Optional[str] = None):
        
        self.dense_layer = DenseLayer(
            input_size=input_size,
            output_size=output_size,
            activation=activation,
            initializer=initializer,
            learning_rate=learning_rate,
            name=name or "Output"
        )
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.name = self.dense_layer.name
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.dense_layer.forward(x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.dense_layer.backward(grad_output)
    
    def get_parameters(self):
        return self.dense_layer.get_parameters()
    
    def set_parameters(self, params):
        self.dense_layer.set_parameters(params)
    
    def __str__(self):
        return f"OutputLayer(input_size={self.input_size}, output_size={self.output_size}, activation={self.activation_name})"
    
    def __repr__(self):
        return self.__str__()


def create_layer(layer_type: str, **kwargs):
    layer_types = {
        'dense': DenseLayer,
        'input': InputLayer,
        'output': OutputLayer
    }
    
    if layer_type.lower() not in layer_types:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return layer_types[layer_type.lower()](**kwargs)


if __name__ == "__main__":
    print("🔗 Neural Network Layers Demonstration")
    print("=" * 50)
    
    np.random.seed(42)
    batch_size = 32
    input_size = 4
    hidden_size = 8
    output_size = 3
    
    print(f"\n📊 Creating layers:")
    print(f"   Input: {input_size} features")
    print(f"   Hidden: {hidden_size} neurons")
    print(f"   Output: {output_size} neurons")
    
    input_layer = InputLayer(input_size, name="Input")
    hidden_layer = DenseLayer(input_size, hidden_size, activation='relu', name="Hidden")
    output_layer = OutputLayer(hidden_size, output_size, activation='softmax', name="Output")
    
    print(f"\n🔗 Layer Architecture:")
    print(f"   {input_layer}")
    print(f"   {hidden_layer}")
    print(f"   {output_layer}")
    
    X = np.random.randn(batch_size, input_size)
    print(f"\n📥 Forward pass with batch size {batch_size}:")
    
    x1 = input_layer.forward(X)
    print(f"   Input output shape: {x1.shape}")
    
    x2 = hidden_layer.forward(x1)
    print(f"   Hidden output shape: {x2.shape}")
    print(f"   Hidden output range: [{x2.min():.3f}, {x2.max():.3f}]")
    
    x3 = output_layer.forward(x2)
    print(f"   Output shape: {x3.shape}")
    print(f"   Output range: [{x3.min():.3f}, {x3.max():.3f}]")
    
    grad_output = np.ones_like(x3)
    print(f"\n📤 Backward pass:")
    
    grad2 = output_layer.backward(grad_output)
    print(f"   Output layer gradient shape: {grad2.shape}")
    
    grad1 = hidden_layer.backward(grad2)
    print(f"   Hidden layer gradient shape: {grad1.shape}")
    
    grad0 = input_layer.backward(grad1)
    print(f"   Input layer gradient shape: {grad0.shape}")
    
    print(f"\n🔍 Layer parameters:")
    hidden_params = hidden_layer.get_parameters()
    output_params = output_layer.get_parameters()
    
    print(f"   Hidden layer weights: {len(hidden_params['weights'])} neurons")
    print(f"   Hidden layer biases: {len(hidden_params['biases'])} values")
    print(f"   Output layer weights: {len(output_params['weights'])} neurons")
    print(f"   Output layer biases: {len(output_params['biases'])} values")
    
    print("\n✅ Layers demonstration complete!")
