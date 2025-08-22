import numpy as np
import matplotlib.pyplot as plt
from ..activations import get_activation
from ..initializers import get_initializer
from ..utils.visualization import plot_neuron_state, plot_training_progress


class Neuron:
    
    def __init__(self, num_inputs, activation='sigmoid', initializer='normal', 
                 name=None, verbose=False):
        self.num_inputs = num_inputs
        self.activation_name = activation
        self.initializer_name = initializer
        self.name = name or f"Neuron_{id(self)}"
        self.verbose = verbose
        
        self.activation_fn = get_activation(activation)
        
        initializer_fn = get_initializer(initializer)
        self.weights = initializer_fn.initialize((num_inputs,))
        self.bias = 0.0
        
        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias
        
        self.training_history = {
            'losses': [],
            'outputs': [],
            'targets': [],
            'weights': [],
            'bias': []
        }
        
        self.last_inputs = None
        self.last_linear_output = None
        self.last_output = None
        self.last_gradients = None
    
    def forward(self, inputs):
        inputs = np.array(inputs, dtype=np.float64)
        self.last_inputs = inputs
        
        linear_output = np.dot(inputs, self.weights) + self.bias
        self.last_linear_output = linear_output
        
        output = self.activation_fn.forward(linear_output)
        self.last_output = output
        
        return output
    
    def backward(self, target):
        if self.last_output is None or self.last_inputs is None:
            raise ValueError("Must run forward pass before backward pass")
        
        error = target - self.last_output
        
        activation_derivative = self.activation_fn.backward(self.last_linear_output)
        
        delta = error * activation_derivative
        
        weight_gradients = delta * self.last_inputs
        bias_gradient = delta
        
        self.last_gradients = {
            'weight_gradients': weight_gradients,
            'bias_gradient': bias_gradient,
            'delta': delta,
            'error': error
        }
        
        return weight_gradients
    
    def backward_gradient(self, grad_output):
        """Backward pass with gradient from next layer."""
        if self.last_output is None or self.last_inputs is None:
            raise ValueError("Must run forward pass before backward pass")
        
        # Calculate activation derivative
        activation_derivative = self.activation_fn.backward(self.last_linear_output)
        
        # Calculate delta (gradient w.r.t. pre-activation)
        delta = grad_output * activation_derivative
        
        # Calculate gradients w.r.t. weights and bias
        weight_gradients = delta * self.last_inputs
        bias_gradient = delta
        
        # Calculate gradient w.r.t. input (for previous layer)
        input_gradient = delta * self.weights
        
        self.last_gradients = {
            'weight_gradients': weight_gradients,
            'bias_gradient': bias_gradient,
            'delta': delta,
            'input_gradient': input_gradient
        }
        
        return input_gradient
    
    def update_weights(self, learning_rate=0.01):
        if self.last_gradients is None:
            raise ValueError("Must run backward pass before updating weights")
        
        # Gradient descent: move in the OPPOSITE direction of the gradient
        self.weights -= learning_rate * self.last_gradients['weight_gradients']
        self.bias -= learning_rate * self.last_gradients['bias_gradient']
    
    def predict(self, inputs):
        return self.forward(inputs)
    
    def train_step(self, inputs, target, learning_rate=0.01, visualize=False):
        old_weights = self.weights.copy()
        old_bias = self.bias
        
        output = self.forward(inputs)
        
        loss = 0.5 * (target - output) ** 2
        
        self.backward(target)
        
        self.update_weights(learning_rate)
        
        weight_changes = self.weights - old_weights
        bias_change = self.bias - old_bias
        
        self.training_history['losses'].append(loss)
        self.training_history['outputs'].append(output)
        self.training_history['targets'].append(target)
        self.training_history['weights'].append(self.weights.copy())
        self.training_history['bias'].append(self.bias)
        
        step_info = {
            'inputs': inputs,
            'target': target,
            'output': output,
            'loss': loss,
            'old_weights': old_weights,
            'new_weights': self.weights.copy(),
            'weight_changes': weight_changes,
            'old_bias': old_bias,
            'new_bias': self.bias,
            'bias_change': bias_change,
            'linear_output': self.last_linear_output,
            'weight_gradients': self.last_gradients['weight_gradients'],
            'bias_gradient': self.last_gradients['bias_gradient']
        }
        
        if visualize:
            plot_neuron_state(self, step_info)
        
        return step_info
    
    def train(self, X, y, epochs=100, learning_rate=0.01, 
              visualize_steps=None, plot_progress=True):
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])
        
        initial_loss = None
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (inputs, target) in enumerate(zip(X, y)):
                step_info = self.train_step(inputs, target, learning_rate)
                epoch_loss += step_info['loss']
                
                if visualize_steps and epoch < visualize_steps:
                    print(f"Epoch {epoch+1}, Step {i+1}: Loss = {step_info['loss']:.6f}")
            
            avg_loss = epoch_loss / len(X)
            epoch_losses.append(avg_loss)
            
            if epoch == 0:
                initial_loss = avg_loss
        
        results = {
            'initial_loss': initial_loss,
            'final_loss': epoch_losses[-1],
            'epoch_losses': epoch_losses,
            'improvement': initial_loss - epoch_losses[-1] if initial_loss else 0
        }
        
        if plot_progress:
            plot_training_progress(self, results)
        
        return results
    
    def reset(self):
        initializer_fn = get_initializer(self.initializer_name)
        self.weights = initializer_fn.initialize((self.num_inputs,))
        self.bias = 0.0
        
        self.initial_weights = self.weights.copy()
        self.initial_bias = self.bias
        
        self.training_history = {
            'losses': [],
            'outputs': [],
            'targets': [],
            'weights': [],
            'bias': []
        }
        
        self.last_inputs = None
        self.last_linear_output = None
        self.last_output = None
        self.last_gradients = None
    
    def get_info(self):
        return {
            'name': self.name,
            'num_inputs': self.num_inputs,
            'activation': self.activation_name,
            'initializer': self.initializer_name,
            'current_weights': self.weights,
            'current_bias': self.bias,
            'training_steps': len(self.training_history['losses'])
        }
    
    def __repr__(self):
        return f"{self.name}(inputs={self.num_inputs}, activation={self.activation_name})"


if __name__ == "__main__":
    neuron = Neuron(num_inputs=2, activation='sigmoid', name="Demo_Neuron")
    
    X = np.array([[1, 2], [2, 3], [3, 1]])
    y = np.array([1, 0, 1])
    
    results = neuron.train(X, y, epochs=10)
    
    test_input = np.array([1.5, 2.5])
    prediction = neuron.predict(test_input)
    print(f"Prediction for {test_input}: {prediction:.4f}")
