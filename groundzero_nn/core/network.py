import numpy as np
from typing import List, Optional, Union, Dict, Any
from ..layers import DenseLayer, InputLayer
from ..losses import get_loss_function
from ..optimizers import get_optimizer


class Network:
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or "GroundZeroNetwork"
        self.layers = []
        self.compiled = False
        
        self.loss_function = None
        self.optimizer = None
        self.metrics = []
        
        self.training_history = {
            'epoch_losses': [],
            'epoch_metrics': [],
            'validation_losses': [],
            'validation_metrics': [],
            'learning_rates': []
        }
        
        self.input_size = None
        self.output_size = None
        self.total_parameters = 0
    
    def add_layer(self, layer):
        self.layers.append(layer)
        self.compiled = False
        
        if len(self.layers) == 1:
            self.input_size = layer.input_size
        self.output_size = layer.output_size
    
    def add_dense(self, units: int, activation: str = 'relu', 
                  initializer: str = 'xavier', name: Optional[str] = None):
        if not self.layers:
            raise ValueError("Cannot add dense layer without knowing input size. "
                           "Add an input layer first or specify input_size.")
        
        input_size = self.layers[-1].output_size
        layer = DenseLayer(
            input_size=input_size,
            output_size=units,
            activation=activation,
            initializer=initializer,
            name=name
        )
        self.add_layer(layer)
    
    def add_input(self, input_size: int, name: Optional[str] = None):
        layer = InputLayer(input_size=input_size, name=name)
        self.add_layer(layer)
    
    def compile(self, loss: str = 'mse', optimizer: str = 'sgd', 
                learning_rate: float = 0.01, metrics: Optional[List[str]] = None):
        if not self.layers:
            raise ValueError("Cannot compile empty network. Add layers first.")
        
        self.loss_function = get_loss_function(loss)
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        self.metrics = metrics or []
        
        self.total_parameters = sum(
            getattr(layer, 'num_parameters', 0) if hasattr(layer, 'get_layer_info') 
            else 0 for layer in self.layers
        )
        
        self.compiled = True
        print(f"Network '{self.name}' compiled successfully!")
        print(f"Total parameters: {self.total_parameters}")
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.compiled:
            raise ValueError("Network must be compiled before forward pass.")
        
        current_output = inputs
        for layer in self.layers:
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, loss_gradient: np.ndarray):
        current_gradient = loss_gradient
        
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                current_gradient = layer.backward(current_gradient)
    
    def train_step(self, inputs: np.ndarray, targets: np.ndarray, 
                   verbose: bool = False) -> Dict[str, Any]:
        if not self.compiled:
            raise ValueError("Network must be compiled before training.")
        
        outputs = self.forward(inputs)
        
        loss = self.loss_function.forward(outputs, targets)
        
        loss_gradient = self.loss_function.backward(outputs, targets)
        
        self.backward(loss_gradient)
        
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(self.optimizer.learning_rate)
        
        step_info = {
            'inputs': inputs,
            'targets': targets,
            'outputs': outputs,
            'loss': loss,
            'loss_gradient': loss_gradient
        }
        
        if verbose:
            print(f"Epoch {len(self.training_history['epoch_losses']) + 1}, Loss: {loss:.6f}")
        
        return step_info
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100, batch_size: Optional[int] = None,
              validation_data: Optional[tuple] = None,
              verbose: int = 1, step_by_step: bool = False) -> Dict[str, List]:
        if not self.compiled:
            raise ValueError("Network must be compiled before training.")
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        n_samples = len(x_train)
        
        if verbose > 0:
            print(f"Training '{self.name}' for {epochs} epochs")
            print(f"Training samples: {n_samples}")
            if validation_data:
                print(f"Validation samples: {len(validation_data[0])}")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            if batch_size is None or batch_size >= n_samples:
                batches = [(x_train, y_train)]
            else:
                batches = []
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batches.append((x_train[i:end_idx], y_train[i:end_idx]))
            
            for batch_x, batch_y in batches:
                if batch_x.ndim == 1:
                    step_info = self.train_step(batch_x, batch_y, 
                                              verbose=(step_by_step and epoch < 3))
                    epoch_losses.append(step_info['loss'])
                else:
                    for sample_x, sample_y in zip(batch_x, batch_y):
                        step_info = self.train_step(sample_x, sample_y,
                                                  verbose=(step_by_step and epoch < 3))
                        epoch_losses.append(step_info['loss'])
            
            avg_loss = np.mean(epoch_losses)
            self.training_history['epoch_losses'].append(avg_loss)
            
            if validation_data:
                x_val, y_val = validation_data
                val_outputs = self.predict(x_val)
                val_loss = self.loss_function.forward(val_outputs, y_val)
                self.training_history['validation_losses'].append(val_loss)
            
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                val_info = ""
                if validation_data:
                    val_info = f" - val_loss: {val_loss:.6f}"
                print(f"Epoch {epoch + 1:4d}/{epochs} - loss: {avg_loss:.6f}{val_info}")
        
        if verbose > 0:
            print("Training completed!")
            print(f"Final training loss: {self.training_history['epoch_losses'][-1]:.6f}")
            if validation_data:
                print(f"Final validation loss: {self.training_history['validation_losses'][-1]:.6f}")
        
        return self.training_history
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.array(inputs)
        
        if inputs.ndim == 1:
            return self.forward(inputs)
        else:
            predictions = []
            for sample in inputs:
                pred = self.forward(sample)
                predictions.append(pred)
            return np.array(predictions)
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, 
                 problem_type: str = "regression") -> Dict[str, float]:
        predictions = self.predict(x_test)
        
        test_loss = self.loss_function.forward(predictions, y_test)
        
        if problem_type == "classification":
            predictions_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(predictions_classes == y_true_classes)
            metrics = {'accuracy': accuracy}
        else:
            mse = np.mean((y_test - predictions) ** 2)
            metrics = {'mse': mse}
        
        metrics['loss'] = test_loss
        
        return metrics
    
    def summary(self):
        print(f"{self.name} - Model Summary")
        print("=" * 70)
        print(f"{'Layer (type)':<25} {'Output Shape':<15} {'Param #':<10}")
        print("=" * 70)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_layer_info'):
                info = layer.get_layer_info()
                layer_name = f"{info['name']} ({info.get('type', 'Dense')})"
                output_shape = f"({info['output_size']},)"
                params = info.get('num_parameters', 0)
                total_params += params
                
                print(f"{layer_name:<25} {output_shape:<15} {params:<10}")
            else:
                print(f"{layer.__class__.__name__}:<25> {'unknown':<15} {'0':<10}")
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        
        if self.compiled:
            print(f"Loss function: {self.loss_function.__class__.__name__}")
            print(f"Optimizer: {self.optimizer.__class__.__name__}")
    
    def plot_training_history(self):
        import matplotlib.pyplot as plt
        
        if not self.training_history['epoch_losses']:
            print("No training history to plot. Train the network first.")
            return
        
        _, axes = plt.subplots(1, 2, figsize=(12, 4)) if self.training_history['validation_losses'] else plt.subplots(1, 1, figsize=(8, 4))
        
        if self.training_history['validation_losses']:
            ax1 = axes[0]
            epochs = range(1, len(self.training_history['epoch_losses']) + 1)
            ax1.plot(epochs, self.training_history['epoch_losses'], 'b-', label='Training Loss')
            ax1.plot(epochs, self.training_history['validation_losses'], 'r-', label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            ax2 = axes[1]
            improvement = []
            initial_loss = self.training_history['epoch_losses'][0]
            for loss in self.training_history['epoch_losses']:
                improvement.append((initial_loss - loss) / initial_loss * 100)
            ax2.plot(epochs, improvement, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Training Improvement')
            ax2.grid(True, alpha=0.3)
        else:
            if hasattr(axes, 'plot'):
                ax = axes
            else:
                ax = axes[0]
            epochs = range(1, len(self.training_history['epoch_losses']) + 1)
            ax.plot(epochs, self.training_history['epoch_losses'], 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset'):
                layer.reset()
        
        self.training_history = {
            'epoch_losses': [],
            'epoch_metrics': [],
            'validation_losses': [],
            'validation_metrics': [],
            'learning_rates': []
        }
    
    def save_weights(self, filepath: str):
        weights_data = []
        for layer in self.layers:
            if hasattr(layer, 'get_weights'):
                layer_weights = {
                    'weights': layer.get_weights(),
                    'biases': layer.get_biases()
                }
                weights_data.append(layer_weights)
        
        np.savez(filepath, weights=weights_data)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        weights_data = data['weights']
        
        for layer, layer_weights in zip(self.layers, weights_data):
            if hasattr(layer, 'set_weights'):
                layer.set_weights(layer_weights['weights'])
                layer.set_biases(layer_weights['biases'])
        
        print(f"Weights loaded from {filepath}")
    
    def __repr__(self):
        return f"Network(name='{self.name}', layers={len(self.layers)}, compiled={self.compiled})"


if __name__ == "__main__":
    network = Network(name="DemoNet")
    
    network.add_input(2)
    network.add_dense(4, activation='relu')
    network.add_dense(1, activation='sigmoid')
    
    network.compile(loss='mse', optimizer='sgd', learning_rate=0.01)
    
    network.summary()
    
    from ..utils import generate_data
    x, y = generate_data('circle', n_samples=100)
    
    history = network.train(x, y, epochs=10, verbose=1)
    
    test_input = np.array([[0.5, 0.5], [-0.5, -0.5]])
    predictions = network.predict(test_input)
    print(f"Test inputs: {test_input}")
    print(f"Predictions: {predictions}")
    
    network.plot_training_history()
