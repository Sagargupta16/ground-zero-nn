__version__ = "0.1.0"

from .core import Neuron, Network
from .activations import (
    ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Linear,
    get_activation, compare_activations
)
from .optimizers import (
    SGD, SGDMomentum, RMSprop, Adam,
    get_optimizer, compare_optimizers_visualization
)
from .losses import (
    MeanSquaredError, MeanAbsoluteError, 
    BinaryCrossEntropy, CategoricalCrossEntropy, HuberLoss,
    get_loss_function, compare_loss_functions
)
from .initializers import (
    ZeroInitializer, RandomNormalInitializer, RandomUniformInitializer,
    XavierUniformInitializer, XavierNormalInitializer, 
    HeUniformInitializer, HeNormalInitializer,
    LeCunUniformInitializer, LeCunNormalInitializer,
    get_initializer, compare_initializers
)
from .layers import DenseLayer, InputLayer
from .utils import (
    generate_data, train_test_split, normalize_data,
    plot_neuron_state, plot_training_progress, plot_decision_boundary_2d,
    animate_training_process, plot_multiple_neurons_comparison
)

__all__ = [
    # Core
    'Neuron', 'Network',
    
    # Activations
    'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'Softmax', 'Linear',
    'get_activation', 'compare_activations',
    
    # Optimizers
    'SGD', 'SGDMomentum', 'RMSprop', 'Adam',
    'get_optimizer', 'compare_optimizers_visualization',
    
    # Loss functions
    'MeanSquaredError', 'MeanAbsoluteError', 
    'BinaryCrossEntropy', 'CategoricalCrossEntropy', 'HuberLoss',
    'get_loss_function', 'compare_loss_functions',
    
    # Initializers
    'ZeroInitializer', 'RandomNormalInitializer', 'RandomUniformInitializer',
    'XavierUniformInitializer', 'XavierNormalInitializer', 
    'HeUniformInitializer', 'HeNormalInitializer',
    'LeCunUniformInitializer', 'LeCunNormalInitializer',
    'get_initializer', 'compare_initializers',
    
    # Layers
    'DenseLayer', 'InputLayer',
    
    # Utilities
    'generate_data', 'train_test_split', 'normalize_data',
    'plot_neuron_state', 'plot_training_progress', 'plot_decision_boundary_2d',
    'animate_training_process', 'plot_multiple_neurons_comparison'
]
