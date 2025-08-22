import numpy as np
import matplotlib.pyplot as plt


class LossFunction:
    def __init__(self, name):
        self.name = name
    
    def forward(self, predictions, targets):
        raise NotImplementedError
    
    def backward(self, predictions, targets):
        raise NotImplementedError
    
    def visualize(self, pred_range=(-2, 2), target_value=0, num_points=100):
        predictions = np.linspace(pred_range[0], pred_range[1], num_points)
        targets = np.full_like(predictions, target_value)
        
        losses = [self.forward(np.array([p]), np.array([target_value])) for p in predictions]
        gradients = [self.backward(np.array([p]), np.array([target_value]))[0] for p in predictions]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(predictions, losses, 'b-', linewidth=2, label=f'{self.name} Loss')
        plt.axvline(x=target_value, color='r', linestyle='--', alpha=0.7, label=f'Target = {target_value}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        plt.title(f'{self.name} Loss Function')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(predictions, gradients, 'r-', linewidth=2, label=f'{self.name} Gradient')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=target_value, color='r', linestyle='--', alpha=0.7, label=f'Target = {target_value}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Prediction')
        plt.ylabel('Gradient')
        plt.title(f'{self.name} Gradient')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__("Mean Squared Error")
    
    def forward(self, predictions, targets):
        error = predictions - targets
        return 0.5 * np.mean(error ** 2)
    
    def backward(self, predictions, targets):
        return (predictions - targets) / len(predictions)


class MeanAbsoluteError(LossFunction):
    def __init__(self):
        super().__init__("Mean Absolute Error")
    
    def forward(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))
    
    def backward(self, predictions, targets):
        return np.sign(predictions - targets) / len(predictions)


class BinaryCrossEntropy(LossFunction):
    def __init__(self, epsilon=1e-15):
        super().__init__("Binary Cross-Entropy")
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        loss = -(targets * np.log(predictions_clipped) + 
                (1 - targets) * np.log(1 - predictions_clipped))
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        gradient = -(targets / predictions_clipped - 
                    (1 - targets) / (1 - predictions_clipped))
        return gradient / len(predictions)


class CategoricalCrossEntropy(LossFunction):
    def __init__(self, epsilon=1e-15):
        super().__init__("Categorical Cross-Entropy")
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        if targets.ndim == 1 or targets.shape[-1] == 1:
            targets_onehot = np.zeros_like(predictions_clipped)
            targets_flat = targets.flatten().astype(int)
            targets_onehot[np.arange(len(targets_flat)), targets_flat] = 1
            targets = targets_onehot
        
        loss = -np.sum(targets * np.log(predictions_clipped), axis=-1)
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        if targets.ndim == 1 or targets.shape[-1] == 1:
            targets_onehot = np.zeros_like(predictions_clipped)
            targets_flat = targets.flatten().astype(int)
            targets_onehot[np.arange(len(targets_flat)), targets_flat] = 1
            targets = targets_onehot
        
        gradient = -targets / predictions_clipped
        return gradient / len(predictions)


class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        super().__init__(f"Huber Loss (δ={delta})")
        self.delta = delta
    
    def forward(self, predictions, targets):
        error = np.abs(predictions - targets)
        
        quadratic = 0.5 * error ** 2
        linear = self.delta * (error - 0.5 * self.delta)
        
        loss = np.where(error <= self.delta, quadratic, linear)
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        error = predictions - targets
        
        gradient = np.where(np.abs(error) <= self.delta, 
                          error, 
                          self.delta * np.sign(error))
        return gradient / len(predictions)


def get_loss_function(name, **kwargs):
    losses = {
        'mse': MeanSquaredError,
        'mean_squared_error': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'mean_absolute_error': MeanAbsoluteError,
        'binary_crossentropy': BinaryCrossEntropy,
        'categorical_crossentropy': CategoricalCrossEntropy,
        'huber': HuberLoss
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}")
    
    return losses[name.lower()](**kwargs)


def compare_loss_functions():
    np.random.seed(42)
    true_targets = np.array([0, 0, 0, 0, 0, 5])
    
    predictions_range = np.linspace(-2, 7, 100)
    
    loss_functions = [
        MeanSquaredError(),
        MeanAbsoluteError(), 
        HuberLoss(delta=1.0),
        HuberLoss(delta=2.0)
    ]
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, target in enumerate([0, 2]):
        plt.subplot(2, 3, i + 1)
        
        for j, loss_fn in enumerate(loss_functions):
            losses = []
            for pred in predictions_range:
                loss = loss_fn.forward(np.array([pred]), np.array([target]))
                losses.append(loss)
            
            plt.plot(predictions_range, losses, color=colors[j], 
                    linewidth=2, label=loss_fn.name)
        
        plt.axvline(x=target, color='black', linestyle='--', alpha=0.7, 
                   label=f'Target = {target}')
        plt.xlabel('Prediction')
        plt.ylabel('Loss')
        plt.title(f'Loss Functions (Target = {target})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    for i, target in enumerate([0, 2]):
        plt.subplot(2, 3, i + 4)
        
        for j, loss_fn in enumerate(loss_functions):
            gradients = []
            for pred in predictions_range:
                grad = loss_fn.backward(np.array([pred]), np.array([target]))
                gradients.append(grad[0])
            
            plt.plot(predictions_range, gradients, color=colors[j], 
                    linewidth=2, label=f'{loss_fn.name} Gradient')
        
        plt.axvline(x=target, color='black', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Prediction')
        plt.ylabel('Gradient')
        plt.title(f'Loss Gradients (Target = {target})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    
    outlier_values = np.linspace(0, 10, 50)
    
    for loss_fn in loss_functions[:3]:
        losses_with_outliers = []
        
        for outlier_val in outlier_values:
            targets = np.array([0, 0, 0, 0, outlier_val])
            predictions = np.array([0, 0, 0, 0, 0])
            
            loss = loss_fn.forward(predictions, targets)
            losses_with_outliers.append(loss)
        
        plt.plot(outlier_values, losses_with_outliers, 
                linewidth=2, label=loss_fn.name)
    
    plt.xlabel('Outlier Value')
    plt.ylabel('Average Loss')
    plt.title('Robustness to Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    
    predictions_prob = np.linspace(0.01, 0.99, 100)
    
    bce = BinaryCrossEntropy()
    
    losses_pos = [bce.forward(np.array([p]), np.array([1])) for p in predictions_prob]
    losses_neg = [bce.forward(np.array([p]), np.array([0])) for p in predictions_prob]
    
    plt.plot(predictions_prob, losses_pos, 'b-', linewidth=2, label='Target = 1')
    plt.plot(predictions_prob, losses_neg, 'r-', linewidth=2, label='Target = 0')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Binary Classification Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("📉 Loss Functions Demonstration")
    print("=" * 50)
    
    predictions = np.array([0.1, 0.8, 0.3])
    targets_regression = np.array([0.0, 1.0, 0.5])
    targets_binary = np.array([0, 1, 0])
    
    regression_losses = [MeanSquaredError(), MeanAbsoluteError(), HuberLoss()]
    
    print("\n📊 Regression Losses:")
    for loss_fn in regression_losses:
        loss_value = loss_fn.forward(predictions, targets_regression)
        gradient = loss_fn.backward(predictions, targets_regression)
        print(f"  {loss_fn.name}:")
        print(f"    Loss: {loss_value:.4f}")
        print(f"    Gradient: {gradient}")
    
    print("\n📊 Classification Losses:")
    bce = BinaryCrossEntropy()
    loss_value = bce.forward(predictions, targets_binary)
    gradient = bce.backward(predictions, targets_binary)
    print(f"  {bce.name}:")
    print(f"    Loss: {loss_value:.4f}")
    print(f"    Gradient: {gradient}")
    
    print("\n🎨 Generating loss functions comparison...")
    compare_loss_functions()
    
    print("\n✅ Loss functions demonstration complete!")
