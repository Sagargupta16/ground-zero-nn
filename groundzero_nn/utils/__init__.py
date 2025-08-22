import numpy as np
from typing import List, Tuple, Any
from .visualization import (
    plot_neuron_state, 
    plot_training_progress, 
    plot_decision_boundary_2d,
    animate_training_process,
    plot_multiple_neurons_comparison
)


TRAINING_STEPS = "Training Steps"
WEIGHT_VALUE = "Weight Value" 
TRAINING_LOSS = "Training Loss"
INPUT_1 = "Input 1"
INPUT_2 = "Input 2"

BOX_STYLE = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
BOX_STYLE_BLUE = {"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8}
BOX_STYLE_GREEN = {"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.8}


def generate_data(data_type="linear", n_samples=100, noise=0.1, random_state=42):
    np.random.seed(random_state)
    
    if data_type == "linear":
        X = np.random.randn(n_samples, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + noise * np.random.randn(n_samples)
        
    elif data_type == "quadratic":
        X = np.random.randn(n_samples, 2)
        y = X[:, 0]**2 + X[:, 1]**2 + noise * np.random.randn(n_samples)
        
    elif data_type == "circle":
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(float)
        
    elif data_type == "xor":
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
        
    elif data_type == "spiral":
        n = n_samples // 2
        t = np.linspace(0, 2*np.pi, n)
        x1 = t * np.cos(t) + noise * np.random.randn(n)
        y1 = t * np.sin(t) + noise * np.random.randn(n)
        x2 = t * np.cos(t + np.pi) + noise * np.random.randn(n)
        y2 = t * np.sin(t + np.pi) + noise * np.random.randn(n)
        
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n), np.ones(n)])
        
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize_data(X, method="standard", axis=0):
    if method == "standard":
        mean = np.mean(X, axis=axis, keepdims=True)
        std = np.std(X, axis=axis, keepdims=True)
        return (X - mean) / (std + 1e-8), mean, std
    
    elif method == "minmax":
        min_val = np.min(X, axis=axis, keepdims=True)
        max_val = np.max(X, axis=axis, keepdims=True)
        return (X - min_val) / (max_val - min_val + 1e-8), min_val, max_val
    
    elif method == "unit":
        norm = np.linalg.norm(X, axis=axis, keepdims=True)
        return X / (norm + 1e-8), norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_data(X_normalized, normalization_params, method="standard"):
    if method == "standard":
        mean, std = normalization_params
        return X_normalized * std + mean
    
    elif method == "minmax":
        min_val, max_val = normalization_params
        return X_normalized * (max_val - min_val) + min_val
    
    elif method == "unit":
        norm = normalization_params
        return X_normalized * norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y))
    
    y_int = y.astype(int)
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y_int] = 1
    return one_hot


def shuffle_data(X, y, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def create_batches(X, y, batch_size, shuffle=True, random_state=42):
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    
    n_samples = len(X)
    batches = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]
        batches.append((batch_X, batch_y))
    
    return batches


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def confusion_matrix(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
    
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true.flatten(), y_pred.flatten()):
        matrix[int(true), int(pred)] += 1
    
    return matrix


def classification_report(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
    
    report = {}
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        report[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': np.sum(cm[i, :])
        }
    
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    report['overall'] = {'accuracy': overall_accuracy}
    
    return report


def save_model(model, filepath):
    try:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model(filepath):
    try:
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def print_summary(title, **kwargs):
    print(f"\n{title}")
    print("=" * len(title))
    for key, value in kwargs.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, (list, np.ndarray)) and len(value) > 5:
            print(f"{key}: [{value[0]:.3f}, {value[1]:.3f}, ..., {value[-1]:.3f}] (length: {len(value)})")
        else:
            print(f"{key}: {value}")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def progress_bar(current, total, bar_length=50, prefix="Progress"):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\r{prefix}: [{arrow}{spaces}] {percent:.1f}%', end='', flush=True)
    
    if current == total:
        print()


if __name__ == "__main__":
    print("🛠️ Utility Functions Demonstration")
    print("=" * 50)
    
    print("\n📊 Data generation examples:")
    data_types = ["linear", "quadratic", "circle", "xor", "spiral"]
    
    for data_type in data_types:
        X, y = generate_data(data_type, n_samples=100)
        print(f"  {data_type.capitalize()}: X shape {X.shape}, y shape {y.shape}")
    
    print("\n🔄 Data preprocessing:")
    X, y = generate_data("linear", n_samples=200)
    
    X_norm, mean, std = normalize_data(X, method="standard")
    print(f"  Standardization: mean={mean.flatten()}, std={std.flatten()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(f"  Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    batches = create_batches(X_train, y_train, batch_size=32)
    print(f"  Batching: {len(batches)} batches of size ~32")
    
    print("\n📈 Metrics examples:")
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    acc = accuracy_score(y_true, y_pred)
    mse = mean_squared_error(y_true.astype(float), y_pred.astype(float))
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    print(f"  Confusion Matrix:\n{cm}")
    
    print("\n✅ Utility functions demonstration complete!")
