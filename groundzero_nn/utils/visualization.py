import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation


def plot_neuron_state(neuron, step_info=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{neuron.name} - Current State', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')
    
    input_positions = [(1, i + 2) for i in range(neuron.num_inputs)]
    for i, (x, y) in enumerate(input_positions):
        circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(x, y, f'x{i}', ha='center', va='center', fontweight='bold')
        
        weight_val = neuron.weights[i]
        color = 'red' if weight_val < 0 else 'green'
        width = abs(weight_val) * 3 + 0.5
        ax1.plot([x + 0.3, 6.5], [y, 4], color=color, linewidth=width)
        ax1.text((x + 6.5) / 2, (y + 4) / 2 + 0.3, f'w{i}={weight_val:.2f}',
                ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    neuron_circle = plt.Circle((7, 4), 0.8, color='yellow', ec='black', linewidth=2)
    ax1.add_patch(neuron_circle)
    ax1.text(7, 4, 'Σ', ha='center', va='center', fontsize=20, fontweight='bold')
    
    output_circle = plt.Circle((9, 4), 0.3, color='lightcoral', ec='black')
    ax1.add_patch(output_circle)
    ax1.text(9, 4, 'y', ha='center', va='center', fontweight='bold')
    
    ax1.plot([7.8, 8.7], [4, 4], 'k-', linewidth=2)
    ax1.text(7, 2.5, f'bias={neuron.bias:.2f}', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax1.set_title('Neuron Architecture')
    ax1.axis('off')
    
    ax2 = axes[0, 1]
    weights_with_bias = np.append(neuron.weights, neuron.bias)
    weight_labels = [f'w{i}' for i in range(neuron.num_inputs)] + ['bias']
    
    colors = ['red' if w < 0 else 'green' for w in weights_with_bias]
    bars = ax2.bar(weight_labels, weights_with_bias, color=colors, alpha=0.7)
    
    for bar, val in zip(bars, weights_with_bias):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    ax2.set_title('Weights and Bias')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax3 = axes[0, 2]
    if hasattr(neuron, 'training_history') and neuron.training_history:
        history = neuron.training_history
        steps = range(len(history['loss']))
        ax3.plot(steps, history['loss'], 'b-', linewidth=2, label='Loss')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training History')
    
    ax4 = axes[1, 0]
    x_range = np.linspace(-3, 3, 100)
    if hasattr(neuron, 'last_raw_output'):
        z = neuron.last_raw_output
        y_vals = [neuron.activation_function.forward(np.array([x]))[0] for x in x_range]
        ax4.plot(x_range, y_vals, 'b-', linewidth=2, label=f'{neuron.activation_function.name}')
        ax4.axvline(x=z, color='red', linestyle='--', alpha=0.7, label=f'Current z={z:.2f}')
        ax4.axhline(y=neuron.last_output, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current output={neuron.last_output:.2f}')
        ax4.plot(z, neuron.last_output, 'ro', markersize=8)
    else:
        y_vals = [neuron.activation_function.forward(np.array([x]))[0] for x in x_range]
        ax4.plot(x_range, y_vals, 'b-', linewidth=2, label=f'{neuron.activation_function.name}')
    
    ax4.set_xlabel('Input (z)')
    ax4.set_ylabel('Output')
    ax4.set_title('Activation Function')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = axes[1, 1]
    if hasattr(neuron, 'training_history') and neuron.training_history:
        history = neuron.training_history
        if 'weights' in history and len(history['weights']) > 0:
            weight_history = np.array(history['weights'])
            steps = range(len(weight_history))
            
            for i in range(neuron.num_inputs):
                ax5.plot(steps, weight_history[:, i], label=f'w{i}', linewidth=2)
            
            ax5.set_xlabel('Training Steps')
            ax5.set_ylabel('Weight Value')
            ax5.set_title('Weight Evolution')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No weight history available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Weight Evolution')
    else:
        ax5.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Weight Evolution')
    
    ax6 = axes[1, 2]
    if step_info:
        info_text = ""
        for key, value in step_info.items():
            if isinstance(value, float):
                info_text += f"{key}: {value:.4f}\n"
            else:
                info_text += f"{key}: {value}\n"
        
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        neuron_info = f"""
Neuron Information:
• Inputs: {neuron.num_inputs}
• Activation: {neuron.activation_function.name}
• Learning Rate: {neuron.learning_rate}
• Last Output: {neuron.last_output:.4f}
        """
        ax6.text(0.1, 0.9, neuron_info, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax6.set_title('Information')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_progress(history, title="Training Progress"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    steps = range(len(history['loss']))
    
    axes[0, 0].plot(steps, history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    if 'accuracy' in history:
        axes[0, 1].plot(steps, history['accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'weights' in history and len(history['weights']) > 0:
        weight_history = np.array(history['weights'])
        for i in range(weight_history.shape[1]):
            axes[1, 0].plot(steps, weight_history[:, i], label=f'w{i}', linewidth=2)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].set_title('Weight Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    if 'bias' in history:
        axes[1, 1].plot(steps, history['bias'], 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Bias Value')
        axes[1, 1].set_title('Bias Evolution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_decision_boundary_2d(model, X, y, resolution=100, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(model, 'predict'):
        Z = model.predict(grid_points)
    else:
        Z = np.array([model.forward(point) for point in grid_points])
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
    plt.colorbar(scatter)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def animate_training_process(training_data, interval=100, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        step_data = training_data[frame]
        
        ax1.plot(range(frame + 1), [data['loss'] for data in training_data[:frame + 1]], 'b-')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss (Step {frame})')
        ax1.grid(True, alpha=0.3)
        
        if 'weights' in step_data:
            weights = step_data['weights']
            ax2.bar(range(len(weights)), weights, alpha=0.7)
            ax2.set_xlabel('Weight Index')
            ax2.set_ylabel('Weight Value')
            ax2.set_title(f'Weights (Step {frame})')
            ax2.grid(True, alpha=0.3)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(training_data), 
                                  interval=interval, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    plt.show()
    return anim


def plot_multiple_neurons_comparison(neurons, title="Neurons Comparison"):
    n_neurons = len(neurons)
    fig, axes = plt.subplots(2, n_neurons, figsize=(4 * n_neurons, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if n_neurons == 1:
        axes = axes.reshape(2, 1)
    
    for i, neuron in enumerate(neurons):
        ax1 = axes[0, i]
        weights_with_bias = np.append(neuron.weights, neuron.bias)
        weight_labels = [f'w{j}' for j in range(neuron.num_inputs)] + ['bias']
        
        colors = ['red' if w < 0 else 'green' for w in weights_with_bias]
        ax1.bar(weight_labels, weights_with_bias, color=colors, alpha=0.7)
        ax1.set_title(f'{neuron.name} - Weights')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1, i]
        x_range = np.linspace(-3, 3, 100)
        y_vals = [neuron.activation_function.forward(np.array([x]))[0] for x in x_range]
        ax2.plot(x_range, y_vals, 'b-', linewidth=2)
        
        if hasattr(neuron, 'last_raw_output'):
            ax2.axvline(x=neuron.last_raw_output, color='red', linestyle='--', alpha=0.7)
            ax2.plot(neuron.last_raw_output, neuron.last_output, 'ro', markersize=8)
        
        ax2.set_title(f'{neuron.name} - Activation')
        ax2.set_xlabel('Input')
        ax2.set_ylabel('Output')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🎨 Visualization Utilities Demonstration")
    print("=" * 50)
    
    try:
        from ..core.neuron import Neuron
        
        print("\n📊 Creating sample neuron for visualization...")
        neuron = Neuron(num_inputs=3, activation='relu', name="Demo_Neuron")
        
        sample_input = np.array([0.5, -0.3, 0.8])
        output = neuron.forward(sample_input)
        
        print(f"Sample forward pass: input={sample_input}, output={output:.4f}")
        
        print("\n🎨 Generating neuron state visualization...")
        plot_neuron_state(neuron)
        
        print("\n📈 Creating sample training history...")
        history = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
            'weights': [[0.1, 0.2, 0.3], [0.15, 0.25, 0.28], [0.2, 0.3, 0.25]],
            'bias': [0.0, 0.1, 0.05]
        }
        
        plot_training_progress(history, "Sample Training Progress")
        
        print("\n✅ Visualization demonstration complete!")
        
    except ImportError:
        print("⚠️  Could not import Neuron class for demonstration")
        print("This is expected when running the module independently")
        
        print("\n📊 Available visualization functions:")
        functions = [
            'plot_neuron_state', 'plot_training_progress', 
            'plot_decision_boundary_2d', 'animate_training_process',
            'plot_multiple_neurons_comparison'
        ]
        
        for func in functions:
            print(f"  • {func}")
        
        print("\n✅ Visualization utilities loaded successfully!")
