import numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = "BaseOptimizer"
    
    def update(self, params, gradients):
        raise NotImplementedError
    
    def reset(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.name = "SGD"
    
    def update(self, params, gradients):
        updated_params = {}
        for key in params:
            updated_params[key] = params[key] - self.learning_rate * gradients[key]
        return updated_params
    
    def get_info(self):
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'properties': ['Simple', 'Fast', 'Can be noisy']
        }


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.name = f"SGD_Momentum(β={momentum})"
        self.velocities = {}
    
    def update(self, params, gradients):
        updated_params = {}
        
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            
            self.velocities[key] = (self.momentum * self.velocities[key] + 
                                  self.learning_rate * gradients[key])
            
            updated_params[key] = params[key] - self.velocities[key]
        
        return updated_params
    
    def reset(self):
        self.velocities = {}
    
    def get_info(self):
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'properties': ['Accelerated', 'Stable', 'Momentum-based']
        }


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.name = f"RMSprop(ρ={decay_rate})"
        self.squared_gradients = {}
    
    def update(self, params, gradients):
        updated_params = {}
        
        for key in params:
            if key not in self.squared_gradients:
                self.squared_gradients[key] = np.zeros_like(params[key])
            
            self.squared_gradients[key] = (
                self.decay_rate * self.squared_gradients[key] + 
                (1 - self.decay_rate) * gradients[key] ** 2
            )
            
            updated_params[key] = params[key] - (
                self.learning_rate * gradients[key] / 
                (np.sqrt(self.squared_gradients[key]) + self.epsilon)
            )
        
        return updated_params
    
    def reset(self):
        self.squared_gradients = {}
    
    def get_info(self):
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate,
            'properties': ['Adaptive', 'Per-parameter learning rates', 'Sparse-friendly']
        }


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = f"Adam(β₁={beta1}, β₂={beta2})"
        
        self.first_moments = {}
        self.second_moments = {}
        self.time_step = 0
    
    def update(self, params, gradients):
        self.time_step += 1
        updated_params = {}
        
        for key in params:
            if key not in self.first_moments:
                self.first_moments[key] = np.zeros_like(params[key])
                self.second_moments[key] = np.zeros_like(params[key])
            
            self.first_moments[key] = (
                self.beta1 * self.first_moments[key] + 
                (1 - self.beta1) * gradients[key]
            )
            
            self.second_moments[key] = (
                self.beta2 * self.second_moments[key] + 
                (1 - self.beta2) * gradients[key] ** 2
            )
            
            first_moment_corrected = (
                self.first_moments[key] / (1 - self.beta1 ** self.time_step)
            )
            
            second_moment_corrected = (
                self.second_moments[key] / (1 - self.beta2 ** self.time_step)
            )
            
            updated_params[key] = params[key] - (
                self.learning_rate * first_moment_corrected / 
                (np.sqrt(second_moment_corrected) + self.epsilon)
            )
        
        return updated_params
    
    def reset(self):
        self.first_moments = {}
        self.second_moments = {}
        self.time_step = 0
    
    def get_info(self):
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'time_step': self.time_step,
            'properties': ['Adaptive', 'Momentum + RMSprop', 'Bias correction', 'Industry standard']
        }


def get_optimizer(name, **kwargs):
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'rmsprop': RMSprop,
        'adam': Adam
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name.lower()](**kwargs)


def compare_optimizers_visualization():
    def objective_function(x, y):
        return x**2 + 10*y**2
    
    def gradient_function(x, y):
        return np.array([2*x, 20*y])
    
    start_point = np.array([4.0, 3.0])
    
    optimizers = [
        SGD(learning_rate=0.1),
        SGDMomentum(learning_rate=0.1, momentum=0.9),
        RMSprop(learning_rate=0.1),
        Adam(learning_rate=0.1)
    ]
    
    paths = {}
    num_steps = 100
    
    for opt in optimizers:
        path = [start_point.copy()]
        current_params = {'x': start_point.copy()}
        
        for step in range(num_steps):
            x, y = current_params['x']
            grad = gradient_function(x, y)
            gradients = {'x': grad}
            
            updated = opt.update(current_params, gradients)
            current_params = updated
            path.append(current_params['x'].copy())
            
            if np.linalg.norm(grad) < 1e-6:
                break
        
        paths[opt.name] = np.array(path)
        opt.reset()
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function(X, Y)
    
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (name, path) in enumerate(paths.items()):
        plt.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], 
                label=name, markersize=3, linewidth=2)
        plt.plot(path[0, 0], path[0, 1], 's', color=colors[i], markersize=8)
    
    plt.plot(0, 0, 'k*', markersize=15, label='Global Minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Paths Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    
    for i, (name, path) in enumerate(paths.items()):
        objective_values = [objective_function(p[0], p[1]) for p in path]
        plt.plot(objective_values, color=colors[i], label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Speed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    convergence_steps = {}
    
    for lr in learning_rates:
        convergence_steps[lr] = {}
        
        for opt_class in [SGD, Adam]:
            opt = opt_class(learning_rate=lr)
            current_params = {'x': start_point.copy()}
            
            for step in range(200):
                x, y = current_params['x']
                grad = gradient_function(x, y)
                gradients = {'x': grad}
                
                updated = opt.update(current_params, gradients)
                current_params = updated
                
                if np.linalg.norm(grad) < 1e-6:
                    convergence_steps[lr][opt.name.split('(')[0]] = step
                    break
            else:
                convergence_steps[lr][opt.name.split('(')[0]] = 200
            
            opt.reset()
    
    sgd_steps = [convergence_steps[lr].get('SGD', 200) for lr in learning_rates]
    adam_steps = [convergence_steps[lr].get('Adam', 200) for lr in learning_rates]
    
    plt.plot(learning_rates, sgd_steps, 'o-', label='SGD', linewidth=2)
    plt.plot(learning_rates, adam_steps, 's-', label='Adam', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Steps to Convergence')
    plt.title('Learning Rate Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 4)
    
    properties = ['Speed', 'Stability', 'Memory', 'Generality']
    sgd_scores = [3, 2, 5, 4]
    momentum_scores = [4, 4, 4, 4]
    rmsprop_scores = [4, 4, 3, 4]
    adam_scores = [5, 5, 2, 5]
    
    x_pos = np.arange(len(properties))
    width = 0.2
    
    plt.bar(x_pos - 1.5*width, sgd_scores, width, label='SGD', alpha=0.8)
    plt.bar(x_pos - 0.5*width, momentum_scores, width, label='SGD+Momentum', alpha=0.8)
    plt.bar(x_pos + 0.5*width, rmsprop_scores, width, label='RMSprop', alpha=0.8)
    plt.bar(x_pos + 1.5*width, adam_scores, width, label='Adam', alpha=0.8)
    
    plt.xlabel('Properties')
    plt.ylabel('Score (1-5)')
    plt.title('Optimizer Properties Comparison')
    plt.xticks(x_pos, properties)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("⚡ Optimizers Demonstration")
    print("=" * 50)
    
    optimizers = [
        SGD(learning_rate=0.01),
        SGDMomentum(learning_rate=0.01, momentum=0.9),
        RMSprop(learning_rate=0.001),
        Adam(learning_rate=0.001)
    ]
    
    for opt in optimizers:
        info = opt.get_info()
        print(f"\n📊 {info['name']}")
        print(f"   Learning Rate: {info['learning_rate']}")
        print(f"   Properties: {', '.join(info['properties'])}")
    
    print("\n🎨 Generating optimizer comparison visualization...")
    compare_optimizers_visualization()
    
    print("\n✅ Optimizers demonstration complete!")
