import numpy as np
import math

def get_hyperband_config(R, eta=3):
    """
    Generate the configuration for Hyperband.
    R: maximum amount of resource per configuration (e.g., epochs or dataset fraction)
    eta: downsampling rate (how much to reduce configurations at each round)
    """
    # Compute max possible iterations and initial number of configurations
    s_max = int(math.log(R, eta))
    B = (s_max + 1) * R

    configs = []
    for s in reversed(range(s_max + 1)):
        n = int(math.ceil(int(B / R / (s + 1)) * eta ** s))  # Initial number of configurations
        r = R * eta ** (-s)  # Initial amount of resources per configuration
        configs.append((n, r))
    return configs

# Example settings
R = 81  # Max epochs or dataset fraction
hyperband_config = get_hyperband_config(R)
print("Hyperband Configuration (n_configs, initial_resource):", hyperband_config)

def successive_halving(n, r, eta, eval_function):
    """
    Perform one round of the Successive Halving algorithm.
    n: number of configurations
    r: amount of resource per configuration
    eta: reduction factor
    eval_function: function to evaluate a configuration
    """
    # Initial number of configurations
    num_configs = n
    resource = r

    # Dictionary to store the performance of configurations
    configs_performance = {}

    # Evaluate initial configurations
    for i in range(num_configs):
        config = np.random.uniform(0.001, 0.1)  # Randomly generated hyperparameter, e.g., learning rate
        performance = eval_function(config, resource)
        configs_performance[config] = performance

    # Reduce configurations
    while num_configs > 1:
        num_configs = int(num_configs / eta)
        resource *= eta

        # Select top configurations based on performance
        top_configs = sorted(configs_performance, key=configs_performance.get, reverse=True)[:num_configs]
        configs_performance = {config: eval_function(config, resource) for config in top_configs}

    return configs_performance

def eval_function(learning_rate, epochs):
    """
    A dummy function to simulate model evaluation.
    learning_rate: the hyperparameter to evaluate
    epochs: number of epochs to simulate
    """
    loss = 1 / (learning_rate * epochs)  # Simplified loss computation
    return -loss  # Return negative loss because higher is better in this context

def run_hyperband():
    configs = get_hyperband_config(R)
    results = {}
    for i, (n, r) in enumerate(configs):
        print(f"Running bracket {i} with {n} configs at {r} resources each")
        results[i] = successive_halving(n, r, 3, eval_function)
    return results

# Execute Hyperband
hyperband_results = run_hyperband()
print("Hyperband Results:", hyperband_results)
