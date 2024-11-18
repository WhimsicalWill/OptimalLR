import numpy as np
from scipy.optimize import curve_fit

# Power law function
def power_law(x, a, b):
    return a * np.power(x, b)

# Example dataset sizes and corresponding optimal learning rates from your 10 runs
dataset_sizes = np.array([39e6, 156e6, 625e6])  # Example dataset sizes
optimal_lrs = np.array([0.01, 0.005, 0.0025])   # Example average optimal LRs for each dataset size

# Fit the power law
params, covariance = curve_fit(power_law, dataset_sizes, optimal_lrs)

print("Fitted parameters:", params)  # a and b in the power law equation

# The below code can be used to generate random learning rate parameters

mu = predicted_lr  # Mean as the predicted learning rate
sigma = 0.001      # Example standard deviation
N = 10             # Number of learning rates to generate
random_lrs =  np.random.normal(mu, sigma, N)
print("Randomly generated learning rates:", random_lrs)
