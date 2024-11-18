import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def power_law(x, a, k):
    return a * np.power(x, k)

# Data points based on the LLaMa 3 scaling strategy example
model_params = np.array([8e9, 70e9, 405e9])  # Model sizes in parameters
max_lrs = np.array([3e-4, 1.5e-4, 8e-5])   # Corresponding max learning rates

# Fit the power law to the data
params, _ = curve_fit(power_law, model_params, max_lrs)

print("Fitted parameters: a =", params[0], ", k =", params[1])

# Plotting for visualization
x_vals = np.linspace(min(model_params), max(model_params), 400)
y_vals = power_law(x_vals, *params)
plt.plot(model_params, max_lrs, 'o', label='Data points')
plt.plot(x_vals, y_vals, '-', label='Fitted power law')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Model Parameters')
plt.ylabel('Max Learning Rate')
plt.title('Power Law Fit to Max Learning Rate vs Model Parameters')
plt.legend()
plt.grid(True)
plt.savefig('llama3_power_law_fit.png')
# plt.show()

# Predictions for max learning rate given different model sizes
# Note: our full model will have roughly 30M parameters
# num_params_list = [4e6, 8e6, 15e6, 30e6]
num_params_list = [3.3e6, 7e6, 11.2e6, 16.3e6, 30.4e6]
for num_params in num_params_list:
    max_lr = power_law(num_params, *params)
    print(f"Predicted max LR for {num_params} parameters: {max_lr}")
