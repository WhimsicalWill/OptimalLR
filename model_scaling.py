import subprocess

# Fixed array of LRs to use in different experiments
NUM_ITERS = 5000
BASE_MODEL_PARAMS = (1, 64, 1)  # base settings for depth, channels, heads
learning_rates = [0.02, 0.04, 0.08, 0.16, 0.32]

def run_model_experiments(scaling_factor):
    """
    Run experiments with a given data scale and different learning rates.

    Args:
        scaling_factor (int): The scaling factor to multiply the base model params by.
    """

    depth, channels, heads = (scaling_factor * param for param in BASE_MODEL_PARAMS)

    # Loop through each learning rate and execute the training command
    for i, lr in enumerate(learning_rates):
        exp_name = f"exp_model_scaling_sf_{scaling_factor}_lr_{lr}"

        print(f"Running experiment: {exp_name} with learning rate: {lr}")

        # Construct the command
        command = [
            './train_gpt2cu',
            '-o', exp_name,
            '-1d', depth,
            '-1c', channels,
            '-1h', heads,
            '-l', str(lr),
            '-x', NUM_ITERS
        ]

        # Execute the command
        subprocess.run(command)

    # Shutdown the machine after all experiments are done
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

# Settings used for full training run:
    # depth: 6
    # channels: 384
    # heads: 6
    # 20000 steps
# its basically an hour per 1000 steps, and we have 5 different LR settings
# another thing to consider is that first 700 steps is warmup
run_data_experiments(1)
# run_data_experiments(2)
# run_data_experiments(3)
