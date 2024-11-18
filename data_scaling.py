import subprocess

# Fixed array of LRs to use in different experiments
learning_rates = [0.02, 0.04, 0.08, 0.16, 0.32]

def run_data_experiments(num_iters):
    """
    Run experiments with a given data scale and different learning rates.

    Args:
        num_iters: number of iterations to run
    """

    # Loop through each learning rate and execute the training command
    for i, lr in enumerate(learning_rates):
        exp_name = f"exp_data_scaling_ni_{num_iters}_lr_{lr}"

        print(f"Running experiment: {exp_name} with learning rate: {lr}")

        # Construct the command
        command = [
            './train_gpt2cu',
            '-o', exp_name,
            '-1d', '6',
            '-1c', '384',
            '-1h', '6',
            '-l', str(lr),
            '-x', num_iters
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
run_data_experiments(1000)
run_data_experiments(1500)
run_data_experiments(2000)