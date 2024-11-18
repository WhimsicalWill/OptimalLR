import subprocess

# Fixed array of LRs to use in different experiments
NUM_ITERS = 5000
MODEL_DEPTH = 1
MODEL_CHANNELS = 64
MODEL_HEADS = 1
EXPERIMENT_GROUP = "exp_model_scaling_sf_1"
S3_BUCKET_NAME = "10605willhw5"
LEARNING_RATES = [1.5355e-1, 5.75e-2, 8e-2, 1.325e-1]

def run_model_experiments(scaling_factor):
    """
    Run experiments with a given data scale and different learning rates.

    Args:
        scaling_factor (int): The scaling factor to multiply the base model params by.
    """
    for lr in LEARNING_RATES:
        exp_name = f"exp_model_scaling_sf_{scaling_factor}_lr_{lr}"

        print(f"Running experiment: {exp_name} with learning rate: {lr}")

        # Construct the command
        command = [
            './train_gpt2cu',
            '-o', exp_name,
            '-1d', scaling_factor * MODEL_DEPTH,
            '-1c', scaling_factor * MODEL_CHANNELS,
            '-1h', scaling_factor * MODEL_HEADS,
            '-l', str(lr),
            '-x', NUM_ITERS
        ]

        # Execute the command
        subprocess.run(command, check=True)

        upload_logs_to_s3(exp_name)

    # Shutdown the machine after all experiments are done
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])


def upload_logs_to_s3(exp_name):
    """
    Upload the experiment results to S3.
    Args:
        exp_name (str): Name of the experiment directory.
    """
    try:
        upload_command = [
            'python', 'upload_to_s3.py',
            S3_BUCKET_NAME,
            exp_name,
            EXPERIMENT_GROUP,
        ]
        subprocess.run(upload_command, check=True)
        print(f"Results uploaded to S3 for experiment: {exp_name}")
    except Exception as e:
        print(f"Error uploading results to S3: {e}")


# Settings used for full training run:
    # depth: 6
    # channels: 384
    # heads: 6
    # 20000 steps
# its basically an hour per 1000 steps
run_model_experiments(1)
