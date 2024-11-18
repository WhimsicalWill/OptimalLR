import subprocess
import numpy as np

MODEL_DEPTH = 1
MODEL_CHANNELS = 64
MODEL_HEADS = 1
NUM_ITERS = 1250  # Number of iterations to run
STARTING_LR_LOW = 1e-4  # 6e-4 is recommended in GPT-3 paper
STARTING_LR_HIGH = 0.16
EXPERIMENT_GROUP = "exp_halving_search_real_scale_1"
S3_BUCKET_NAME = "10605willhw5"


def parse_logfile(log_file_path):
    """
    Parse the log file to extract validation losses.
    Args:
        log_file_path (str): Path to the log file.
    Returns:
        float: Final validation loss in the log file.
    Raises:
        ValueError: If the log file does not contain enough data lines.
    """
    val_loss_list = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'tel' in line:
                parts = line.strip().split()
                loss = float(parts[1].split('tel:')[1])
                val_loss_list.append(loss)

    if not val_loss_list:
        raise ValueError("No validation loss data found.")

    return val_loss_list[-1]  # Return the last recorded validation loss


def successive_halving_lr(low, high, eps=0.001):
    """
    Perform a successive halving search to find the learning rate that minimizes validation loss.
    Args:
        low (float): Lower bound of the learning rate to start the search.
        high (float): Upper bound of the learning rate to start the search.
        eps (float): The precision of the learning rate search.
    """
    N = 3  # Number of points to test within the interval
    lr_cache = {}  # Cache to store learning rates and their validation losses

    while high - low > eps:
        gap = (high - low) / (N + 1)
        lr_list = [low + i * gap for i in range(1, N+1)]
        val_losses = []

        print(f"Running training jobs with LRs: {lr_list}")

        for lr in lr_list:
            if lr in lr_cache:
                val_loss = lr_cache[lr]
                print(f"Using cached result for LR: {lr}")
            else:
                exp_name = f"exp_halving_search_{lr}"
                log_file = f"./{exp_name}/main.log"
                command = [
                    './train_gpt2cu',
                    '-o', exp_name,
                    '-1d', str(MODEL_DEPTH),
                    '-1c', str(MODEL_CHANNELS),
                    '-1h', str(MODEL_HEADS),
                    '-l', str(lr),
                    '-x', str(NUM_ITERS)
                ]

                print(f"Running training job with LR: {lr}")
                subprocess.run(command, check=True)
                val_loss = parse_logfile(log_file)
                lr_cache[lr] = val_loss
                upload_logs_to_s3(exp_name)

            val_losses.append(val_loss)

        best_idx = np.argmin(val_losses)
        # Adjust interval to include the neighbors of the best index
        if best_idx == 0:
            high = lr_list[best_idx + 1]
        elif best_idx == N - 1:
            low = lr_list[best_idx - 1]
        else:
            low = lr_list[best_idx - 1]
            high = lr_list[best_idx + 1]

        print(f"Updated interval: [{low}, {high}] with losses: {val_losses}")

    best_lr = (low + high) / 2
    print(f"The learning rate that minimizes the validation loss is approximately: {best_lr}")

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


successive_halving_lr(STARTING_LR_LOW, STARTING_LR_HIGH)
