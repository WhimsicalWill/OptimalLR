import subprocess

import numpy as np

MODEL_PARAMS = (1, 64, 1)  # Model parameters (depth, channels, heads)
NUM_ITERS = 1250  # Number of iterations to run (full training run is 20_000 steps)
STARTING_LR_LOW = 0.01
STARTING_LR_HIGH = 0.5
LOSS_INCREASE_RATIO = 1
EXPERIMENT_GROUP = "exp_binary_search_scale_1"
S3_BUCKET_NAME = "10605willhw5"


def parse_logfile(log_file_path):
    """
    Parse the log file to extract training and validation losses.
    Args:
        log_file_path (str): Path to the log file.
    Returns:
        dict: Dictionary containing lists of losses and their corresponding steps.
    Raises:
        ValueError: If the log file does not contain more than one data line.
    """
    train_loss_list = []  # "trl" in logs
    val_loss_list = []  # "tel" in logs
    with open(log_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if 'trl' in line:
                step = int(parts[0].split('s:')[1])
                loss = float(parts[1].split('trl:')[1])
                train_loss_list.append(loss)
            elif 'tel' in line:
                step = int(parts[0].split('s:')[1])
                loss = float(parts[1].split('tel:')[1])
                val_loss_list.append(loss)

    if len(train_loss_list) < 2 or len(val_loss_list) < 2:
        raise ValueError("Insufficient data for divergence checks.")

    return train_loss_list, val_loss_list


def check_divergence(log_file_path):
    """
    Check if the training has diverged based on the training logs, considering both training and validation losses.
    Args:
        log_file_path (str): Path to the log file.
    Returns:
        bool: True if training diverged, False otherwise.
    """
    try:
        train_loss_list, val_loss_list = parse_logfile(log_file_path)

        # Check for NaNs or Infs in training or validation losses
        if np.any(np.isnan(train_loss_list) | np.isinf(train_loss_list)):
            return True
        if np.any(np.isnan(val_loss_list) | np.isinf(val_loss_list)):
            return True

        # Check for significant increase in training loss
        if train_loss_list[-1] > train_loss_list[0] * LOSS_INCREASE_RATIO:
            return True

        # Check if any validation loss is higher than the previous one
        for i in range(1, len(val_loss_list)):
            if val_loss_list[i] > val_loss_list[i - 1]:
                return True

    except Exception as e:
        print(f"Error processing log file: {e}")
        return True  # Assume divergence in case of any error processing the log

    return False


def binary_search_lr(low, high, eps=0.001):
    """
    Perform a binary search to find the highest learning rate that does not cause divergence.
    The number of iterations needed is log((high - low) / eps), where eps is the minimum precision.
    Args:
        low (float): Lower bound of the learning rate to test.
        high (float): Upper bound of the learning rate to test.
        eps (float): The minimum precision of the learning rate.
    Returns:
        float: The highest learning rate that does not cause divergence.
    """
    best_lr = low
    while high - low > eps:
        mid = (high + low) / 2
        exp_name = f"exp_binary_search_lr_{mid}"
        log_file = f"./{exp_name}/main.log"

        print(f"Starting training job with LR: {mid}")

        # Run the experiment
        command = [
            './train_gpt2cu',
            '-o', exp_name,
            '-1d', str(MODEL_PARAMS[0]),
            '-1c', str(MODEL_PARAMS[1]),
            '-1h', str(MODEL_PARAMS[2]),
            '-l', str(mid),
            '-x', str(NUM_ITERS)
        ]
        subprocess.run(command, check=True)

        # Check if the training diverged
        if check_divergence(log_file):
            print(f"Training diverged with LR: {mid}")
            high = mid
        else:
            print(f"Training did not diverge with LR: {mid}")
            best_lr = mid
            low = mid

        upload_logs_to_s3(exp_name)

    print(f"The highest learning rate that doesn't diverge is {best_learning_rate}")

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


best_learning_rate = binary_search_lr(STARTING_LR_LOW, STARTING_LR_HIGH)
