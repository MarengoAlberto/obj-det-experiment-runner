import os

def initialize_directory(cfg):

    if os.path.isdir(cfg.checkpoint.dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(cfg.checkpoint.dir)]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = cfg.checkpoint.version

    # Update the training config default directory
    checkpoint_dir = os.path.join(cfg.checkpoint.dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Model Checkpoint at: {checkpoint_dir}")

    return checkpoint_dir, version_name
