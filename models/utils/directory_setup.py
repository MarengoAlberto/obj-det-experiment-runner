import os

def initialize_directory(cfg):

    if os.path.isdir(cfg.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(cfg.root_log_dir)]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = cfg.log_dir

    # Update the training config default directory
    cfg.log_dir = os.path.join(cfg.root_log_dir, version_name)
    cfg.checkpoint_dir = os.path.join(cfg.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"Logging at: {cfg.log_dir}")
    print(f"Model Checkpoint at: {cfg.checkpoint_dir}")

    return cfg, version_name
