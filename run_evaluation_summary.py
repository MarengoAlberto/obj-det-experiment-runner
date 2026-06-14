import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

from models import Model, utils

model_root = Path("trained_model")
csv_path = Path("evaluation/experiment_results.csv")
experiments_root = Path("experiments")

def find_pth_files(root_folder: str | Path) -> list[Path]:
    """
    Return all .pth files under root_folder recursively.
    """
    root = Path(root_folder).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    return list(root.rglob("*.pth"))

def get_fastest_device() -> torch.device:
    """
    Return the fastest available PyTorch device:
    CUDA GPU > Apple MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")

    return torch.device("cpu")

def run_model_evaluation(cfg, path_to_model: str):
    cfg.model.metadata.best_model_folder = os.path.dirname(path_to_model)
    if OmegaConf.select(cfg, "model.attention_type") is not None:
        attention_type = cfg.model.attention_type
        print("attention_type:", attention_type)
    else:
        print("attention_type does not exist")
        cfg.model.attention_type = None
    model_name = os.path.splitext(os.path.basename(path_to_model))[0]
    model = Model(cfg, model_name=model_name)
    model.device = get_fastest_device()
    print(f"Running evaluation on {model.device}")
    dataset = cfg.dataset
    _, _, data_yaml = utils.process_yaml(dataset)
    model.data_yaml = data_yaml
    return model.evaluate('test' if 'test' in data_yaml else 'val')

if __name__ == "__main__":
    pth_files = find_pth_files(model_root)

    for pth_file in pth_files:
        model_name, model_tag = utils.get_model_name(pth_file)
        print(f"Evaluating {model_name} ({model_tag})")
        eval_model_name = f"{model_name}_{model_tag}"

        if utils.model_exists_in_csv(csv_path, eval_model_name):
            print(f"Skipping {eval_model_name} because it already exists in the CSV.")
            continue

        try:
            cfg = utils.load_model_yaml(model_name, experiments_root)
        except Exception as e:
            print(f"Error loading YAML for {model_name}: {e}")
            continue

        try:
            map_results = run_model_evaluation(cfg, str(pth_file))
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            continue

        try:
            appended = utils.append_model_results_to_csv(
                csv_path,
                eval_model_name,
                cfg,
                map_results,
            )
            if appended:
                print(f"Appended results for {eval_model_name} to CSV.")
            else:
                print(f"Skipped appending for {eval_model_name} because it already exists in CSV.")
        except Exception as e:
            print(f"Error appending results for {eval_model_name} to CSV: {e}")