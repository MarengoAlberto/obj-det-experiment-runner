import os
import yaml
import torch
import numbers
import numpy as np
import requests
import zipfile
from typing import Union
from box import Box
from collections.abc import Mapping

def load_model(model, model_folder: str, model_name: str = "my_yolo"):
    path = os.path.join(model_folder, f"{model_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    # Load trained model's state dict.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
    except:
        model.load_state_dict(torch.load(path, map_location=device))
        start_epoch = 0
    return model.to(device), start_epoch

def _direct_download_url(url: str) -> str:
    # Turn a Dropbox share link into a direct download URL
    if "dropbox.com" in url and "dl=1" not in url:
        if "?" in url:
            return url + "&dl=1"
        return url + "?dl=1"
    return url

def download_and_unzip_zip(url: str, extract_dir: str = 'dataset', zip_name: Union[str, None] = None, timeout: int = 60):
    """
    Download a ZIP from `url` into `save_dir` and unzip it.
    Skips work if extracted folder already exists.
    Validates the downloaded file is a real ZIP.
    """
    os.makedirs(extract_dir, exist_ok=True)
    url = _direct_download_url(url)

    if zip_name is None:
        zip_name = "dataset.zip"

    zip_path = os.path.join(extract_dir, zip_name)

    # If already extracted, nothing to do
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"✅ Already extracted: {extract_dir}")
        return extract_dir

    # (Re)download if file missing or not a valid zip
    need_download = True
    if os.path.exists(zip_path):
        # Quick validity check
        try:
            with open(zip_path, "rb") as f:
                magic = f.read(4)
            if magic == b"PK\x03\x04" and zipfile.is_zipfile(zip_path):
                need_download = False
            else:
                print("⚠️ Existing file is not a valid ZIP. Re-downloading...")
        except Exception:
            print("⚠️ Could not read existing file. Re-downloading...")

    if need_download:
        tmp_path = zip_path + ".partial"
        # clean up any partial
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

        print(f"⬇️  Downloading: {url}")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, zip_path)
        print(f"✅ Downloaded to: {zip_path}")

        # Validate ZIP after download
        with open(zip_path, "rb") as f:
            if f.read(4) != b"PK\x03\x04" or not zipfile.is_zipfile(zip_path):
                raise ValueError(
                    "Downloaded file is not a valid ZIP. "
                    "If this is a Dropbox/Drive link, ensure it's a direct download (e.g., ?dl=1)."
                )

    # Unzip
    print(f"📂 Extracting to: {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("✅ Extraction complete")
    return extract_dir

def check_data_exists(data_path: str, data_dir: str = 'dataset'):

    with open(data_path, 'r') as file:
        data = yaml.safe_load(file)

    needs_download = True
    url = data['metadata']['url']
    train_folder = os.path.join(data_dir, data['train'].replace('../', ''))
    val_folder = os.path.join(data_dir, data['val'].replace('../', ''))
    if os.path.isdir(train_folder) and os.listdir(val_folder):
        needs_download = False
    data['full_train_path'] = train_folder
    data['full_val_path'] = val_folder
    return needs_download, url, Box(data)

def get_val_yaml_file_path(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at: {data_path}")
    return Box({"full_val_path": data_path})

def to_python_number(value):
    """Convert common tensor/NumPy scalar types to plain Python numbers."""
    if isinstance(value, numbers.Number):
        return value

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape {value.shape}")
        return value.item()

    raise TypeError(f"Unsupported metric value type: {type(value)}")

def merge_metric_dicts(*dicts: Mapping, prefix_conflicts: bool = False) -> dict[str, float]:
    """
    Merge multiple metric dicts and convert values to plain Python numbers.

    Args:
        *dicts: Metric dictionaries.
        prefix_conflicts: If True, later duplicate keys are allowed only if
            you pre-prefix keys yourself before calling this. Otherwise duplicates error.

    Returns:
        Flat dict with Python scalar values.
    """
    merged: dict[str, float] = {}

    for metric_dict in dicts:
        for key, value in metric_dict.items():
            if isinstance(value, dict):
                continue
            if not isinstance(key, str):
                key = str(key)

            if key in merged and not prefix_conflicts:
                raise KeyError(f"Duplicate metric key: {key}")
            num_value = to_python_number(value)
            if num_value is not None:
                merged[key] = num_value

    return merged

def refactor_dict(d: dict, prefix: str) -> dict:
    """Add a prefix to all keys in the dictionary."""
    return {f"{prefix}_{k}": v for k, v in d.items()}
