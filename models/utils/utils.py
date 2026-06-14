import os
import time
import torch
import numbers
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import requests
import zipfile
from omegaconf import DictConfig, OmegaConf
from typing import Union, cast, Any, Tuple
from collections.abc import Mapping
from pathlib import Path
import json
import re

def load_model(model, model_folder: str, *args, **kwargs):
    model_name = kwargs.get("model_name", "my_yolo")
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

    zip_name_str = "dataset.zip" if zip_name is None else str(zip_name)

    zip_path = os.path.join(extract_dir, zip_name_str)
    lock_path = os.path.join(extract_dir, ".download.lock")
    ready_marker = os.path.join(extract_dir, ".extract_complete")

    # Explicit completion marker avoids false positives when the folder contains only a partial zip.
    if os.path.exists(ready_marker):
        print(f"✅ Already extracted: {extract_dir}")
        return extract_dir

    # Serialize download+extract across local processes (DDP workers on the same node).
    lock_fd = None
    start_wait = time.time()
    lock_timeout = max(timeout * 5, 300)
    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if os.path.exists(ready_marker):
                print(f"✅ Already extracted by another worker: {extract_dir}")
                return extract_dir
            if time.time() - start_wait > lock_timeout:
                raise TimeoutError(f"Timed out waiting for dataset lock: {lock_path}")
            time.sleep(0.5)

    try:
        # Another process may have completed extraction while we were waiting.
        if os.path.exists(ready_marker):
            print(f"✅ Already extracted by another worker: {extract_dir}")
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

        # Marker is written only after successful extraction.
        with open(ready_marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        print("✅ Extraction complete")
        return extract_dir
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

def process_yaml(data_cfg: DictConfig, default_data_dir: str):
    data_dir = data_cfg.path if "path" in data_cfg else default_data_dir
    needs_download = True
    url = data_cfg.metadata.url if "url" in data_cfg.metadata else None
    train_folder = os.path.join(data_dir, data_cfg.train.replace('../', ''))
    val_folder = os.path.join(data_dir, data_cfg.val.replace('../', ''))
    if 'test' in data_cfg:
        test_folder = os.path.join(data_dir, data_cfg.test.replace('../', ''))
    if (
            os.path.isdir(train_folder)
            and os.path.isdir(val_folder)
            and bool(os.listdir(train_folder))
            and bool(os.listdir(val_folder))
    ):
        needs_download = False
    data_cfg.full_train_path = train_folder
    data_cfg.full_val_path = val_folder
    if 'test' in data_cfg:
        data_cfg.full_test_path = test_folder
    return needs_download, url, data_cfg

def check_data_exists(yaml_path: str, default_data_dir: str = 'dataset'):
    data_cfg = OmegaConf.load(yaml_path)
    return process_yaml(data_cfg, default_data_dir)

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

def refactor_dict(d: dict, prefix: str) -> dict:
    """
    Add a prefix to all top-level keys.

    Example:
        {"loss": 1.0} with prefix="train" -> {"train_loss": 1.0}
    """
    if d is None:
        return {}

    return {f"{prefix}_{k}": v for k, v in d.items()}

def _metric_value_to_float(value: Any) -> float | None:
    """
    Convert scalar-like metric values to Python float.

    Returns None for non-scalar tensors/arrays/lists.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return None

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return None

    if isinstance(value, np.generic):
        return float(value.item())

    if isinstance(value, numbers.Number):
        return float(value)

    return None

def merge_metric_dicts(
    *dicts: Mapping,
    prefix_conflicts: bool = False,
    class_names: list[str] | None = None,
    drop_classes_vector: bool = True,
) -> dict[str, float]:
    """
    Merge metric dictionaries into a flat W&B-safe scalar dict.

    Handles:
      - scalar tensors -> float
      - scalar numpy values -> float
      - vector tensors -> expanded as key/class_name or key/index
      - nested dicts -> flattened recursively
      - classes vector -> dropped by default

    Example:
        {"val_map_per_class": tensor([0.1, 0.2])}
        ->
        {
            "val_map_per_class/class0": 0.1,
            "val_map_per_class/class1": 0.2,
        }
    """
    merged: dict[str, float] = {}

    def add_metric(key: str, value: float):
        key = str(key)

        if key in merged and not prefix_conflicts:
            raise KeyError(f"Duplicate metric key: {key}")

        merged[key] = float(value)

    def consume(prefix: str, value: Any):
        prefix = str(prefix)

        # Flatten nested dictionaries.
        if isinstance(value, Mapping):
            for k, v in value.items():
                new_key = f"{prefix}_{k}" if prefix else str(k)
                consume(new_key, v)
            return

        # Drop class index vectors like tensor([0, 1, ..., 9]).
        if drop_classes_vector and prefix.endswith("classes"):
            return

        # Scalar values.
        scalar = _metric_value_to_float(value)
        if scalar is not None:
            add_metric(prefix, scalar)
            return

        # Torch vector.
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()

            if value.ndim == 1:
                for i, v in enumerate(value.tolist()):
                    suffix = class_names[i] if class_names and i < len(class_names) else str(i)
                    add_metric(f"{prefix}/{suffix}", float(v))
            return

        # NumPy vector.
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                for i, v in enumerate(value.tolist()):
                    suffix = class_names[i] if class_names and i < len(class_names) else str(i)
                    add_metric(f"{prefix}/{suffix}", float(v))
            return

        # Python list/tuple vector.
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return

            if all(isinstance(x, numbers.Number) for x in value):
                if len(value) == 1:
                    add_metric(prefix, float(value[0]))
                else:
                    for i, v in enumerate(value):
                        suffix = class_names[i] if class_names and i < len(class_names) else str(i)
                        add_metric(f"{prefix}/{suffix}", float(v))
            return

        # Ignore unsupported values.

    for metric_dict in dicts:
        if metric_dict is None:
            continue

        # Handles accidental tuple-wrapped dicts like ({...},)
        if isinstance(metric_dict, tuple):
            for item in metric_dict:
                if isinstance(item, Mapping):
                    for k, v in item.items():
                        consume(str(k), v)
            continue

        if not isinstance(metric_dict, Mapping):
            continue

        for key, value in metric_dict.items():
            consume(str(key), value)

    return merged

def append_model_results_to_csv(
    csv_path: str | Path,
    model_name: str,
    model_config: dict[str, Any],
    map_results: dict[str, Any],
    model_name_col: str = "model_name",
) -> bool:
    """
    Append one model evaluation result to a CSV file.

    Behavior:
    - Creates the CSV if it does not exist.
    - Skips writing if model_name already exists.
    - Adds config keys and metric keys as columns.
    - Automatically handles new columns by expanding the CSV schema.
    - Converts numpy scalar values, such as np.float64, into regular Python values.

    Returns:
        True if a new row was added.
        False if model_name already existed and the row was skipped.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def normalize_value(value: Any) -> Any:
        """
        Convert values into CSV-friendly formats.
        """
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)

        return value

    row = {
        model_name_col: model_name,
        **{key: normalize_value(value) for key, value in model_config.items()},
        **{key: normalize_value(value) for key, value in map_results.items()},
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    if not df.empty and model_name_col in df.columns:
        existing_models = df[model_name_col].astype(str).tolist()

        if model_name in existing_models:
            return False

    new_row_df = pd.DataFrame([row])

    # This automatically adds any new columns from the new row.
    df = pd.concat([df, new_row_df], ignore_index=True, sort=False)

    df.to_csv(csv_path, index=False)

    return True

def get_model_name(pth_file: str | Path) -> Tuple[str, str]:
    pth_file = str(pth_file)
    if pth_file.endswith("YOLO_train.pth"):
        model_name = os.path.dirname(pth_file).split("/")[-2].replace("yolo_", "")
        model_tag = os.path.dirname(pth_file).split("/")[-1]
        return model_name, model_tag
    else:
        model_name = os.path.splitext(os.path.basename(pth_file))[0].replace("yolo_", "")
        return model_name, "best_map"

def load_yaml_cfg(yaml_path: str | Path):
    yaml_path = Path(yaml_path).expanduser().resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    cfg = OmegaConf.load(yaml_path)
    return cfg

def normalize_name(name: str) -> str:
    """
    Normalize names so:
    'deft_mountain' == 'deft-mountain-46' partially
    """
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def load_model_yaml(
    model_name: str,
    root_dir: str | Path,
    yaml_patterns: tuple[str, ...] = ("*.yaml", "*.yml"),
) -> dict[str, Any]:
    """
    Find a folder under root_dir whose normalized name contains the normalized model_name,
    then load the YAML file inside that folder.
    """

    root_dir = Path(root_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    normalized_model_name = normalize_name(model_name)

    matching_folders = sorted(
        folder
        for folder in root_dir.rglob("*")
        if folder.is_dir()
        and normalized_model_name in normalize_name(folder.name)
    )

    if not matching_folders:
        available_folders = [p.name for p in root_dir.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"No folder found where folder name contains '{model_name}' under: {root_dir}\n"
            f"Available folders: {available_folders}"
        )

    matched_folder = matching_folders[0]

    yaml_files = []
    for pattern in yaml_patterns:
        yaml_files.extend(matched_folder.glob(pattern))

    yaml_files = sorted(set(yaml_files))

    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML file found inside matched folder: {matched_folder}"
        )

    if len(yaml_files) > 1:
        raise ValueError(
            f"Multiple YAML files found inside {matched_folder}: "
            f"{[str(path) for path in yaml_files]}"
        )

    yaml_path = yaml_files[0]

    return load_yaml_cfg(yaml_path)

def model_exists_in_csv(
    csv_path: str | Path,
    model_name: str,
    model_name_col: str = "model_name",
) -> bool:
    """
    Check whether model_name already exists in a CSV file.

    Returns:
        True if the model exists.
        False if the CSV does not exist, the column does not exist, or the model is not found.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)

    if model_name_col not in df.columns:
        return False

    return model_name in df[model_name_col].astype(str).values

def find_model_pth_paths(
    root_model_folder: str | Path,
    model_name: str,
    tags: str | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """
    Find .pth files for a model.

    Logic:
    1. Look for a model subfolder whose name contains model_name.
       Example:
           model_name='deft-mountain'
           folder='deft_mountain'

    2. If model folder exists:
       - If tags are provided, only search inside subfolders whose names
         exactly match one of the tags after normalization.
       - If no tags are provided, return all .pth files inside the model folder.

    3. If no model subfolder exists:
       - Search for .pth files in the root folder whose filename contains model_name.
       - The tag is ignored in this fallback case because there are no tag subfolders.
    """

    root_model_folder = Path(root_model_folder).expanduser().resolve()

    if not root_model_folder.exists():
        raise FileNotFoundError(f"Root model folder does not exist: {root_model_folder}")

    if not root_model_folder.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_model_folder}")

    normalized_model_name = normalize_name(model_name)

    if tags is None:
        normalized_tags = None
    elif isinstance(tags, str):
        normalized_tags = {normalize_name(tags)}
    else:
        normalized_tags = {normalize_name(tag) for tag in tags}

    model_folders = sorted(
        folder
        for folder in root_model_folder.iterdir()
        if folder.is_dir()
        and normalized_model_name in normalize_name(folder.name)
    )

    if model_folders:
        matches: list[Path] = []

        for model_folder in model_folders:
            if normalized_tags is None:
                matches.extend(sorted(model_folder.rglob("*.pth")))
            else:
                tag_folders = sorted(
                    folder
                    for folder in model_folder.iterdir()
                    if folder.is_dir()
                    and normalize_name(folder.name) in normalized_tags
                )

                for tag_folder in tag_folders:
                    matches.extend(sorted(tag_folder.rglob("*.pth")))

        return sorted(set(matches))

    root_pth_files = sorted(root_model_folder.glob("*.pth"))

    root_matches = [
        pth_file
        for pth_file in root_pth_files
        if normalized_model_name in normalize_name(pth_file.stem)
    ]

    return root_matches

def handle_yaml(cfg_yaml: DictConfig) -> DictConfig:
    if 'nc' not in cfg_yaml.dataset:
        try:
            cfg_yaml.dataset.nc = len(cfg_yaml.dataset.names)
        except:
            OmegaConf.update(cfg_yaml, "dataset.nc", len(cfg_yaml.dataset.names), force_add=True)
    model_name = OmegaConf.select(cfg_yaml, "model.name")
    if model_name == "yolo":
        names = OmegaConf.select(cfg_yaml, "dataset.names")

        if names is not None:
            if isinstance(names, list):
                cfg_yaml.dataset.names = [
                    cls for cls in names
                    if cls != "__background__"
                ]
            try:
                cfg_yaml.dataset.nc = len(cfg_yaml.dataset.names)
            except :
                OmegaConf.update(cfg_yaml, "dataset.nc", len(cfg_yaml.dataset.names), force_add=True)
    print(cfg_yaml)
    return cfg_yaml

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_rank()

    # Works under torchrun even before init_process_group()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    return 0

def is_main_process() -> bool:
    return get_rank() == 0

def distributed_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def run_python_script_string_once(
    script: str,
    context: dict[str, Any] | None = None,
    marker_path: str = "dataset/.script_complete",
    lock_path: str = "dataset/.script.lock",
    timeout: int = 3600,
) -> dict[str, Any] | None:
    """
    Run a Python script once across torchrun/DDP workers.

    Rank 0 executes.
    Other ranks wait for marker_path.
    Works even before torch.distributed.init_process_group(),
    because it uses os.environ['RANK'] fallback.
    """
    marker = Path(marker_path)
    lock = Path(lock_path)
    marker.parent.mkdir(parents=True, exist_ok=True)

    rank = get_rank()
    result = None

    if marker.exists():
        return None

    if rank == 0:
        # Simple lock so accidental duplicate launch on same node is safer.
        lock_fd = None
        try:
            lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))

            if not marker.exists():
                result = run_python_script_string(script, context=context)

                # Write marker only after success.
                marker.write_text("ok\n", encoding="utf-8")

        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    else:
        start = time.time()
        while not marker.exists():
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for dataset setup marker: {marker}")
            time.sleep(1.0)

    distributed_barrier()
    return result

def run_python_script_string(
    script: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run Python code stored as a string.

    Args:
        script: Python code as a string.
        context: Variables to make available inside the script.

    Returns:
        The globals dictionary after execution.
    """

    exec_globals = {
        "__name__": "__main__",
    }

    if context:
        exec_globals.update(context)

    exec(script, exec_globals)

    return exec_globals

def boxes_to_xyxy(
    boxes,
    box_format: str,
    image_size=None,
    normalized: bool = False,
    clip: bool = False,
):
    """
    Convert boxes to pixel xyxy format.

    Args:
        boxes:
            Tensor-like of shape [N, 4].
        box_format:
            - "xyxy":   [x_min, y_min, x_max, y_max]
            - "xywh":   [x_min, y_min, width, height]
            - "cxcywh": [center_x, center_y, width, height]
        image_size:
            Optional (height, width). Required if normalized=True or clip=True.
        normalized:
            If True, input boxes are in [0, 1] and will be scaled to pixels.
        clip:
            If True, clip output xyxy boxes to image boundaries.

    Returns:
        Tensor of shape [N, 4] in pixel xyxy format.
    """
    boxes = torch.as_tensor(boxes)

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4).float()

    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)

    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes shape [N, 4], got {tuple(boxes.shape)}")

    boxes = boxes.clone().float()

    if normalized:
        if image_size is None:
            raise ValueError("image_size=(height, width) is required when normalized=True")

        height, width = image_size
        boxes[:, [0, 2]] *= float(width)
        boxes[:, [1, 3]] *= float(height)

    if box_format == "xyxy":
        out = boxes

    elif box_format == "xywh":
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        out = torch.stack(
            [x1, y1, x1 + w, y1 + h],
            dim=1,
        )

    elif box_format == "cxcywh":
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        out = torch.stack(
            [
                cx - 0.5 * w,
                cy - 0.5 * h,
                cx + 0.5 * w,
                cy + 0.5 * h,
            ],
            dim=1,
        )

    else:
        raise ValueError(
            f"Unknown box_format={box_format}. Expected one of: 'xyxy', 'xywh', 'cxcywh'."
        )

    if clip:
        if image_size is None:
            raise ValueError("image_size=(height, width) is required when clip=True")

        height, width = image_size
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0, float(width))
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0, float(height))

    return out

def boxes_to_encoder_space(
    boxes,
    box_format: str,
    image_size: tuple[int, int],
    normalized: bool,
):
    """
    Convert boxes to the coordinate scale expected by YOLODataEncoder.

    Important:
      - This does NOT change the box format.
      - It only changes normalized coordinates into pixel coordinates.
      - Output format remains whatever `box_format` is.

    Examples:
      xyxy normalized   [x1, y1, x2, y2] -> pixel xyxy
      xywh normalized   [x1, y1, w, h]   -> pixel xywh
      cxcywh normalized [cx, cy, w, h]   -> pixel cxcywh

    image_size = (height, width)
    """
    if box_format not in {"xyxy", "xywh", "cxcywh"}:
        raise ValueError(
            f"Unknown box_format={box_format}. "
            "Expected one of: 'xyxy', 'xywh', 'cxcywh'."
        )

    boxes = torch.as_tensor(boxes, dtype=torch.float).clone()

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)

    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes shape [N, 4], got {tuple(boxes.shape)}")

    if not normalized:
        return boxes

    height, width = image_size

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image_size={image_size}")

    # This scaling is valid for xyxy, xywh, and cxcywh.
    # Columns 0 and 2 are x/width-like values.
    # Columns 1 and 3 are y/height-like values.
    boxes[:, [0, 2]] *= float(width)
    boxes[:, [1, 3]] *= float(height)

    return boxes
