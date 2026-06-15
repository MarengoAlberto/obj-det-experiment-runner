import cv2
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from ..src import DataEncoder
from .utils import is_main_process, boxes_to_encoder_space
from .augmentations import get_augmentations

class DataSetup:

    def __init__(self, cfg, data, data_encoder, use_ddp=False, rank=0, world_size=1):
        self.cfg = cfg
        self.data = data
        self.data_encoder = data_encoder
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.image_size = cfg.model.image_size
        self.height = self.image_size[0]
        self.width = self.image_size[1]
        self.classes = cfg.dataset.names

    def get_loaders(self, _batch_size):

        batch_size = _batch_size if _batch_size else self.cfg.experiment.train.batch_size
        train_path = self.data.full_train_path
        val_path = self.data.full_val_path
        num_workers = self.cfg.experiment.train.num_workers

        train_augmentations, valid_augmentations = get_augmentations(height=self.height, width=self.width,
                                                                     box_format=self.cfg.dataset.metadata.box_format)

        # Create custom datasets.
        train_dataset = self.get_dataset(train_path, train_augmentations, train=True)

        valid_dataset = self.get_dataset(val_path, valid_augmentations, train=False)

        if self.use_ddp:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
            )
            val_sampler = DistributedSampler(
                valid_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
            )
        else:
            train_sampler = None
            val_sampler = None

        # Create Custom DataLoaders

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=train_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, valid_loader, train_sampler, val_sampler

    def get_one_loader(self, _batch_size, split_name='val', drop_last=False):

        batch_size = _batch_size if _batch_size else self.cfg.experiment.train.batch_size

        if split_name == 'train':
            path = self.data.full_train_path
        elif split_name == 'val':
            path = self.data.full_val_path
        elif split_name == 'test':
            path = self.data.full_test_path
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        num_workers = self.cfg.experiment.train.num_workers

        train_augmentation, valid_augmentations = get_augmentations(height=self.height, width=self.width,
                                                                     box_format=self.cfg.dataset.metadata.box_format)

        # Create custom dataset.
        if split_name == 'train':
            dataset = self.get_dataset(path, train_augmentation, train=True)
        else:
            dataset = self.get_dataset(path, valid_augmentations, train=False)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )

        return loader

    def get_dataset(self, path, trasform_function, train=True):

        if 'is_standard_folder_structure' in self.cfg.dataset.metadata:
            is_standard_folder_structure = self.cfg.dataset.metadata.is_standard_folder_structure
        else:
            is_standard_folder_structure = True
        if self.cfg.model.name == "fpn":
            dataset = FPNDataset(
                data_path=path,
                data_encoder=self.data_encoder,
                transform=trasform_function,
                classes=self.classes,
                input_size=self.image_size,
                is_train=train,
                debug=self.cfg.experiment.train.debug,
                is_standard_folder_structure=is_standard_folder_structure,
                box_normalized=self.cfg.dataset.metadata.box_normalized,
            )
        elif self.cfg.model.name == "yolo":
            dataset = YoloDataset(
                data_path=path,
                data_encoder=self.data_encoder,
                transform=trasform_function,
                classes=self.classes,
                input_size=self.image_size,
                is_train=train,
                debug=self.cfg.experiment.train.debug,
                is_standard_folder_structure=is_standard_folder_structure,
                box_normalized=self.cfg.dataset.metadata.box_normalized,
            )
        else:
             raise ValueError(f"Unknown model_type: {self.cfg.model.name}")
        return dataset

class FPNDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_encoder,
        classes,
        transform=None,
        is_train=True,
        input_size=(300, 300, 3),
        debug=False,
        is_standard_folder_structure=True,
        box_normalized=False,
    ):
        self.data_path = os.path.expanduser(data_path)
        self.classes = classes
        self.transforms = transform
        self.input_size = input_size
        self.is_train = is_train
        self.encoder = data_encoder
        self.box_normalized = box_normalized

        self.image_paths, self.boxes, self.labels, self.num_samples = load_groundtruths(data_path,
                                                                                        train=is_train,
                                                                                        shuffle=is_train,
                                                                                        debug=debug,
                                                                                        is_standard_folder_structure=is_standard_folder_structure)
    def __len__(self):
        # Get size of the Dataset.
        return self.num_samples

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        indexed_boxes = self.boxes[idx]
        indexed_labels = self.labels[idx]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        height = img.shape[0]
        width = img.shape[1]

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)

        else: # Mandatory transforms to be applied.

            common_transforms = A.Compose(
                                [A.Resize(
                                        height=self.input_size[0], width=self.input_size[1],
                                        interpolation=4
                                       ),
                                ToTensorV2()],
                                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
                                )

            transformed = common_transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)


        transformed_img    = transformed["image"]
        transformed_boxes  = transformed["bboxes"]
        transformed_labels = transformed["category_ids"]

        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float)
        transformed_labels = torch.tensor(transformed_labels, dtype=torch.int)

        # ===========================================================
        # Generate Encoded bounding boxes and labels
        # ===========================================================

        boxes_for_encoder = boxes_to_encoder_space(
            transformed_boxes,
            box_format=self.encoder.box_format,
            image_size=self.input_size[:2],  # after A.Resize, model image size
            normalized=self.box_normalized,  # add this attribute from cfg.dataset.metadata.box_normalized
        )

        loc_target, cls_target = self.encoder.encode(boxes_for_encoder, transformed_labels)

        original_size = torch.tensor((height, width), dtype=torch.int)

        return transformed_img, transformed_boxes, transformed_labels, loc_target, cls_target, original_size

    def collate_fn(self, batch):
        return list(zip(*batch))

def list_files_in_directory(directory_path):
    try:
        entries = os.listdir(directory_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return files
    except FileNotFoundError:
        if is_main_process():
            print(f"Error: Directory '{directory_path}' not found.")
        return []


def load_groundtruths(data_path, train=True, shuffle=True, debug=False, is_standard_folder_structure=True):
    image_paths = []
    boxes = []
    labels = []

    file_names = list_files_in_directory(data_path)
    num_samples = len(file_names)
    for image_name in file_names:
        image_id, _ = os.path.splitext(os.path.basename(image_name))
        if is_standard_folder_structure:
            filepath = os.path.join(data_path.replace('images', 'labels'), image_id + '.txt')
        else:
            filepath = os.path.join(data_path, 'Label', image_id + '.txt')

        with open(filepath) as f:
            lines = f.readlines()

        image_paths.append(os.path.join(data_path, image_name))
        box = []
        label = []

        for line in lines:
            splited = line.strip().split()
            bbox = splited[-4:]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            if is_standard_folder_structure:
                class_label = int(splited[0])
            else:
                class_label = int(1)
            box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            label.append(class_label)
        boxes.append(box)
        labels.append(label)
    if is_main_process():
        print(f"Total {num_samples} images and {sum(len(image_boxes) for image_boxes in boxes)} boxes and {sum(len(image_label) for image_label in labels)} labels loaded from: {os.path.relpath(data_path, os.getcwd())}")

    # Shuffle or Sort
    if shuffle:
        temp = list(zip(image_paths, boxes, labels))
        random.shuffle(temp)
        image_paths, boxes, labels = zip(*temp)
    else:
        image_paths, boxes, labels = zip(*sorted(zip(image_paths, boxes, labels)))

    image_paths = list(image_paths)
    boxes = list(boxes)
    labels = list(labels)

    if debug:
        if train:
            num_samples = 100
        else:
            num_samples = 10
        if is_main_process():
            print(f"Debug mode enabled: Only using {num_samples} samples for set: {'train' if train else 'validation'}")
        image_paths = image_paths[:num_samples]
        boxes = boxes[:num_samples]
        labels = labels[:num_samples]

    return image_paths, boxes, labels, num_samples

def filter_valid_boxes(boxes, labels=None, box_format="xyxy", min_size=1e-6):
    """
    Filter invalid boxes while respecting box format.

    box_format:
      - "xyxy":   [x_min, y_min, x_max, y_max]
      - "xywh":   [x_min, y_min, width, height]
      - "cxcywh": [center_x, center_y, width, height]
    """
    boxes_t = torch.as_tensor(boxes)

    if boxes_t.numel() == 0:
        empty_boxes = []
        empty_labels = [] if labels is not None else None
        return (empty_boxes, empty_labels) if labels is not None else empty_boxes

    if boxes_t.ndim == 1:
        boxes_t = boxes_t.unsqueeze(0)

    if boxes_t.shape[1] < 4:
        raise ValueError(f"Expected boxes with at least 4 values, got shape {boxes_t.shape}")

    if box_format == "xyxy":
        widths = boxes_t[:, 2] - boxes_t[:, 0]
        heights = boxes_t[:, 3] - boxes_t[:, 1]

    elif box_format in ("xywh", "cxcywh"):
        widths = boxes_t[:, 2]
        heights = boxes_t[:, 3]

    else:
        raise ValueError(
            f"Unknown box_format={box_format}. Expected 'xyxy', 'xywh', or 'cxcywh'."
        )

    valid = (widths > min_size) & (heights > min_size)

    filtered_boxes = boxes_t[valid].tolist()

    if labels is not None:
        labels_t = torch.as_tensor(labels)
        if labels_t.ndim == 0:
            labels_t = labels_t.unsqueeze(0)

        filtered_labels = labels_t[valid].tolist()
        return filtered_boxes, filtered_labels

    return filtered_boxes

class YoloDataset(FPNDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        indexed_boxes = self.boxes[idx]
        indexed_labels = self.labels[idx]
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        height = img.shape[0]
        width = img.shape[1]

        if self.transforms:
            indexed_boxes, indexed_labels = filter_valid_boxes(indexed_boxes, indexed_labels, self.encoder.box_format)
            transformed = self.transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)

        else:  # Mandatory transforms to be applied.

            common_transforms = A.Compose(
                [A.Resize(
                    height=self.input_size[0], width=self.input_size[1],
                    interpolation=4
                ),
                    ToTensorV2()],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
            )

            transformed = common_transforms(image=img, bboxes=indexed_boxes, category_ids=indexed_labels)

        transformed_img = transformed["image"]
        transformed_boxes = transformed["bboxes"]
        transformed_labels = transformed["category_ids"]

        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float)
        transformed_labels = torch.tensor(transformed_labels, dtype=torch.int)

        boxes_for_encoder = boxes_to_encoder_space(
            transformed_boxes,
            box_format=self.encoder.box_format,
            image_size=self.input_size[:2],  # after A.Resize, model image size
            normalized=self.box_normalized,  # add this attribute from cfg.dataset.metadata.box_normalized
        )

        if len(self.classes) == 1:
            transformed_labels = transformed_labels - 1
            encoded = self.encoder.encode(boxes_for_encoder, transformed_labels, background_id=-1)
        else:
            encoded = self.encoder.encode(boxes_for_encoder, transformed_labels)

        original_size = torch.tensor((height, width), dtype=torch.int)

        return transformed_img, transformed_boxes, transformed_labels, encoded, original_size
