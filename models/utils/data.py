import cv2
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from ..src import DataEncoder
from .augmentations import get_augmentations

class DataSetup:

    def __init__(self, cfg, data, use_ddp=False, rank=0, world_size=1):
        self.cfg = cfg
        self.data = data
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.image_size = cfg.model.image_size
        self.height = self.image_size[0]
        self.width = self.image_size[1]
        self.classes = cfg.dataset.names

    def get_loaders(self, _batch_size):

        batch_size = _batch_size if _batch_size else self.cfg.experiment.train.batch_size
        train_path = self.cfg.dataset.full_train_path
        val_path = self.cfg.dataset.full_val_path
        num_workers = self.cfg.experiment.train.num_workers

        train_augmentations, valid_augmentations = get_augmentations(height=self.height, width=self.width)

        # Create custom datasets.
        train_dataset = PlateDataset(
            data_path=train_path,
            transform=train_augmentations,
            classes=self.classes,
            input_size=self.image_size,
            is_train=True,
            debug=self.cfg.experiment.train.debug,
        )

        valid_dataset = PlateDataset(
            data_path=val_path,
            transform=valid_augmentations,
            classes=self.classes,
            input_size=self.image_size,
            is_train=False,
            debug=self.cfg.experiment.train.debug,
        )

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

class PlateDataset(Dataset):
    def __init__(
        self,
        data_path,
        classes,
        transform=None,
        is_train=True,
        input_size=(300, 300, 3),
        debug=False
    ):
        self.data_path = os.path.expanduser(data_path)
        self.classes = classes
        self.transforms = transform
        self.input_size = input_size
        self.is_train = is_train
        self.encoder = DataEncoder(self.input_size[:2], self.classes)

        self.image_paths, self.boxes, self.labels, self.num_samples = load_groundtruths(data_path, train=is_train, shuffle=is_train, debug=debug)

    def __len__(self):
        # Get size of the Dataset.
        return self.num_samples

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        indexed_boxes = self.boxes[idx]
        indexed_labels = self.labels[idx]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

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

        loc_target, cls_target = self.encoder.encode(transformed_boxes, transformed_labels)

        return transformed_img, transformed_boxes, transformed_labels, loc_target, cls_target

    def collate_fn(self, batch):
        return list(zip(*batch))

def list_files_in_directory(directory_path):
    try:
        entries = os.listdir(directory_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return files
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
        return []

def load_groundtruths(data_path, train=True, shuffle=True, debug=False):
    image_paths = []
    boxes = []
    labels = []

    file_names = list_files_in_directory(data_path)
    num_samples = len(file_names)
    for image_name in file_names:
        image_id, _ = os.path.splitext(os.path.basename(image_name))
        filepath = os.path.join(data_path, 'label', image_id+'.txt')

        with open(filepath) as f:
            lines = f.readlines()

        image_paths.append(os.path.join(data_path, image_name))
        box = []
        label = []

        for line in lines:
            splited = line.strip().split()[-4:]
            xmin = splited[0]
            ymin = splited[1]
            xmax = splited[2]
            ymax = splited[3]

            class_label = int(1)
            box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            label.append(class_label)
        boxes.append(box)
        labels.append(label)

    print(f"Total {num_samples} images and {len(boxes)} boxes loaded from: {os.path.relpath(data_path, os.getcwd())}")

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
            mum_samples = 10
        print(f"Debug mode enabled: Only using {num_samples} samples for set: {'train' if train else 'validation'}")
        image_paths = image_paths[:mum_samples]
        boxes = boxes[:mum_samples]
        labels = labels[:mum_samples]

    return image_paths, boxes, labels, num_samples