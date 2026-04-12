import albumentations as A
from albumentations.augmentations import Normalize
from albumentations.pytorch.transforms import ToTensorV2


def get_augmentations(height=300, width=300):
    train_transforms = [
        A.Resize(height=height, width=width, interpolation=4),
        A.RandomResizedCrop(size=(height, width), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(p=0.9),

        # Photometric (keep moderate to avoid label drift)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Blur/Sharpen/Noise
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.UnsharpMask(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        A.GaussNoise(p=0.2),

        # Geometric crops that respect bboxes
        A.SafeRotate(limit=5, p=0.2),
        A.RandomCropFromBorders(crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, p=0.2),

        # Regularization
        A.CoarseDropout(p=0.5),

        # Final size
        A.LongestMaxSize(max_size=width, p=1.0),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, p=1.0),
        ###################

        Normalize(),
        ToTensorV2()
    ]

    valid_transforms = [
        A.Resize(height=height, width=width, interpolation=4),
        Normalize(),
        ToTensorV2()
    ]

    train_transforms = A.Compose(
        train_transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                                 min_visibility=0.01,  # drop boxes almost fully occluded
                                 min_area=4.0,  # drop tiny/degenerate boxes
                                 check_each_transform=True),
    )

    valid_transforms = A.Compose(
        valid_transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                                 min_visibility=0.01,  # drop boxes almost fully occluded
                                 min_area=4.0,  # drop tiny/degenerate boxes
                                 check_each_transform=True),
    )

    return train_transforms, valid_transforms