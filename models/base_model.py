import cv2
import numpy as np
import torch
from torchinfo import summary

from .trainer import Trainer
from src import Detector, DataEncoder
import utils

class Model:

    def __init__(self, cfg, load_model=True):

        # Initialize Logger
        self.logger = utils.get_logger()

        # Initialize Directories
        cfg, current_version_name = utils.initialize_directory(cfg)

        # MODEL Initialization
        if cfg.model == 'baseline':
            self.model = Detector(backbone_name=cfg.backbone_name,
                                  num_classes=cfg.num_classes,
                                  fpn_channels=cfg.fpn_channels,
                                  num_anchors=cfg.num_anchors,)
            self.data_encoder = DataEncoder(input_size=cfg.image_size[:2], classes=cfg.classes)
            try:
                if load_model:
                    self.model = utils.load_model(cfg.model.best_model_folder)
            except FileNotFoundError as e:
                self.logger.warning(f"Best model not found at {cfg.model.best_model_folder}. Starting with a new model.")
        else:
            raise NotImplementedError(f"Model {cfg.model} not implemented yet.")

        self.logger.info(summary(self.model,
                                 input_size=(1,) + cfg.image_size[::-1],
                                 row_settings=["var_names"]))

        self.height = cfg.image_size[0]
        self.width = cfg.image_size[1]
        self.transforms = utils.get_augmentations(height=self.height, width=self.width)
        self.classes = cfg.classes

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image_path, *args, **kwargs):

        orig_image = cv2.imread(image_path)[..., ::-1]
        orig_image_cpy = orig_image.copy()
        orig_image = orig_image.astype(np.int32)

        # Resize Image
        img = cv2.resize(orig_image_cpy, (self.width, self.height), cv2.INTER_CUBIC)
        img = np.ascontiguousarray(img)

        trans_img = self.transforms[1](image=img)

        # Rescale ratio
        imH, imW = orig_image.shape[:2]
        IMG_SIZE_H, IMG_SIZE_W = img.shape[:2]

        ratio_h = imH / IMG_SIZE_H
        ratio_w = imW / IMG_SIZE_W

        results = self.predict(trans_img)

        final_preds = []
        for pred in results['predictions']:
            pred['final_boxes'] = pred['boxes']
            pred['final_boxes'][:, [0, 2]] *= ratio_w
            pred['final_boxes'][:, [1, 3]] *= ratio_h
            pred['final_boxes'] = pred['final_boxes'].squeeze(0).cpu().numpy()
            pred['class_names'] = [self.classes[idx] for idx in pred['labels'].squeeze(0).cpu().numpy()]
            final_preds.append(pred)

        return final_preds

    def train(self, data, n_epochs=None, batch_size=None, data_dir='dataset'):

        # Check if data is already downloaded and preprocessed, if not, do it.
        needs_download, url = utils.check_data_exists(data, data_dir)
        if needs_download:
            utils.download_and_unzip_zip(url, data_dir)

        # Initialize Trainer
        trainer = Trainer(self.model, data, self.cfg, logger=self.logger)
        # Start Training
        history = trainer.train(n_epochs=n_epochs, batch_size=batch_size)
        return history

    def predict(self,
                image_array,
                y_true=None,
                criterion=None,
                nms_threshold=0.5,
                score_threshold=0.5,
                device=None):

        loc_device = device or self.device
        image_batch = image_array.unsqueeze(0).to(loc_device)
        self.model.eval()
        self.model = self.model.to(loc_device)

        with torch.no_grad():
            pred_boxes, pred_labels = self.model(image_batch)

        if criterion and y_true:
            pred = (pred_boxes, pred_labels)
            loss = criterion(y_true, pred)

            loc_loss = loss["loc_loss"]
            cls_loss = loss["cls_loss"]
            total_loss = loss["total_loss"]

        predictions = []
        if self.data_encoder:
            for idx in range(image_batch.shape[0]):
                prediction_data = self.data_encoder.decode(pred_boxes[idx],
                                                           pred_labels[idx],
                                                           loc_device,
                                                           nms_threshold,
                                                           score_threshold)
                pred_bbox = prediction_data[:, :4]
                pred_conf = prediction_data[:, 4]
                pred_cls_id = prediction_data[:, 5]
                pred_dict = dict(
                    boxes=pred_bbox,
                    scores=pred_conf,
                    labels=pred_cls_id.int()
                )

                predictions.append(pred_dict)

        return {
            "predictions": predictions,
            "loc_loss": loc_loss.item() if criterion and y_true else None,
            "cls_loss": cls_loss.item() if criterion and y_true else None,
            "total_loss": total_loss.item() if criterion and y_true else None
        }
