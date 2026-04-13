import cv2
import numpy as np
import torch
from torchinfo import summary

from .trainer import Trainer
from .src import Detector, DataEncoder
from . import utils

class Model:

    def __init__(self, cfg, load_model=True):

        # Initialize Logger
        self.logger = utils.get_logger('my_yolo')

        # MODEL Initialization
        if cfg.model.name == 'mlp':
            self.model = Detector(backbone_name=cfg.model.backbone_name,
                                  num_classes=len(cfg.dataset.names),
                                  fpn_channels=cfg.model.fpn_channels,
                                  num_anchors=cfg.model.num_anchors,)
            self.data_encoder = DataEncoder(input_size=cfg.model.image_size[:2], classes=cfg.dataset.names)
            try:
                if load_model:
                    self.model = utils.load_model(self.model, cfg.model.metadata.best_model_folder)
            except FileNotFoundError as e:
                self.logger.warning(f"Best model not found at {cfg.model.metadata.best_model_folder} - ERROR: {e}. Starting with a new model.")
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented yet.")
        self.logger.info(summary(self.model,
                                 input_size=(1,) + tuple(cfg.model.image_size)[::-1],
                                 row_settings=["var_names"]))

        self.height = cfg.model.image_size[0]
        self.width = cfg.model.image_size[1]
        self.transform = utils.get_inference_transforms(height=self.height, width=self.width)
        self.classes = cfg.dataset.names

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image_path, *args, **kwargs):

        orig_image = cv2.imread(image_path)[..., ::-1]
        orig_image_cpy = orig_image.copy()
        orig_image = orig_image.astype(np.int32)

        # Resize Image
        img = cv2.resize(orig_image_cpy, (self.width, self.height), cv2.INTER_CUBIC)
        img = np.ascontiguousarray(img)

        trans_img = self.transform(image=img)

        # Rescale ratio
        imH, imW = orig_image.shape[:2]
        IMG_SIZE_H, IMG_SIZE_W = img.shape[:2]

        ratio_h = imH / IMG_SIZE_H
        ratio_w = imW / IMG_SIZE_W

        results = self.predict(trans_img["image"])

        final_preds = []
        for pred in results['predictions']:
            pred['image_array'] = orig_image
            pred['final_boxes'] = pred['boxes']
            pred['final_boxes'][:, [0, 2]] *= ratio_w
            pred['final_boxes'][:, [1, 3]] *= ratio_h
            pred['final_boxes'] = pred['final_boxes'].cpu().numpy().tolist()
            pred['class_names'] = [self.classes[idx] for idx in pred['labels'].cpu().numpy().tolist()]
            final_preds.append(pred)

        return final_preds[0]

    def train(self, data, n_epochs=None, batch_size=None, data_dir='dataset'):

        # Initialize Directories
        cfg, current_version_name = utils.initialize_directory(self.cfg)

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
                nms_threshold=None,
                score_threshold=None,
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
                                                           nms_threshold=nms_threshold or self.cfg.model.nms_threshold,
                                                           score_threshold=score_threshold or self.cfg.model.score_threshold)
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
