import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from torchinfo import summary

from .base_model import BaseModel
from .trainers.fpn_trainer import Trainer
from .src import Detector, DataEncoder
from . import utils

class FPNModel(BaseModel):

    start_epoch = 0

    def __init__(self, cfg, load_model=True, *args, **kwargs):

        # Initialize Logger
        self.logger = utils.get_logger('fpn')

        # MODEL Initialization
        self.model = Detector(backbone_name=cfg.model.backbone_name,
                              num_classes=len(cfg.dataset.names),
                              fpn_channels=cfg.model.fpn_channels,
                              num_anchors=cfg.model.num_anchors,)
        self.data_encoder = DataEncoder(input_size=cfg.model.image_size[:2], classes=cfg.dataset.names)
        try:
            if load_model:
                self.model, self.start_epoch = utils.load_model(self.model, cfg.model.metadata.best_model_folder, *args, **kwargs)
        except FileNotFoundError as e:
            self.logger.warning(f"Best model not found at {cfg.model.metadata.best_model_folder} - ERROR: {e}. Starting with a new model.")

        self.logger.info(summary(self.model,
                                 input_size=(1,) + tuple(cfg.model.image_size)[::-1],
                                 row_settings=["var_names"]))

        self.height = cfg.model.image_size[0]
        self.width = cfg.model.image_size[1]
        self.transform = utils.get_inference_transforms(height=self.height, width=self.width)
        self.classes = cfg.dataset.names
        self.logger.info(f"Classes: {self.classes}. On image size: {self.height}x{self.width}")

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

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

    def train(self, data, n_epochs=None, batch_size=None, data_dir='dataset', coco_eval=True):

        # Check if data is already downloaded and preprocessed, if not, do it.
        needs_download, url, data_yaml = utils.check_data_exists(data, data_dir)
        self.logger.info(f"Data check - needs download: {needs_download}, url: {url}, data_yaml: {data_yaml}")
        if needs_download:
            self.logger.info(f"Downloading data from {url}")
            utils.download_and_unzip_zip(url, data_dir)

        # Initialize Trainer
        trainer = Trainer(self, data_yaml, self.cfg, logger=self.logger, close_when_done=(not coco_eval))
        # Start Training
        history = trainer.train(n_epochs=n_epochs, batch_size=batch_size, start_epoch=self.start_epoch)
        coco_eval_results = None
        if coco_eval:
            coco_eval_results = self.evaluate(self.cfg.dataset.val.replace('..', data_dir))
            if trainer.wandb:
                trainer.wandb.log({'coco_eval_results': coco_eval_results})
                trainer.wandb.finish()
        return {
            "history": history,
            "coco_eval_results": coco_eval_results
        }

    def predict(self,
                image_array,
                y_true=None,
                criterion=None,
                nms_threshold=None,
                score_threshold=None,
                device=None):

        loc_device = device or self.device
        if image_array.ndim == 3:
            image_array = image_array.unsqueeze(0)
        image_batch = image_array.to(loc_device)
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

    def evaluate(self, data_folder, batch_size=64):

        data = utils.get_val_yaml_file_path(data_folder)
        data_class = utils.DataSetup(self.cfg, data)
        loader = data_class.get_one_loader(batch_size)

        iterator = tqdm(loader, dynamic_ncols=True)

        preds = []
        targets = []
        for i, batch_sample in enumerate(iterator):

            image_batch = torch.stack(batch_sample[0]).to(self.device)
            box_targets = torch.stack(batch_sample[3]).to(self.device)
            cls_targets = torch.stack(batch_sample[4]).to(self.device)

            y_true = (box_targets, cls_targets)
            nms_threshold = self.cfg.model.nms_threshold if self.data_encoder else None
            score_threshold = self.cfg.model.score_threshold if self.data_encoder else None
            results = self.predict(image_batch,
                                   y_true=y_true,
                                   device=self.device,
                                   nms_threshold=nms_threshold,
                                   score_threshold=score_threshold)

            predictions = results["predictions"]
            preds.extend(predictions)

            # Prepare targets and predictions for evaluations.
            for idx, (box_raw, label_raw, original_size) in enumerate(zip(batch_sample[1], batch_sample[2], batch_sample[5])):
                boxes_raw_per_image = box_raw.to(self.device)
                labels_raw_per_image = label_raw.to(self.device)

                target_dict = dict(
                    boxes=boxes_raw_per_image,
                    labels=labels_raw_per_image,
                    img_size = original_size
                )

                targets.append(target_dict)

            status = f"[Validation][{i+1}]"

            iterator.set_description(status)

        return utils.coco_eval(targets, preds)
