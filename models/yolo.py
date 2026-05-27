import torch
from tqdm.auto import tqdm

from . import utils
from . fpn import FPNModel

class YOLO(FPNModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self,
                image_array,
                y_true=None,
                criterion=None,
                nms_threshold=None,
                score_threshold=None,
                max_dets=None,
                device=None):

        loc_device = device or self.device
        if image_array.ndim == 3:
            image_array = image_array.unsqueeze(0)
        image_batch = image_array.to(loc_device)
        self.model.eval()
        self.model = self.model.to(loc_device)

        with torch.no_grad():
            logits = self.model(image_batch)

        if criterion and y_true is not None:
            loss_results = criterion(y_true, logits)

            obj_loss = loss_results["obj_loss"].item()
            loc_loss = loss_results["loc_loss"].item()
            cls_loss = loss_results["cls_loss"].item()
            total_loss = loss_results["total_loss"]

        predictions = []
        if self.data_encoder:
            for idx in range(image_batch.shape[0]):
                prediction_data = self.data_encoder.decode(logits[idx],
                                                           nms_threshold=nms_threshold or self.cfg.model.nms_threshold,
                                                           score_threshold=score_threshold or self.cfg.model.score_threshold,
                                                           max_dets=max_dets or self.cfg.model.max_detections)
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
            "obj_loss": obj_loss if criterion and y_true is not None else None,
            "loc_loss": loc_loss if criterion and y_true is not None else None,
            "cls_loss": cls_loss if criterion and y_true is not None else None,
            "total_loss": total_loss if criterion and y_true is not None else None
        }

    def evaluate(self, split_name, batch_size=64):
        if self.data_yaml is None:
            raise ValueError("Data yaml is not loaded. Please load data first.")

        data_class = utils.DataSetup(self.cfg, self.data_yaml, self.data_encoder)
        loader = data_class.get_one_loader(batch_size, split_name=split_name)

        iterator = tqdm(loader, dynamic_ncols=True)

        preds = []
        true_labels = []
        for i, batch_sample in enumerate(iterator):

            image_batch = torch.stack(batch_sample[0]).to(self.device)
            targets = torch.stack(batch_sample[3]).to(self.device)

            nms_threshold = self.cfg.model.nms_threshold if self.data_encoder else None
            score_threshold = self.cfg.model.score_threshold if self.data_encoder else None
            results = self.predict(image_batch,
                                   y_true=targets,
                                   device=self.device,
                                   nms_threshold=nms_threshold,
                                   score_threshold=score_threshold)

            predictions = results["predictions"]
            preds.extend(predictions)

            # Prepare targets and predictions for evaluations.
            for idx, (box_raw, label_raw, original_size) in enumerate(zip(batch_sample[1], batch_sample[2], batch_sample[4])):
                boxes_raw_per_image = box_raw.to(self.device)
                labels_raw_per_image = label_raw.to(self.device)

                target_dict = dict(
                    boxes=utils.boxes_to_xyxy(boxes_raw_per_image, self.cfg.dataset.metadata.box_format),
                    labels=labels_raw_per_image,
                    img_size = original_size
                )

                true_labels.append(target_dict)

            status = f"[Validation][{i+1}]"

            iterator.set_description(status)
        if self.cfg.model.name == "yolo":
            cat_id = int(self.cfg.dataset.nc) - 1
            cat_name = self.cfg.dataset.names[cat_id]
            return utils.coco_eval(true_labels, preds, cat_id=cat_id, cat_name=cat_name)
        return utils.coco_eval(true_labels, preds)
