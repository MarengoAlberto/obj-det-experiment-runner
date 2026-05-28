import torch
from tqdm.auto import tqdm
import numpy as np

from .fpn_trainer import FPNTrainer
from ..utils import boxes_to_xyxy

class YOLOTrainer(FPNTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train_step(self, *args, **kwargs):
        self.model.train()

        iterator = tqdm(self.train_loader, dynamic_ncols=True)

        obj_loss_avg = []
        cls_loss_avg = []
        loc_loss_avg = []
        total_loss_avg = []

        for i, batch_sample in enumerate(iterator):
            self.optimizer.zero_grad()
            image_batch = torch.stack(batch_sample[0]).to(self.device)
            targets = torch.stack(batch_sample[3]).to(self.device)
            # PREDICTION
            pred = self.model(image_batch)
            # LOSS CALCULATION
            loss_results = self.criterion(targets, pred)
            loss = loss_results["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            optimizer_lr = self.optimizer.param_groups[0]["lr"]

            obj_loss = loss_results["obj_loss"].item()
            loc_loss = loss_results["loc_loss"].item()
            cls_loss = loss_results["cls_loss"].item()
            total_loss = loss.item()
            if self.use_ddp:
                obj_loss = self.reduce_mean(obj_loss, self.device, self.world_size)
                loc_loss = self.reduce_mean(loc_loss, self.device, self.world_size)
                cls_loss = self.reduce_mean(cls_loss, self.device, self.world_size)
                total_loss = self.reduce_mean(total_loss, self.device, self.world_size)

            obj_loss_avg.append(obj_loss)
            cls_loss_avg.append(cls_loss)
            loc_loss_avg.append(loc_loss)
            total_loss_avg.append(total_loss)

            status = f"[Train][{i + 1}] Total Loss: {np.mean(total_loss_avg):.4f}, "
            status += f"Obj Loss: {np.mean(obj_loss_avg):.4f}, "
            status += f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "
            status += f"LR: {optimizer_lr:.3f}"

            iterator.set_description(status)

        return {"obj_loss": np.mean(obj_loss_avg),
                "loc_loss": np.mean(loc_loss_avg),
                "cls_loss": np.mean(cls_loss_avg),
                "total_loss": np.mean(total_loss_avg)}

    def _validation_step(self, *args, **kwargs):

        self.model.eval()

        iterator = tqdm(self.val_loader, dynamic_ncols=True)

        obj_loss_avg = []
        cls_loss_avg = []
        loc_loss_avg = []
        total_loss_avg = []

        self.metric.reset()

        for i, batch_sample in enumerate(iterator):

            image_batch = torch.stack(batch_sample[0]).to(self.device)
            targets = torch.stack(batch_sample[3]).to(self.device)

            nms_threshold = self.cfg.model.nms_threshold if self.data_encoder else None
            score_threshold = self.cfg.model.score_threshold if self.data_encoder else None
            max_dets = self.cfg.model.max_detections if self.data_encoder else None
            results = self.wrapper.predict(image_batch,
                                           criterion=self.criterion,
                                           y_true=targets,
                                           device=self.device,
                                           nms_threshold=nms_threshold,
                                           score_threshold=score_threshold,
                                           max_dets=max_dets)

            predictions = results["predictions"]
            obj_loss = results["obj_loss"]
            loc_loss = results["loc_loss"]
            cls_loss = results["cls_loss"]
            total_loss = results["total_loss"]

            if self.use_ddp:
                obj_loss = self.reduce_mean(obj_loss, self.device, self.world_size)
                loc_loss = self.reduce_mean(loc_loss, self.device, self.world_size)
                cls_loss = self.reduce_mean(cls_loss, self.device, self.world_size)
                total_loss = self.reduce_mean(total_loss, self.device, self.world_size)

            obj_loss_avg.append(obj_loss)
            cls_loss_avg.append(cls_loss)
            loc_loss_avg.append(loc_loss)
            total_loss_avg.append(total_loss)

            # Prepare targets and predictions for evaluations.
            targets = []
            for idx, (box_raw, label_raw) in enumerate(zip(batch_sample[1], batch_sample[2])):
                boxes_raw_per_image = box_raw.to(self.device)
                labels_raw_per_image = label_raw.to(self.device)

                target_dict = dict(
                    boxes=boxes_to_xyxy(boxes_raw_per_image,
                                        box_format=self.cfg.dataset.metadata.box_format,
                                        image_size=(self.wrapper.height, self.wrapper.width),
                                        normalized=self.cfg.dataset.metadata.box_normalized,
                                        clip=True),
                    labels=labels_raw_per_image
                )

                targets.append(target_dict)

            self.metric.update(predictions, targets)

            status = f"[Validation][{i + 1}] Total Loss: {np.mean(total_loss_avg):.4f}, "
            status += f"Obj Loss: {np.mean(obj_loss_avg):.4f}, "
            status += f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "

            iterator.set_description(status)

        metrics_dict = self.metric.compute()

        map_50 = float(metrics_dict["map_50"])
        status += f"val_mAP@50: {map_50:.3f}"

        iterator.set_description(status)

        output = {"obj_loss": np.mean(obj_loss_avg),
                  "loc_loss": np.mean(loc_loss_avg),
                  "cls_loss": np.mean(cls_loss_avg),
                  "total_loss": np.mean(total_loss_avg),
                  "metrics": metrics_dict}
        return output
