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

        if not self.optimized_cpu_inference:
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
        drop_last = True if (self.optimized_cpu_inference and not utils.is_cuda_available()) else False
        loader = data_class.get_one_loader(batch_size, split_name=split_name, drop_last=drop_last)

        iterator = tqdm(loader, dynamic_ncols=True)

        if (self.optimized_cpu_inference and not utils.is_cuda_available() and not self.compiled_model):
            self.model, self.compiled_model = utils.compile_model(self.full_model_path,
                                                                  self.model,
                                                                  (self.height, self.width),
                                                                  batch_size,)

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

                img_size = (self.height, self.width)
                boxes_xyxy = utils.boxes_to_xyxy(boxes_raw_per_image,
                                              box_format=self.cfg.dataset.metadata.box_format,
                                              image_size=img_size,
                                              normalized=self.cfg.dataset.metadata.box_normalized,
                                              clip=True)

                target_dict = dict(
                    boxes=boxes_xyxy,
                    labels=labels_raw_per_image,
                    img_size=img_size
                )

                if self.cfg.experiment.train.debug:
                    if i == 0 and idx < 5:
                        num_classes = len(self.cfg.dataset.names)

                        boxes_cpu = boxes_xyxy.detach().cpu()
                        labels_cpu = labels_raw_per_image.detach().cpu().long()

                        H, W = img_size  # your img_size is currently (height, width)

                        print("\n" + "=" * 80)
                        print(f"DEBUG TARGET image={idx}")
                        print("=" * 80)

                        print(f"box_format(raw): {self.cfg.dataset.metadata.box_format}")
                        print(f"model image size: H={H}, W={W}")
                        print(f"num boxes: {boxes_cpu.shape[0]}")
                        print(f"num classes: {num_classes}")

                        if boxes_cpu.numel() == 0:
                            print("No boxes after transform/conversion.")
                            continue

                        wh = boxes_cpu[:, 2:] - boxes_cpu[:, :2]
                        areas = wh[:, 0] * wh[:, 1]

                        valid_size = (wh[:, 0] > 0) & (wh[:, 1] > 0)
                        inside = (
                                (boxes_cpu[:, 0] >= 0) &
                                (boxes_cpu[:, 1] >= 0) &
                                (boxes_cpu[:, 2] <= W) &
                                (boxes_cpu[:, 3] <= H)
                        )
                        valid_labels = (
                                (labels_cpu >= 0) &
                                (labels_cpu < num_classes)
                        )

                        print(f"labels unique: {sorted(labels_cpu.unique().tolist())}")
                        print(f"labels min/max: {labels_cpu.min().item()} / {labels_cpu.max().item()}")
                        print(f"valid labels: {valid_labels.sum().item()} / {len(labels_cpu)}")

                        print(f"valid box size: {valid_size.sum().item()} / {len(boxes_cpu)}")
                        print(f"inside image: {inside.sum().item()} / {len(boxes_cpu)}")

                        print(
                            "box xyxy min/max:",
                            round(boxes_cpu.min().item(), 3),
                            round(boxes_cpu.max().item(), 3),
                        )

                        print(
                            "width min/mean/max:",
                            round(wh[:, 0].min().item(), 3),
                            round(wh[:, 0].mean().item(), 3),
                            round(wh[:, 0].max().item(), 3),
                        )

                        print(
                            "height min/mean/max:",
                            round(wh[:, 1].min().item(), 3),
                            round(wh[:, 1].mean().item(), 3),
                            round(wh[:, 1].max().item(), 3),
                        )

                        print(
                            "area min/mean/max:",
                            round(areas.min().item(), 3),
                            round(areas.mean().item(), 3),
                            round(areas.max().item(), 3),
                        )

                        small = areas < 32 ** 2
                        medium = (areas >= 32 ** 2) & (areas < 96 ** 2)
                        large = areas >= 96 ** 2

                        print(
                            f"COCO area buckets: "
                            f"small={small.sum().item()}, "
                            f"medium={medium.sum().item()}, "
                            f"large={large.sum().item()}"
                        )

                        print("\nraw boxes sample:")
                        print(box_raw[:5])

                        print("\nconverted pixel xyxy sample:")
                        print(boxes_cpu[:5])

                        print("\nlabels sample:")
                        print(labels_cpu[:20])

                        bad = ~(valid_size & inside & valid_labels)
                        if bad.any():
                            bad_idx = torch.where(bad)[0][:10]
                            print("\nBAD TARGET EXAMPLES:")
                            for bi in bad_idx:
                                print(
                                    f"idx={bi.item()} "
                                    f"box={boxes_cpu[bi].tolist()} "
                                    f"wh={wh[bi].tolist()} "
                                    f"label={labels_cpu[bi].item()} "
                                    f"valid_size={valid_size[bi].item()} "
                                    f"inside={inside[bi].item()} "
                                    f"valid_label={valid_labels[bi].item()}"
                                )

                true_labels.append(target_dict)

            status = f"[Validation][{i+1}]"

            iterator.set_description(status)

        return utils.coco_eval(true_labels, preds, self.cfg.dataset.names)
