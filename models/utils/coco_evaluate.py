from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [
        float(x1),
        float(y1),
        float(max(0.0, x2 - x1)),
        float(max(0.0, y2 - y1)),
    ]


def build_coco_from_simple(gt_simple, class_names):
    """
    gt_simple item format:
        {
            "img_size": (width, height),
            "boxes": [[x1, y1, x2, y2], ...],
            "labels": [0, 1, ..., num_classes-1]
        }

    class_names:
        list of foreground class names.
        Example VisDrone 10-class list.
    """
    images = []
    annos = []
    ann_id = 1

    categories = [
        {"id": i + 1, "name": name}
        for i, name in enumerate(class_names)
    ]

    num_classes = len(class_names)

    for image_id, sample in enumerate(gt_simple, start=1):
        w, h = sample["img_size"]
        images.append(
            {
                "id": image_id,
                "width": int(w),
                "height": int(h),
            }
        )

        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])

        if len(boxes) != len(labels):
            raise ValueError(
                f"GT boxes/labels length mismatch for image {image_id}: "
                f"{len(boxes)} boxes vs {len(labels)} labels"
            )

        for box, lab in zip(boxes, labels):
            lab = int(lab)

            # Skip invalid/ignored labels.
            # Model labels should be 0..num_classes-1.
            if lab < 0 or lab >= num_classes:
                continue

            x1, y1, x2, y2 = map(float, box)
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)

            if w_box <= 0 or h_box <= 0:
                continue

            annos.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    # COCO category ids are 1..num_classes.
                    "category_id": lab + 1,
                    "bbox": [x1, y1, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annos,
        "categories": categories,
        "info": {"description": "COCO dataset eval"},
        "licenses": [],
    }


def convert_preds_xyxy_to_coco(pred_simple, class_names, max_dets_per_image=None):
    """
    pred_simple item format:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "scores": [score, ...],
            "labels": [0, 1, ..., num_classes-1]
        }
    """
    coco_dets = []
    num_classes = len(class_names)

    for image_id, p in enumerate(pred_simple, start=1):
        boxes = p.get("boxes", [])
        scores = p.get("scores", [])
        labels = p.get("labels", [])

        if len(boxes) != len(scores) or len(boxes) != len(labels):
            raise ValueError(
                f"Prediction boxes/scores/labels length mismatch for image {image_id}: "
                f"{len(boxes)} boxes, {len(scores)} scores, {len(labels)} labels"
            )

        rows = []
        for box, sc, lab in zip(boxes, scores, labels):
            lab = int(lab)

            # Skip invalid labels.
            if lab < 0 or lab >= num_classes:
                continue

            rows.append(
                {
                    "category_id": lab + 1,  # model 0-based -> COCO 1-based
                    "image_id": image_id,
                    "bbox": xyxy_to_xywh(box),
                    "score": float(sc),
                }
            )

        # Optional global max detections per image before COCO eval.
        if max_dets_per_image is not None and len(rows) > max_dets_per_image:
            rows = sorted(rows, key=lambda x: x["score"], reverse=True)
            rows = rows[:max_dets_per_image]

        coco_dets.extend(rows)

    return coco_dets


def coco_eval(
    gt,
    preds,
    class_names,
    max_dets=(1, 10, 100),
    iou_type="bbox",
    max_dets_per_image=None,
):
    gt_coco_dict = build_coco_from_simple(gt, class_names)
    det_list = convert_preds_xyxy_to_coco(
        preds,
        class_names,
        max_dets_per_image=max_dets_per_image,
    )

    coco_gt = COCO()
    coco_gt.dataset = gt_coco_dict
    coco_gt.createIndex()

    if len(det_list):
        coco_dt = coco_gt.loadRes(det_list)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": gt_coco_dict["images"],
            "annotations": [],
            "categories": gt_coco_dict["categories"],
        }
        coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.maxDets = list(max_dets)

    # Evaluate all categories.
    coco_eval.params.catIds = [i + 1 for i in range(len(class_names))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP@[.5:.95]": coco_eval.stats[0],
        "AP@0.50": coco_eval.stats[1],
        "AP@0.75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_max=1": coco_eval.stats[6],
        "AR_max=10": coco_eval.stats[7],
        "AR_max=100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }
