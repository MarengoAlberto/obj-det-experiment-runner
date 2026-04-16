from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def build_coco_from_simple(gt_simple, cat_id=1, cat_name="Reg-plate"):
    images = []
    annos = []
    ann_id = 1
    for id, sample in enumerate(gt_simple, start=1):
        w, h = sample['img_size']
        images.append({"id": id, "width": int(w), "height": int(h)})

        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [cat_id] * len(boxes))
        for box, lab in zip(boxes, labels):
            x1, y1, x2, y2 = map(float, box)
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            if w_box <= 0 or h_box <= 0:  # skip degenerate
                continue
            annos.append({
                "id": ann_id,
                "image_id": id,
                "category_id": int(lab),
                "bbox": [x1, y1, w_box, h_box],  # COCO expects xywh
                "area": w_box * h_box,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [{"id": cat_id, "name": cat_name}]
    return {"images": images, "annotations": annos, "categories": categories, "info": "COCO dataset eval"}


def convert_preds_xyxy_to_coco(pred_simple, cat_id=1):
    coco_dets = []
    for id, p in enumerate(pred_simple, start=1):
        for box, sc in zip(p["boxes"], p["scores"]):
            coco_dets.append({
                "category_id": cat_id,
                "image_id": id,
                "bbox": xyxy_to_xywh(box),  # convert to xywh
                "score": float(sc),
            })
    return coco_dets

def coco_eval(gt, preds, max_dets=(1, 10, 100), iou_type="bbox"):
    gt_coco_dict = build_coco_from_simple(gt)
    det_list = convert_preds_xyxy_to_coco(preds)
    # Build COCO API from an in-memory dict (no need to write a file)
    coco_gt = COCO()
    coco_gt.dataset = gt_coco_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(det_list) if len(det_list) else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.maxDets = list(max_dets)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Return as a dict too
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
