import math
import torch
from torchvision.ops import nms


class YOLODataEncoder:
    def __init__(self,
                 input_size=(300, 300),
                 classes=("__background__", "person"),
                 strides=[4, 8, 16, 32, 64, 128]):
        self.input_size = input_size
        self.classes = classes
        self.strides = strides
        self.grid_sizes = []
        self.grid_centers = self.get_all_centers()

    def encode(self, boxes, classes):
        if boxes.shape[0] == 0 and classes.shape[0] == 0:
            return torch.zeros((self.grid_centers.shape[0], 4 + 1 + 1),
                                device=self.grid_centers.device,
                                dtype=self.grid_centers.dtype)
        idx, classes_assigned = self._assign_boxes_to_cells(boxes, classes)
        bboxes_assigned = self._assign_tensor_with_zeros(boxes, idx)
        objness = (idx >= 0).reshape(-1, 1)
        classes_assigned = classes_assigned.reshape(-1, 1)
        bboxes_enc = self._encode_boxes(bboxes_assigned, objness)
        return torch.cat([objness, bboxes_enc, classes_assigned], dim=1)

    def decode(self, logits, nms_threshold=0.5, score_threshold=0.5, max_dets=100):
        input_w, input_h = self.input_size
        device = logits.device
        cell_centers = self.grid_centers[:, :2]
        cell_sizes = self.grid_centers[:, 2]

        min_size_clamp = torch.tensor([0., 0., 0., 0.], device=device)
        max_size_clamp = torch.tensor([input_w, input_h, input_w, input_h], device=device)

        obj_mask = logits[:, 0]
        boxes_enc = logits[:, 1:4]
        cls_pred = logits[:, 4:]

        # loc_pred shape: [#anchors, 4], # cls_pred shape: [#anchors, #num_classes]
        pred_boxes, cls_pred, obj_scores = self._decode_boxes(boxes_enc, cls_pred, obj_mask)

        pred_boxes = torch.clamp(pred_boxes, min=min_size_clamp, max=max_size_clamp)

        pred_confs = cls_pred.softmax(dim=1)  # shape: [#anchors, #num_classes]

        # Perform Argmax
        max_class_prob, conf_argmax = pred_confs.max(dim=1, keepdim=True)  # shape: [#anchors, 1]

        # objectness logits -> objectness probability
        obj_prob = obj_scores.sigmoid().reshape(-1, 1)
        final_conf_scores = max_class_prob * obj_prob

        # Combined Tensor: shape [#anchors ,6].
        # 6: [xmin, ymin, xmax, ymax, conf_score, class_id]
        combined_tensor = torch.cat([pred_boxes, final_conf_scores, conf_argmax], dim=1)

        # Store final boxes that needs to be retained.
        chosen_boxes = []

        for cls_idx, cls_name in enumerate(self.classes):

            if cls_name == "__background__":
                continue

            # Get current class ID from comnined_tensor
            class_ids = torch.where(combined_tensor[:, 5].int() == cls_idx)[0]

            class_tensor = combined_tensor[class_ids]  # shape: [#class_ids, 6]
            class_boxes = class_tensor[:, :4]
            class_conf = class_tensor[:, 4]

            keep = nms(boxes=class_boxes, scores=class_conf, iou_threshold=nms_threshold)
            filtered_ids = torch.where(class_conf[keep] > score_threshold)[0]

            # Final boxes and conf. scores to be retained for the current class
            # after NMS.
            # The number of final boxes is constrained by max_dets.
            final_box_data = class_tensor[keep][filtered_ids][:max_dets]

            chosen_boxes.append(final_box_data)

        return torch.cat(chosen_boxes)

    def get_all_centers(self):
        all_centers = []
        for fm_size in self.strides:
            all_centers.append(self._get_cell_centers(fm_size))
        return torch.cat(all_centers, dim=0)

    def _get_cell_centers(self, fm_size = 4):

        img_h, img_w = self.input_size

        grid_h = math.ceil(img_h / fm_size)
        grid_w = math.ceil(img_w / fm_size)

        grid_h_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_h + grid_h / 2
        grid_w_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_w + grid_w / 2

        x, y = torch.meshgrid(grid_w_coords, grid_h_coords, indexing="xy")

        xy = torch.stack([x, y], dim=2)
        cell_centers = xy.reshape(-1, 2)
        self.grid_sizes.append((grid_h, grid_w, fm_size))
        return torch.cat([cell_centers, torch.tensor((grid_h, grid_w, fm_size)).repeat(cell_centers.shape[0], 1)], dim=1)

    def _assign_boxes_to_cells(self, boxes, classes, background_id=0):
        cell_centers = self.grid_centers[:, :2]
        cell_sizes = self.grid_centers[:, 2]
        classes = torch.tensor(classes, dtype=torch.int64)
        box_centers = torch.stack(
            [
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
            ],
            dim=1
        )

        diff = cell_centers[:, None, :] - box_centers[None, :, :]
        half = (cell_sizes * 0.5)[:, None, None]
        inside = (diff.abs() <= half).all(dim=2)

        assigned_box_ids = torch.full((cell_centers.shape[0],), -1, dtype=torch.long)
        assigned_classes = torch.full((cell_centers.shape[0],), background_id, dtype=torch.long)

        has_box = inside.any(dim=1)
        first_match = inside.float().argmax(dim=1)

        assigned_box_ids[has_box] = first_match[has_box]
        assigned_classes[has_box] = classes[first_match[has_box]].squeeze(-1)

        return assigned_box_ids, assigned_classes

    def _assign_tensor_with_zeros(self, tensor, idx, out_shape=4):
        # bboxes_norm: [n_boxes, 4]
        # idx: [n_cells], values in [0, n_boxes-1] or -1

        out = torch.zeros((idx.shape[0], out_shape), device=tensor.device, dtype=bboxes_norm.dtype)

        valid = idx >= 0
        out[valid] = tensor[idx[valid]]

        return out

    def _encode_boxes(self, boxes, obj_mask, variances=(0.1, 0.2)):
        cell_centers = self.grid_centers[:, :2]
        cell_sizes = self.grid_centers[:, 2]
        b_wh = boxes[:, 2:] - boxes[:, :2]
        b_ctr = boxes[:, :2] + 0.5 * b_wh

        if cell_sizes.ndim == 1:
            cell_sizes_xy = cell_sizes[:, None]     # [16, 1]
        else:
            cell_sizes_xy = cell_sizes
        dxdy = (b_ctr - cell_centers) / (cell_sizes_xy * variances[0])
        dwdh = torch.log((b_wh).clamp(min=1e-6)) / variances[1]
        dxdy = dxdy.masked_fill(~obj_mask, 0)
        dwdh = dwdh.masked_fill(~obj_mask, 0)
        return torch.cat([dxdy, dwdh], dim=1)

    def _decode_boxes(self, deltas, classes, obj_mask, variances=(0.1, 0.2)):
        cell_centers = self.grid_centers[:, :2]
        cell_sizes = self.grid_centers[:, 2]
        dxy = (deltas[:, :2] * variances[0])
        dwh = (deltas[:, 2:] * variances[1])
        dwh = dwh.clamp(min=-4.0, max=4.0).exp()

        if cell_sizes.ndim == 1:
            cell_sizes_xy = cell_sizes[:, None]     # [16, 1]
        else:
            cell_sizes_xy = cell_sizes

        p_ctr = cell_centers + dxy * cell_sizes_xy
        half = 0.5 * dwh
        decoded_boxes = torch.cat([p_ctr - half, p_ctr + half], dim=1)
        obj_mask = (obj_mask >= 0.5)  # bool mask
        keep = obj_mask.squeeze(-1) if obj_mask.ndim > 1 else obj_mask
        return decoded_boxes[keep], classes[keep], obj_mask[keep]

class DataEncoder:
    def __init__(self, input_size=(300, 300), classes=("__background__", "person")):
        self.input_size = input_size
        self.anchor_boxes, self.aspect_ratios, self.scales = get_all_anchor_boxes(input_size=self.input_size)
        self.classes = classes

    def encode(self, boxes, classes):
        if boxes.shape[0] == 0 and classes.shape[0] == 0:
            return (torch.zeros((self.anchor_boxes.shape[0], 4),
                                device=self.anchor_boxes.device,
                                dtype=self.anchor_boxes.dtype),
                    torch.squeeze(torch.zeros((self.anchor_boxes.shape[0], 1),
                                              device=self.anchor_boxes.device,
                                              dtype=self.anchor_boxes.dtype)))
        iou = compute_iou(src=boxes, dst=self.anchor_boxes)
        iou, ids = iou.max(dim=1)

        cls_targets = classes[ids]
        cls_targets[iou < 0.5] = -1
        cls_targets[iou < 0.3] = 0

        loc_targets = encode_boxes(boxes=boxes[ids], anchors=self.anchor_boxes)
        return loc_targets, cls_targets

    def decode(self,
               loc_pred,
               cls_pred,
               device,
               nms_threshold=0.5,
               score_threshold=0.5,
               max_dets=100,
               ):

        input_h, input_w = self.input_size[:2]

        min_size_clamp = torch.tensor([0., 0., 0., 0.], device=device)
        max_size_clamp = torch.tensor([input_w, input_h, input_w, input_h], device=device)

        self.anchor_boxes = self.anchor_boxes.to(device)

        # loc_pred shape: [#anchors, 4], # cls_pred shape: [#anchors, #num_classes]
        pred_boxes = decode_boxes(deltas=loc_pred, anchors=self.anchor_boxes)  # shape: [#anchors, 4]

        pred_boxes = torch.clamp(pred_boxes, min=min_size_clamp, max=max_size_clamp)

        pred_confs = cls_pred.softmax(dim=1)  # shape: [#anchors, #num_classes]

        # Perform Argmax
        max_conf_score, conf_argmax = pred_confs.max(dim=1, keepdim=True)  # shape: [#anchors, 1]

        # Combined Tensor: shape [#anchors ,6].
        # 6: [xmin, ymin, xmax, ymax, conf_score, class_id]
        combined_tensor = torch.cat([pred_boxes, max_conf_score, conf_argmax], dim=1)

        # Store final boxes that needs to be retained.
        chosen_boxes = []

        for cls_idx, cls_name in enumerate(self.classes):

            if cls_name == "__background__":
                continue

            # Get current class ID from comnined_tensor
            class_ids = torch.where(combined_tensor[:, 5].int() == cls_idx)[0]

            class_tensor = combined_tensor[class_ids]  # shape: [#class_ids, 6]
            class_boxes = class_tensor[:, :4]
            class_conf = class_tensor[:, 4]

            keep = nms(boxes=class_boxes, scores=class_conf, iou_threshold=nms_threshold)
            # keep = soft_nms(class_boxes, class_conf, sigma=nms_threshold, score_thresh=0.001, topk=300) #score_threshold

            filtered_ids = torch.where(class_conf[keep] > score_threshold)[0]

            # Final boxes and conf. scores to be retained for the current class
            # after NMS.
            # The number of final boxes is constrained by max_dets.
            final_box_data = class_tensor[keep][filtered_ids][:max_dets]

            chosen_boxes.append(final_box_data)

        return torch.cat(chosen_boxes)

    def get_num_anchors(self):
        return len(self.aspect_ratios) * len(self.scales)

def get_all_anchor_boxes(input_size, anchor_areas=None, aspect_ratios=None, scales=None):
    if anchor_areas is None:
        anchor_areas = [16 ** 2, 32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 512 ** 2]
        # [
        #     8 * 8,  # For feature_map size: 38x38
        #     16 * 16.0,  # For feature_map size: 19x19
        #     32 * 32.0,  # For feature_map size: 10x10
        #     64 * 64.0,  # For feature_map size: 5x5
        #     128 * 128,  # For feature_map size: 3x3
        # ]  # p3 -> p7

    if aspect_ratios is None:
        aspect_ratios = [5.0, 3.0, 2.0, 1.0]  # [4.0, 3.0, 2.0, 1.5]  #[1.19, 1.81, 2.45, 3.82]  #[0.5, 1.0, 2.0]

    if scales is None:
        scales = [
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P2 [0.1, 0.25, 0.5, 1.0]
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P3 (small/med) [0.1, 0.25, 0.5, 1.0]
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P4
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P5 [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P6  [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
            [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P7  [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
        ]
        # [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)] #[1.0, 2**(1/3), 2**(2/3)]
        # [1, pow(2, 1 / 3.0), pow(2, 3.5 / 3.0)] #[1, pow(2, 1 / 3.0), pow(2, 2 / 3.0)]

    num_fms = len(anchor_areas)
    # fm_sizes = [math.ceil(input_size[0] / pow(2.0, i + 3)) for i in range(num_fms)]
    strides = [4, 8, 16, 32, 64, 128][:num_fms]
    fm_sizes = [math.ceil(input_size[0] / s) for s in strides]

    anchor_boxes = []

    for idx, fm_size in enumerate(fm_sizes):
        anchors = generate_anchors(anchor_areas[idx], aspect_ratios, scales[idx])
        anchor_grid = generate_anchor_grid(input_size, fm_size, anchors)
        anchor_boxes.append(anchor_grid)

    anchor_boxes = torch.concat(anchor_boxes, dim=0)

    return anchor_boxes, aspect_ratios, scales

def generate_anchors(anchor_area, aspect_ratios, scales):
    anchors = []

    for scale in scales:
        for ratio in aspect_ratios:
            h = scale * (math.sqrt(anchor_area / ratio))  # h*w * h/w = sqrt(h**2)
            w = ratio * h  # w/h * h

            # Assume the anchor box is centered at origin (0, 0)
            # Get xmin, ymin, xmax, ymax of anchor box w.r.t origin (0, 0)
            box_w_half = w / 2
            box_h_half = h / 2

            x1 = 0.0 - box_w_half
            y1 = 0.0 - box_h_half

            x2 = 0.0 + box_w_half
            y2 = 0.0 + box_h_half

            anchors.append([x1, y1, x2, y2])

    return torch.tensor(anchors, dtype=torch.float)

def generate_anchor_grid(input_size, fm_size, anchors):
    img_h, img_w = input_size

    grid_h = math.ceil(img_h / fm_size)
    grid_w = math.ceil(img_w / fm_size)

    grid_h_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_h + grid_h / 2
    grid_w_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_w + grid_w / 2

    # Create a numpy-like cartesian meshgrid.
    x, y = torch.meshgrid(grid_w_coords, grid_h_coords, indexing="xy")

    xyxy = torch.stack([x, y, x, y], dim=2)
    anchors = anchors.reshape(-1, 1, 1, 4)
    boxes = (xyxy + anchors).permute(1, 2, 0, 3).reshape(-1, 4)

    return boxes

def compute_iou(src, dst):
    # src: [Ns,4], dst: [Nd,4]  (xyxy)
    lt = torch.maximum(dst[:, None, :2], src[:, :2])  # [Nd,Ns,2]
    rb = torch.minimum(dst[:, None, 2:], src[:, 2:])  # [Nd,Ns,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_s = (src[:, 2] - src[:, 0]).clamp(min=0) * (src[:, 3] - src[:, 1]).clamp(min=0)
    area_d = (dst[:, 2] - dst[:, 0]).clamp(min=0) * (dst[:, 3] - dst[:, 1]).clamp(min=0)
    union = area_d[:, None] + area_s - inter + 1e-9
    return inter / union


# SSD-style box coder w/o +1/-1 and with variances (next section):
VAR_CTR, VAR_SIZE = 0.1, 0.2

def encode_boxes(boxes, anchors):
    a_wh = anchors[:, 2:] - anchors[:, :2]
    a_ctr = anchors[:, :2] + 0.5 * a_wh
    b_wh = boxes[:, 2:] - boxes[:, :2]
    b_ctr = boxes[:, :2] + 0.5 * b_wh

    dxdy = (b_ctr - a_ctr) / (a_wh * VAR_CTR)
    dwdh = torch.log((b_wh / a_wh).clamp(min=1e-6)) / VAR_SIZE
    return torch.cat([dxdy, dwdh], dim=1)

def decode_boxes(deltas, anchors):
    a_wh = anchors[:, 2:] - anchors[:, :2]
    a_ctr = anchors[:, :2] + 0.5 * a_wh

    dxy = deltas[:, :2] * VAR_CTR
    dwh = deltas[:, 2:] * VAR_SIZE
    dwh = dwh.clamp(min=-4.0, max=4.0).exp()

    p_ctr = a_ctr + dxy * a_wh
    p_wh = a_wh * dwh
    half = 0.5 * p_wh
    return torch.cat([p_ctr - half, p_ctr + half], dim=1)
