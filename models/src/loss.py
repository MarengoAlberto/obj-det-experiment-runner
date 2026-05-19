import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Loss:
    def __init__(self, cfg, data_encoder=None):
        self.cfg = cfg
        assert data_encoder is not None, "data_encoder must be provided for loss calculation"
        self.data_encoder = data_encoder
        self.yolo_mode = cfg.model.name == "yolo"
        if self.yolo_mode:
            self.loss_fn = YOLODetectionLoss(num_classes=cfg.dataset.nc,
                                             grid_centers=data_encoder.grid_centers,
                                             obj_loss=cfg.loss.obj_loss,
                                             cls_loss=cfg.loss.cls_loss,
                                             box_loss=cfg.loss.box_loss,
                                             lambda_obj=cfg.loss.lambda_obj,
                                             lambda_bbox=cfg.loss.lambda_bbox,
                                             lambda_cls=cfg.loss.lambda_cls,
                                             focal_alpha=cfg.loss.focal_alpha,
                                             focal_gamma=cfg.loss.focal_gamma,
                                             apply_negative_mining=cfg.loss.apply_negative_mining,
                                             obj_neg_weight=cfg.loss.obj_neg_weight,
                                             s1_beta=cfg.loss.s1_beta,
                                             s1_reduction=cfg.loss.s1_reduction,
                                             hybrid_ciou_weight=cfg.loss.hybrid_ciou_weight,
                                             sw_base=cfg.loss.stride_weight.base,
                                             sw_min=cfg.loss.stride_weight.min,
                                             sw_max=cfg.loss.stride_weight.max,
                                             )
        else:
            self.loc_wt = cfg.loss.loc_loss.loss_weight
            self.cls_wt = cfg.loss.cls_loss.loss_weight
            self.loss_fn = {
                "loc_loss": self.get_loc_loss_fn(cfg.loss.loc_loss.name),
                "cls_loss": self.get_cls_loss_fn(cfg.loss.cls_loss.name)
            }

    def __call__(self, y_true, y_pred):
        if self.yolo_mode:
            pred_logits = y_pred[0] if isinstance(y_pred, (tuple, list)) else y_pred
            target_encoded = y_true[0] if isinstance(y_true, (tuple, list)) else y_true
            total_loss, parts = self.loss_fn(pred_logits, target_encoded)
            return {
                "loc_loss": parts["bbox"],
                "cls_loss": parts["cls"],
                "obj_loss": parts["obj"],
                "total_loss": total_loss,
            }

        pred_boxes, pred_labels = y_pred
        box_targets, cls_targets = y_true
        loc_loss = self.loss_fn["loc_loss"](pred_boxes, box_targets, cls_targets)
        cls_loss = self.loss_fn["cls_loss"](pred_labels, cls_targets)
        total_loss = self.loc_wt * loc_loss + self.cls_wt * cls_loss
        return {"loc_loss": loc_loss, "cls_loss": cls_loss, "total_loss": total_loss}

    def get_loc_loss_fn(self, loc_loss):
        if loc_loss == "smooth_l1":
            return torch.nn.SmoothL1Loss(reduction="mean")
        elif loc_loss == "IoU":
            return IoULoss(anchors=self.data_encoder.anchor_boxes,
                           encoded=self.cfg.loss.loc_loss.encoded,
                           iou_type=self.cfg.loss.loc_loss.iou_type,
                           eps=self.cfg.loss.loc_loss.eps)
        else:
            raise NotImplementedError(f"Localization loss {loc_loss} not implemented yet.")

    def get_cls_loss_fn(self, cls_loss):
        if cls_loss == "focal":
            return FocalLoss(num_classes=self.cfg.dataset.nc,
                             alpha=self.cfg.loss.cls_loss.alpha,
                             gamma=self.cfg.loss.cls_loss.gamma)
        elif cls_loss == "ohem":
            return OHEMLoss()
        else:
            raise NotImplementedError(f"Classification loss {cls_loss} not implemented yet.")

class SmoothL1Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loc_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(self, loc_preds, loc_targets, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets).
        '''

        ################################################################
        # loc_loss
        ################################################################

        cls_targets = cls_targets.long()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.sum()  # Scalar
        mask = pos.unsqueeze(dim=2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [num_pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [num_pos,4]

        loc_loss = self.loc_loss_fn(masked_loc_preds, masked_loc_targets)
        loc_loss = torch.nan_to_num(loc_loss.sum() / num_pos.float())

        return loc_loss

class OHEMLoss(nn.Module):
    def __init__(self, num_classes=2, neg2pos_ratio=3, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.negpos_ratio = neg2pos_ratio
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = OHEMLoss(cls_preds, cls_targets).
        '''

        ################################################################
        # cls_loss
        ################################################################

        cls_targets = cls_targets.long()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos_per_image_batch = pos.sum(dim=1, keepdim=True)  # [N, 1]
        total_pos = num_pos_per_image_batch.sum().float().clamp(min=1.0)  # Scalar

        cls_preds_reshaped = cls_preds.permute(dims=(0, 2, 1))  # [N, #classes, #anchors]
        cls_loss = self.cls_loss_fn(cls_preds_reshaped, cls_targets)  # [N, #anchors]

        pos_cls_loss = cls_loss[pos]  # [#total_positives, ]
        cls_loss[pos] = 0

        _, loss_idx = cls_loss.sort(dim=1, descending=True)  # [N, #anchors]

        _, idx_rank = loss_idx.sort(dim=1)  # [N, #anchors]

        num_neg_per_image_batch = torch.clamp(self.negpos_ratio * num_pos_per_image_batch, min=1,
                                              max=pos.shape[1] - 1)  # [N, 1]

        neg = idx_rank < num_neg_per_image_batch.expand_as(idx_rank)  # [N, #anchors]
        neg_cls_loss = cls_loss[neg]  # [#total_negatives, ]

        cls_loss = (pos_cls_loss.sum() + neg_cls_loss.sum()) / total_pos

        return cls_loss

# -----------------------------
# Classification: Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for dense detection.
    Accepts either softmax logits (C>=2) or a single sigmoid logit (C=1).

    Args:
        num_classes: number of classes. For binary softmax use 2.
        alpha: weight for the positive class (e.g., 0.25).
        gamma: focusing parameter (e.g., 2.0).
        ignore_index: targets equal to this are ignored.
    """

    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, cls_preds, cls_targets):
        """
        Args:
            cls_preds: [N, A, C] logits.
            cls_targets: [N, A] with values in {ignore_index, 0..C-1}.
                         For binary sigmoid, targets should be {0,1} or {ignore,0,1}.
        Returns:
            cls_loss: scalar
        """
        N, A, C = cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2]
        targets = cls_targets.long().view(-1)  # [N*A]
        logits = cls_preds.view(-1, C)  # [N*A, C]

        valid_mask = (targets != self.ignore_index)  # [N*A]
        if valid_mask.sum() == 0:
            return logits.new_tensor(0.0)

        # normalize by number of positives (like common one-stage practice)
        pos_mask = (targets == 1) if C >= 2 else (targets == 1)
        num_pos = torch.clamp(pos_mask[valid_mask].sum().float(), min=1.0)

        if C >= 2:
            # ----- Softmax focal -----
            # CE per anchor
            ce = F.cross_entropy(logits[valid_mask], targets[valid_mask], reduction="none")  # [Nv]
            with torch.no_grad():
                probs = F.softmax(logits[valid_mask], dim=-1)  # [Nv, C]
                p_t = probs[torch.arange(probs.size(0), device=probs.device), targets[valid_mask]]  # [Nv]
                # alpha for positive (class 1); 1-alpha for background (class 0)
                alpha_t = torch.where(targets[valid_mask] == 1,
                                      torch.full_like(p_t, self.alpha),
                                      torch.full_like(p_t, 1.0 - self.alpha))
            loss = (alpha_t * (1.0 - p_t).pow(self.gamma) * ce).sum() / num_pos
            return torch.nan_to_num(loss)

        else:
            # ----- Binary sigmoid focal (C == 1) -----
            # logits: [N*A, 1] -> [N*A]
            logits_bin = logits.squeeze(-1)
            targets_bin = targets.float()
            ce = F.binary_cross_entropy_with_logits(logits_bin[valid_mask], targets_bin[valid_mask], reduction="none")
            with torch.no_grad():
                p = torch.sigmoid(logits_bin[valid_mask])
                p_t = p * targets_bin[valid_mask] + (1 - p) * (1 - targets_bin[valid_mask])
                alpha_t = self.alpha * targets_bin[valid_mask] + (1 - self.alpha) * (1 - targets_bin[valid_mask])
            loss = (alpha_t * (1.0 - p_t).pow(self.gamma) * ce).sum() / num_pos
            return torch.nan_to_num(loss)

# -----------------------------
# Regression: IoU (CIoU/GIoU)
# -----------------------------
class IoULoss(nn.Module):
    """
    IoU-based regression loss: CIoU (default) or GIoU.

    Usage 1 (preferred): pass decoded boxes (xyxy) to forward().
        loc_preds/loc_targets shape: [N, A, 4] in xyxy.

    Usage 2: if you only have encoded deltas relative to anchors,
             provide anchors (cx,cy,w,h) to the constructor and set encoded=True.
             The class will decode deltas using Retina/SSD-style parameterization:
                x = ax + dx * aw * v0
                y = ay + dy * ah * v0
                w = aw * exp(dw * v1)
                h = ah * exp(dh * v1)
    """

    def __init__(self, iou_type: str = "ciou", encoded: bool = False,
                 anchors: torch.Tensor = None, variances=(0.1, 0.2), eps=1e-7):
        super().__init__()
        assert iou_type in {"giou", "ciou"}, "iou_type must be 'giou' or 'ciou'"
        self.iou_type = iou_type
        self.encoded = encoded
        self.anchors = anchors  # Tensor [A,4] in (cx,cy,w,h) if encoded=True
        self.v0, self.v1 = float(variances[0]), float(variances[1])
        self.eps = float(eps)

    @staticmethod
    def _xywh_to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _decode(self, deltas, anchors):
        # deltas, anchors: [..., 4] with anchors in cx,cy,w,h
        dx, dy, dw, dh = deltas.unbind(-1)
        acx, acy, aw, ah = anchors.unbind(-1)
        px = acx + dx * aw * self.v0
        py = acy + dy * ah * self.v0
        pw = aw * torch.exp(dw * self.v1)
        ph = ah * torch.exp(dh * self.v1)
        return torch.stack((px, py, pw, ph), dim=-1)

    def _giou_ciou_loss(self, pred_xyxy, tgt_xyxy):
        # pred_xyxy/tgt_xyxy: [M,4] (only positives)
        x1p, y1p, x2p, y2p = pred_xyxy.unbind(-1)
        x1t, y1t, x2t, y2t = tgt_xyxy.unbind(-1)

        # areas
        wp = (x2p - x1p).clamp(min=self.eps)
        hp = (y2p - y1p).clamp(min=self.eps)
        wt = (x2t - x1t).clamp(min=self.eps)
        ht = (y2t - y1t).clamp(min=self.eps)
        area_p = wp * hp
        area_t = wt * ht

        # intersection
        x1i = torch.max(x1p, x1t)
        y1i = torch.max(y1p, y1t)
        x2i = torch.min(x2p, x2t)
        y2i = torch.min(y2p, y2t)
        wi = (x2i - x1i).clamp(min=0)
        hi = (y2i - y1i).clamp(min=0)
        inter = wi * hi

        union = area_p + area_t - inter + self.eps
        iou = inter / union

        if self.iou_type == "giou":
            # enclosing box
            x1c = torch.min(x1p, x1t)
            y1c = torch.min(y1p, y1t)
            x2c = torch.max(x2p, x2t)
            y2c = torch.max(y2p, y2t)
            wc = (x2c - x1c).clamp(min=self.eps)
            hc = (y2c - y1c).clamp(min=self.eps)
            area_c = wc * hc
            giou = iou - (area_c - union) / area_c
            loss = 1.0 - giou
            return loss

        # CIoU (includes distance + aspect terms)
        # center distance
        cxp = (x1p + x2p) * 0.5
        cyp = (y1p + y2p) * 0.5
        cxt = (x1t + x2t) * 0.5
        cyt = (y1t + y2t) * 0.5
        rho2 = (cxp - cxt) ** 2 + (cyp - cyt) ** 2

        # diagonal length of smallest enclosing box
        x1c = torch.min(x1p, x1t)
        y1c = torch.min(y1p, y1t)
        x2c = torch.max(x2p, x2t)
        y2c = torch.max(y2p, y2t)
        c2 = ((x2c - x1c) ** 2 + (y2c - y1c) ** 2).clamp(min=self.eps)

        # aspect ratio consistency
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(wt / ht) - torch.atan(wp / hp), 2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + self.eps)

        ciou = iou - (rho2 / c2) - alpha * v
        loss = 1.0 - ciou
        return loss

    def forward(self, loc_preds, loc_targets, cls_targets):
        """
        Args:
            loc_preds:  [N, A, 4] (xyxy if encoded=False; deltas if encoded=True)
            loc_targets:[N, A, 4] (xyxy if encoded=False; deltas if encoded=True)
            cls_targets:[N, A] (0=bg, >0=pos, -1=ignore)
        Returns:
            loc_loss: scalar
        """
        N, A, _ = loc_preds.shape
        cls_targets = cls_targets.long()
        pos = (cls_targets > 0)  # [N,A]
        num_pos = torch.clamp(pos.sum().float(), min=1.0)

        if pos.sum() == 0:
            return loc_preds.new_tensor(0.0)

        if self.encoded:
            assert self.anchors is not None and self.anchors.shape[0] == A, \
                "Provide anchors [A,4] in (cx,cy,w,h) to decode deltas"
            anchors = self.anchors.to(loc_preds.device).unsqueeze(0).expand(N, A, 4)  # [N,A,4]
            pred_xywh = self._decode(loc_preds, anchors)  # [N,A,4] (cxcywh)
            tgt_xywh = self._decode(loc_targets, anchors)
            pred = self._xywh_to_xyxy(pred_xywh)[pos].view(-1, 4)
            tgt = self._xywh_to_xyxy(tgt_xywh)[pos].view(-1, 4)
        else:
            pred = loc_preds[pos].view(-1, 4)  # xyxy
            tgt = loc_targets[pos].view(-1, 4)  # xyxy

        loss_vec = self._giou_ciou_loss(pred, tgt)  # [num_pos]
        loss = torch.nan_to_num(loss_vec.sum() / num_pos)
        return loss

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def softmax_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    # targets: class indices [K], logits: [K, C]
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    # Simple scalar alpha (same for all positive classes)
    at = torch.full_like(pt, alpha)
    loss = at * (1.0 - pt).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

class YOLODetectionLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 grid_centers,
                 obj_loss="focal",
                 cls_loss="focal",
                 box_loss="s1",
                 lambda_obj=1.0,
                 lambda_bbox=5.0,
                 lambda_cls=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 apply_negative_mining=False,
                 obj_neg_weight=1.0,
                 s1_beta=1.0,
                 s1_reduction="mean",
                 hybrid_ciou_weight=0.5,
                 sw_base=12.0,
                 sw_min=0.75,
                 sw_max=2.5,):
        super().__init__()
        self.num_classes = int(num_classes)
        self.obj_loss = obj_loss
        self.cls_loss = cls_loss
        self.box_loss = box_loss
        self.lambda_obj = float(lambda_obj)
        self.lambda_bbox = float(lambda_bbox)
        self.lambda_cls = float(lambda_cls)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.apply_negative_mining = apply_negative_mining
        self.obj_neg_weight = float(obj_neg_weight)
        self.s1_beta = float(s1_beta)
        self.s1_reduction = s1_reduction
        self.hybrid_ciou_weight = float(hybrid_ciou_weight)
        self.sw_base = sw_base
        self.sw_min = sw_min
        self.sw_max = sw_max
        self.register_buffer("grid_centers", grid_centers)

    # ------------------------------------------------------------------
    # IoU-family helpers
    # ------------------------------------------------------------------

    def _decode_deltas_to_cxcywh(self, deltas, pos_mask, batch_size):
        encoded_space = 4 + self.num_classes
        # Tile grid_centers to match the flattened batch layout [B*N, 5]
        gc_tiled = self.grid_centers.to(deltas.device).unsqueeze(0).expand(batch_size, -1, -1)
        gc_tiled = gc_tiled.reshape(-1, encoded_space)
        gc = gc_tiled[pos_mask]

        cell_centers = gc[:, :2]
        strides = gc[:, 4:5]

        dxy = deltas[:, :2]
        dwh = deltas[:, 2:]

        p_ctr = cell_centers + dxy * strides
        p_wh = dwh.clamp(min=-4.0, max=4.0).exp() * strides

        return torch.cat([p_ctr, p_wh], dim=1)  # (cx, cy, w, h) in pixels

    def _get_pos_strides(self, pos_mask, batch_size, device):
        """
        Returns stride for each positive prediction after pred/target flattening.

        pos_mask shape: [B * N]
        output shape: [num_pos]
        """
        grid_centers = self.grid_centers.to(device)  # [N, 5]
        strides = grid_centers[:, 4]  # [N]

        strides_tiled = strides.unsqueeze(0).expand(batch_size, -1).reshape(-1)

        return strides_tiled[pos_mask]

    @staticmethod
    def _decode_to_xyxy(boxes):
        """
        Convert (cx, cy, w, h) → (x1, y1, x2, y2).
        Works for any leading batch dims.
        """
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2,
                             cx + w / 2, cy + h / 2], dim=-1)

    @staticmethod
    def _iou_base(pred_xyxy, tgt_xyxy):
        """
        Returns (iou, intersection, union, pred_area, tgt_area)
        for matched pairs — no cross-product.
        """
        inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h

        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
                    (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        tgt_area  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(min=0) * \
                    (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(min=0)

        union = pred_area + tgt_area - intersection + 1e-7
        iou   = intersection / union

        return iou, intersection, union, pred_area, tgt_area

    def _giou_loss(self, pred, tgt):
        """
        GIoU loss = 1 - GIoU
        Penalises non-overlapping boxes via the smallest enclosing box.
        Good general replacement for Smooth-L1.
        """
        pred_xyxy = self._decode_to_xyxy(pred)
        tgt_xyxy  = self._decode_to_xyxy(tgt)

        iou, _, union, _, _ = self._iou_base(pred_xyxy, tgt_xyxy)

        # Enclosing box
        enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])

        enc_area = (enc_x2 - enc_x1).clamp(min=0) * \
                   (enc_y2 - enc_y1).clamp(min=0) + 1e-7

        giou = iou - (enc_area - union) / enc_area
        return (1 - giou).mean()

    def _diou_loss(self, pred, tgt):
        """
        DIoU loss = 1 - DIoU
        Adds a centre-distance penalty on top of IoU.
        Converges faster than GIoU, especially when boxes don't overlap.
        """
        pred_xyxy = self._decode_to_xyxy(pred)
        tgt_xyxy  = self._decode_to_xyxy(tgt)

        iou, _, _, _, _ = self._iou_base(pred_xyxy, tgt_xyxy)

        # Centre distance²
        pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        tgt_cx  = (tgt_xyxy[:, 0]  + tgt_xyxy[:, 2])  / 2
        tgt_cy  = (tgt_xyxy[:, 1]  + tgt_xyxy[:, 3])  / 2
        centre_dist2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

        # Diagonal of enclosing box²
        enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
        diag2  = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        diou = iou - centre_dist2 / diag2
        return (1 - diou).mean()

    def _ciou_loss(self, pred, tgt):
        """
        CIoU loss = 1 - CIoU
        Extends DIoU with an aspect-ratio consistency term v.
        Best overall localization loss — use this if unsure.
        """
        pred_xyxy = self._decode_to_xyxy(pred)
        tgt_xyxy  = self._decode_to_xyxy(tgt)

        iou, _, _, _, _ = self._iou_base(pred_xyxy, tgt_xyxy)

        # Centre distance² / enclosing diagonal²  (same as DIoU)
        pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        tgt_cx  = (tgt_xyxy[:, 0]  + tgt_xyxy[:, 2])  / 2
        tgt_cy  = (tgt_xyxy[:, 1]  + tgt_xyxy[:, 3])  / 2
        centre_dist2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

        enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
        enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
        enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
        enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
        diag2  = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        # Aspect-ratio consistency term
        pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=1e-7)
        pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=1e-7)
        tgt_w  = (tgt_xyxy[:, 2]  - tgt_xyxy[:, 0]).clamp(min=1e-7)
        tgt_h  = (tgt_xyxy[:, 3]  - tgt_xyxy[:, 1]).clamp(min=1e-7)

        v = (4 / (math.pi ** 2)) * (
            torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)
        ) ** 2

        with torch.no_grad():
            alpha_ciou = v / (1 - iou + v + 1e-7)  # trade-off weight

        ciou = iou - centre_dist2 / diag2 - alpha_ciou * v
        return (1 - ciou).mean()

    def _compute_smooth_l1_bbox_loss(self, p, t, pos_mask, batch_size):
        if self.s1_reduction in ["mean", "sum"]:
            return F.smooth_l1_loss(p, t, beta=self.s1_beta, reduction=self.s1_reduction)
        elif self.s1_reduction == "custom_stride_weighted":
            pos_strides = self._get_pos_strides(
                pos_mask=pos_mask,
                batch_size=batch_size,
                device=p.device,
            )

            bbox_loss_per = F.smooth_l1_loss(
                p,
                t,
                beta=0.5,
                reduction="none",
            ).sum(dim=1)

            weights = (self.sw_base / pos_strides).clamp(self.sw_min, self.sw_max)

            return (bbox_loss_per * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            return F.smooth_l1_loss(p, t, beta=self.s1_beta)


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pred_logits, target_encoded):

        batch_size = pred_logits.shape[0] if pred_logits.ndim == 3 else 1

        if pred_logits.ndim == 3:
            pred_logits    = pred_logits.reshape(-1, pred_logits.shape[-1])
            target_encoded = target_encoded.reshape(-1, target_encoded.shape[-1])

        pred_obj  = pred_logits[:, 0]
        pred_bbox = pred_logits[:, 1:5]
        pred_cls  = pred_logits[:, 5:5 + self.num_classes]

        tgt_obj  = target_encoded[:, 0].to(pred_obj.dtype)
        tgt_bbox = target_encoded[:, 1:5].to(pred_bbox.dtype)
        tgt_cls  = target_encoded[:, 5].long()

        pos_mask = tgt_obj > 0.5
        neg_mask = ~pos_mask

        # ---- objectness loss ----
        if self.obj_loss == "focal":
            loss_obj_pos = sigmoid_focal_loss(
                pred_obj[pos_mask], tgt_obj[pos_mask],
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean",
            ) if pos_mask.any() else pred_obj.new_tensor(0.0)

            if self.apply_negative_mining:
                hard_neg_mask = self._hard_negative_mining(pred_obj, neg_mask)
                active_neg = hard_neg_mask
            else:
                active_neg = neg_mask

            loss_obj_neg = sigmoid_focal_loss(
                pred_obj[active_neg], tgt_obj[active_neg],
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean",
            ) if active_neg.any() else pred_obj.new_tensor(0.0)

            loss_obj = loss_obj_pos + self.obj_neg_weight * loss_obj_neg

        elif self.obj_loss == "bce":
            loss_obj_pos = F.binary_cross_entropy_with_logits(
                pred_obj[pos_mask], tgt_obj[pos_mask], reduction="mean"
            ) if pos_mask.any() else pred_obj.new_tensor(0.0)

            active_neg = self._hard_negative_mining(pred_obj, neg_mask) if self.apply_negative_mining else neg_mask

            loss_obj_neg = F.binary_cross_entropy_with_logits(
                pred_obj[active_neg], tgt_obj[active_neg], reduction="mean"
            ) if active_neg.any() else pred_obj.new_tensor(0.0)

            loss_obj = loss_obj_pos + self.obj_neg_weight * loss_obj_neg
        else:
            raise ValueError(f"Unsupported obj_loss: {self.obj_loss}")

        # ---- box + cls loss (positives only) ----
        if pos_mask.any():
            p = pred_bbox[pos_mask]
            t = tgt_bbox[pos_mask]

            if self.box_loss == "s1":
                loss_bbox = self._compute_smooth_l1_bbox_loss(p, t, pos_mask, batch_size)
            elif self.box_loss == "l1":
                loss_bbox = F.l1_loss(p, t)
            elif self.box_loss in ("giou", "diou", "ciou", "hybrid_s1_ciou"):
                # ← Decode deltas → absolute (cx,cy,w,h) before IoU geometry
                p_dec = self._decode_deltas_to_cxcywh(p, pos_mask, batch_size)
                t_dec = self._decode_deltas_to_cxcywh(t, pos_mask, batch_size)
                if self.box_loss == "giou":
                    loss_bbox = self._giou_loss(p_dec, t_dec)
                elif self.box_loss == "diou":
                    loss_bbox = self._diou_loss(p_dec, t_dec)
                elif self.box_loss == "ciou":
                    loss_bbox = self._ciou_loss(p_dec, t_dec)
                elif self.box_loss == "hybrid_s1_ciou":
                    ciou_loss = self._ciou_loss(p_dec, t_dec)
                    smooth_l1_loss = self._compute_smooth_l1_bbox_loss(p, t, pos_mask, batch_size)
                    loss_bbox = smooth_l1_loss + self.hybrid_ciou_weight * ciou_loss
            else:
                raise ValueError(f"Unsupported box_loss: {self.box_loss}")

            if self.cls_loss == "focal":
                loss_cls = softmax_focal_loss(
                    pred_cls[pos_mask], tgt_cls[pos_mask],
                    alpha=self.focal_alpha, gamma=self.focal_gamma)
            elif self.cls_loss == "ce":
                loss_cls = F.cross_entropy(pred_cls[pos_mask], tgt_cls[pos_mask])
            elif self.cls_loss == "bce":
                tgt_onehot = F.one_hot(tgt_cls[pos_mask],
                                       num_classes=self.num_classes).to(pred_cls.dtype)
                loss_cls = F.binary_cross_entropy_with_logits(
                    pred_cls[pos_mask], tgt_onehot)
            else:
                raise ValueError(f"Unsupported cls_loss: {self.cls_loss}")
        else:
            z = pred_logits.new_tensor(0.0)
            loss_bbox, loss_cls = z, z

        total = (self.lambda_obj  * loss_obj  +
                 self.lambda_bbox * loss_bbox +
                 self.lambda_cls  * loss_cls)

        return total, {
            "obj":  loss_obj.detach(),
            "bbox": loss_bbox.detach(),
            "cls":  loss_cls.detach(),
        }

    # ------------------------------------------------------------------
    # Hard negative mining
    # ------------------------------------------------------------------

    def _hard_negative_mining(self, pred_obj, neg_mask, topk_ratio=0.03, min_negatives=256):
        if not neg_mask.any():
            return neg_mask

        neg_scores = pred_obj[neg_mask].detach().sigmoid()
        n_hard = min(max(min_negatives, int(topk_ratio * neg_scores.numel())),
                     neg_scores.numel())

        _, hard_idx   = neg_scores.topk(n_hard)
        hard_neg_mask = torch.zeros_like(neg_mask)
        neg_indices   = neg_mask.nonzero(as_tuple=False).squeeze(1)
        hard_neg_mask[neg_indices[hard_idx]] = True

        return hard_neg_mask
