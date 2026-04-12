import torch
from torch import optim
import math

class OptimizerSetup:
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = None
        self.scheduler = None

    def get_optimizer(self):
        if self.cfg.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self._make_param_groups(wd=self.cfg.wd),
                                         lr=self.cfg.lr,
                                         betas=(self.cfg.beta_min, self.cfg.beta_max),
                                         eps=self.cfg.eps)
        else:
            raise ValueError('Unknown optimizer')
        if self.cfg.scheduler == "cosine_warmup":
            self.scheduler =self._build_cosine_warmup(self.optimizer,
                                                      self.cfg.epochs,
                                                      self.cfg.warmup_epochs,
                                                      self.cfg.min_lr_ratio)
        else:
            print(f"Unsupported scheduler: {self.cfg.scheduler}. No scheduler will be used.")
        return self.optimizer, self.scheduler

    def _build_cosine_warmup(self,
            optimizer: torch.optim.Optimizer,
            total_epochs: int,
            warmup_epochs: int = 1,
            min_lr_ratio: float = 0.05,):
        """
        Epoch-based scheduler: call scheduler.step() ONCE at the END of each epoch.
        Keeps LR constant within an epoch; updates between epochs only.
        """
        assert total_epochs > 0, "total_epochs must be > 0"
        warmup_epochs = max(0, int(warmup_epochs))
        remain = max(1, total_epochs - warmup_epochs)

        def lr_lambda(epoch_idx: int):
            # Linear warmup: from 1/warmup -> 1.0 of base LR
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)

            # Cosine decay: from 1.0 -> min_lr_ratio over remaining epochs
            progress = (epoch_idx - warmup_epochs) / float(remain)
            progress = min(max(progress, 0.0), 1.0)  # clamp [0,1]
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        # Scales each param group's base LR by the returned factor (keeps group ratios intact)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _make_param_groups(self, wd=5e-4):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_norm = ("bn" in n.lower()) or ("norm" in n.lower())
            if is_norm or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]