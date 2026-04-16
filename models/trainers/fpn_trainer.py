import os
import torch
from tqdm.auto import tqdm
import numpy as np
import random
from pathlib import Path
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_trainer import BaseTrainer
from ..src import Loss
from ..utils import DataSetup, OptimizerSetup, Metric, Wandb, initialize_directory, get_logger

class Trainer(BaseTrainer):

    rank, world_size, local_rank = 0, 1, 0
    use_ddp = False
    checkpoint_dir = "checkpoints"
    version = "version_0"
    train_loader = None
    val_loader = None
    train_sampler = None
    val_sampler = None
    optimizer = None
    scheduler = None
    criterion = None
    metric = None
    wandb = None
    device = None
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mAP": [], "val_mAP@50": []}

    def __init__(self, wrapper, data, cfg, logger=None):

        self.wrapper = wrapper
        self.model = wrapper.model
        self.data_encoder = wrapper.data_encoder
        self.cfg = cfg
        self.data = data

        # Initialize Logger
        self.logger = logger if logger else get_logger()

        # Initialize Training Components
        self._initialize_trainer()


    def train(self, n_epochs=None, batch_size=None, start_epoch=0):

        epochs = n_epochs or self.cfg.experiment.train.epochs
        if batch_size:
            self.cfg.experiment.train.batch_size = batch_size

        self.logger.info(f"Training Configuration: Epochs: {epochs}, Batch Size: {self.cfg.experiment.train.batch_size}")

        self.logger.info("Start training...")

        # DATA LOADER Initialization
        data_class = DataSetup(self.cfg, self.data, self.use_ddp, self.rank, self.world_size)
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = data_class.get_loaders(batch_size)

        self.logger.info(f"Train Loader size: {len(self.train_loader.dataset)}, Val Loader size: {len(self.val_loader.dataset)}")
        self.logger.info(f"Train Loader: {len(self.train_loader)}, Val Loader: {len(self.val_loader)}")

        output_path = os.path.join(self.checkpoint_dir, self.model.__class__.__name__) + '_train.pth'
        iterator = tqdm(range(start_epoch, epochs + start_epoch), dynamic_ncols=True)

        self.logger.info(f"Saving checkpoint: {output_path}")

        if self.is_main_process(self.rank):
            if self.wandb:
                self.wandb.init()

        for epoch in iterator:

            if self.use_ddp:
                self.train_sampler.set_epoch(epoch)

            output_train = self._train_step()
            output_val = self._validation_step()

            if self.is_main_process(self.rank):
                if self.wandb:
                    self.wandb.log(output_train, output_val, epoch)
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(output_train["total_loss"].item())
                self.history["val_loss"].append(output_val["total_loss"].item())
                self.history["val_mAP"].append(output_val["metrics"]["map"].numpy().item())
                self.history["val_mAP@50"].append(output_val["metrics"]["map_50"].numpy().item())

                self._end_epoch_verbose(iterator, epoch, output_train, output_val)

                # Save model based on best validation mAP@50.
                best_acc = max(self.history["val_mAP"])
                current_acc = output_val["metrics"]["map"].numpy()

                if current_acc >= best_acc:
                    model_to_save = self.model.module if self.use_ddp else self.model
                    self.save_checkpoint(output_path,  model_to_save, epoch, self.cfg)

            if self.scheduler:
                self.scheduler.step()

        if self.is_main_process(self.rank):
            if self.wandb:
                self.wandb.finish()

        if self.use_ddp:
            self.cleanup_ddp()

        return self.history

    def _initialize_trainer(self):

        # Initialize Directory
        checkpoint_path, version = initialize_directory(self.cfg)
        self.checkpoint_dir = Path(checkpoint_path)
        self.version = version

        self.logger.info(f"Checkpoint Directory: {self.checkpoint_dir}")
        self.logger.info(f"Version: {self.version}")

        self.use_ddp = self.is_distributed()

        if self.use_ddp:
            self.rank, self.world_size, self.local_rank = self.setup_ddp()
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(self.cfg.experiment.train.seed + self.rank)
        self.logger.info(f"Using device: {self.device}, DDP: {self.use_ddp}, Rank: {self.rank}, World Size: {self.world_size}")

        # MODEL setup for DDP
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            self.logger.info(f'DDP: {self.model}')

        # OPTIMIZER Initialization
        optimizer_class = OptimizerSetup(self.cfg, model=self.model)
        self.optimizer, self.scheduler = optimizer_class.get_optimizer()
        self.logger.info(f'Optimizer: {self.optimizer}, Scheduler: {self.scheduler}')

        # LOSS Initialization
        self.criterion = Loss(self.cfg)
        self.logger.info(f'Criterion: {self.criterion}')

        # METRIC Initialization
        self.metric = Metric(self.cfg, self.device)
        self.logger.info(f'Metric: {self.metric}')

        # Initialize WandB
        if self.cfg.experiment.train.use_wandb:
            self.wandb = Wandb(self.cfg, self.logger)

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def is_distributed(self):
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ

    def setup_ddp(self):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank

    def cleanup_ddp(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def is_main_process(self, rank: int):
        return rank == 0

    def reduce_mean(self, value: float, device: torch.device, world_size: int):
        t = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / world_size).item()

    def save_checkpoint(self, path: str, model: nn.Module, epoch: int, cfg: DictConfig):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "epoch": epoch,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            path,
        )

    def _end_epoch_verbose(self, iterator, epoch, output_train, output_test):

        val_map_50 = float(output_test["metrics"]["map_50"])
        train_loss = output_train["total_loss"]
        test_loss = output_test["total_loss"]

        iterator.set_description(
            f"epoch: {epoch+1}, val_mAP@50: {val_map_50:.3f}, train_loss: {train_loss:.3f}, test_loss: {test_loss:.3f}"
        )

    def _train_step(self):
        self.model.train()

        iterator = tqdm(self.train_loader, dynamic_ncols=True)

        cls_loss_avg = []
        loc_loss_avg = []
        total_loss_avg = []

        for i, batch_sample in enumerate(iterator):
            self.optimizer.zero_grad()
            image_batch = torch.stack(batch_sample[0]).to(self.device)
            box_targets = torch.stack(batch_sample[3]).to(self.device)
            cls_targets = torch.stack(batch_sample[4]).to(self.device)

            pred_boxes, pred_labels = self.model(image_batch)

            pred = (pred_boxes, pred_labels)
            y_true = (box_targets, cls_targets)
            loss = self.criterion(y_true, pred)
            total_loss = loss["total_loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            optimizer_lr = self.optimizer.param_groups[0]["lr"]

            loc_loss = loss["loc_loss"].item()
            cls_loss = loss["cls_loss"].item()
            total_loss = total_loss.item()
            if self.use_ddp:
                loc_loss = self.reduce_mean(loc_loss, self.device, self.world_size)
                cls_loss = self.reduce_mean(cls_loss, self.device, self.world_size)
                total_loss = self.reduce_mean(total_loss, self.device, self.world_size)

            cls_loss_avg.append(loc_loss)
            loc_loss_avg.append(cls_loss)
            total_loss_avg.append(total_loss)

            status = f"[Train][{i+1}] Total Loss: {np.mean(total_loss_avg):.4f}, "
            status += f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "
            status += f"LR: {optimizer_lr:.3f}"

            iterator.set_description(status)

        return {"loc_loss": np.mean(loc_loss_avg), "cls_loss": np.mean(cls_loss_avg),
                "total_loss": np.mean(total_loss_avg)}

    def _validation_step(self):

        self.model.eval()

        iterator = tqdm(self.val_loader, dynamic_ncols=True)

        cls_loss_avg = []
        loc_loss_avg = []
        total_loss_avg = []

        self.metric.reset()

        for i, batch_sample in enumerate(iterator):

            image_batch = torch.stack(batch_sample[0]).to(self.device)
            box_targets = torch.stack(batch_sample[3]).to(self.device)
            cls_targets = torch.stack(batch_sample[4]).to(self.device)

            y_true = (box_targets, cls_targets)
            nms_threshold = self.cfg.model.nms_threshold if self.data_encoder else None
            score_threshold = self.cfg.model.score_threshold if self.data_encoder else None
            results = self.wrapper.predict(image_batch,
                                         criterion=self.criterion,
                                         y_true=y_true,
                                         device=self.device,
                                         nms_threshold=nms_threshold,
                                         score_threshold=score_threshold)

            predictions = results["predictions"]
            loc_loss = results["loc_loss"]
            cls_loss = results["cls_loss"]
            total_loss = results["total_loss"]

            if self.use_ddp:
                loc_loss = self.reduce_mean(loc_loss, self.device, self.world_size)
                cls_loss = self.reduce_mean(cls_loss, self.device, self.world_size)
                total_loss = self.reduce_mean(total_loss, self.device, self.world_size)

            cls_loss_avg.append(cls_loss)
            loc_loss_avg.append(loc_loss)
            total_loss_avg.append(total_loss)

            # Prepare targets and predictions for evaluations.
            targets = []
            for idx, (box_raw, label_raw) in enumerate(zip(batch_sample[1], batch_sample[2])):
                boxes_raw_per_image = box_raw.to(self.device)
                labels_raw_per_image = label_raw.to(self.device)

                target_dict = dict(
                    boxes=boxes_raw_per_image,
                    labels=labels_raw_per_image
                )

                targets.append(target_dict)

            self.metric.update(predictions, targets)

            status = f"[Validation][{i+1}] Total Loss: {np.mean(total_loss_avg):.4f}, "
            status += f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "

            iterator.set_description(status)

        metrics_dict = self.metric.compute()

        map_50 = float(metrics_dict["map_50"])
        status += f"val_mAP@50: {map_50:.3f}"

        iterator.set_description(status)

        output = {"loc_loss": np.mean(loc_loss_avg), "cls_loss": np.mean(cls_loss_avg),
                  "total_loss": np.mean(total_loss_avg),
                  "metrics": metrics_dict}
        return output
