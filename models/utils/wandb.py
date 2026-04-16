import os
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

from . import utils
from .logger import get_logger

load_dotenv()

wandb_key = os.getenv('WANDB_KEY')

class Wandb:

    run=None

    def __init__(self, cfg, logger=None, close_when_done=False):
        self.cfg = cfg
        self.logger = logger if logger else get_logger()
        self.close_when_done = close_when_done

    def init(self):
        try:
            wandb.login(key=wandb_key)
            self.run = wandb.init(
                project=self.cfg.project.name,
                mode="online" if self.cfg.experiment.train.use_wandb else "disabled",
                config=self.get_config(),
            )
        except Exception as e:
            self.logger.warning("Failed to initialize Weights & Biases. Please check your WANDB_KEY and internet connection. Error: {}".format(e))

    def log(self, data):

        try:
            output_train = utils.refactor_dict(data.get('output_train', {}), 'train')
            output_val = utils.refactor_dict(data.get('output_val', {}), 'val')
            eval_metrics = utils.refactor_dict(output_val.get('val_metrics', {}), 'val')
            coco_eval_results = utils.refactor_dict(data.get('coco_eval_results', {}), 'coco_eval')
            epoch = data.get('epoch', None)
            epochs = {'epoch': epoch} if epoch is not None else {}
            metrics = utils.merge_metric_dicts(epochs, output_train, output_val, eval_metrics, coco_eval_results)
            self.run.log(metrics)
        except Exception as e:
            self.logger.warning("Failed to log metrics to Weights & Biases. Error: {}".format(e))

    def finish(self):
        try:
            self.run.finish()
        except Exception as e:
            self.logger.warning("Failed to finish Weights & Biases run. Error: {}".format(e))

    def get_config(self):
        return OmegaConf.to_container(
            self.cfg,
            resolve=True,
            throw_on_missing=True,
        )
