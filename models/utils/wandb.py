import os
import wandb
from dotenv import load_dotenv

from .logger import get_logger

load_dotenv()

wandb_key = os.getenv('WANDB_KEY')

class Wandb:

    run=None

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger if logger else get_logger()

    def init(self):
        try:
            wandb.login(key=wandb_key)
            self.run = wandb.init(
                entity=self.cfg.entity,
                project=self.cfg.project.name,
                mode="online" if self.cfg.experiment.train.use_wandb else "disabled",
                config=self.get_config(),
            )
        except:
            self.logger.warning("Failed to initialize Weights & Biases. Please check your WANDB_KEY and internet connection.")

    def log(self, output_train, output_val, epoch):
        try:
            metrics = {
                'epoch': epoch,
            }
            self.run.log(metrics)
        except:
            self.logger.warning("Failed to log metrics to Weights & Biases.")

    def finish(self):
        try:
            self.run.finish()
        except:
            self.logger.warning("Failed to finish Weights & Biases run.")

    def get_config(self):
        return {}
