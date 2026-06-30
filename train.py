import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from models import Model

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = Model(cfg, load_model=False)
    choices = HydraConfig.get().runtime.choices
    dataset_name = choices["dataset"]

    dataset_yaml_path = f"configs/dataset/{dataset_name}.yaml"

    print(dataset_yaml_path)

    history = model.train(data=dataset_yaml_path,
                          n_epochs=cfg.experiment.train.epochs,
                          batch_size=cfg.experiment.train.batch_size)
    print(history)

if __name__ == "__main__":
    main()
