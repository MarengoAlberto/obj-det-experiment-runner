import hydra
from omegaconf import DictConfig, OmegaConf

from models import Model

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.model.name == 'yolo':
        cfg.dataset.names = [cls for cls in cfg.dataset.names if cls != '__background__']
        cfg.dataset.nc = len(cfg.dataset.names)
    print(cfg)
    print('-------------------')
    print(OmegaConf.to_yaml(cfg))
    print(cfg.model)
    model = Model(cfg, load_model=False)
    history = model.train(data='configs/dataset/data.yaml',
                          n_epochs=cfg.experiment.train.epochs,
                          batch_size=cfg.experiment.train.batch_size)
    print(history)

if __name__ == "__main__":
    main()
