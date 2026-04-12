import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print('-------------------')
    print(OmegaConf.to_yaml(cfg))
    print(cfg.model)

if __name__ == "__main__":
    main()
