from .fpn import FPNModel

class Model:
    def __init__(self, cfg, load_model=True, *args, **kwargs):
        if cfg.model.name == "fpn":
            self.wrapper = FPNModel(cfg, load_model, *args, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {cfg.model.name}")

    def train(self, *args, **kwargs):
        return self.wrapper.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.wrapper.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.wrapper.evaluate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.wrapper(*args, **kwargs)
