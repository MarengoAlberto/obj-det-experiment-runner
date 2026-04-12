from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Metric:
    def __init__(self, cfg):
        if cfg.metric == 'torch_mAP':
            self.metric = MeanAveragePrecision(class_metrics=True).to(cfg.device)
        else:
            raise NotImplementedError(f"Metric {cfg.metric} not implemented yet.")

    def update(self, y_true, y_pred):
        return self.metric(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
