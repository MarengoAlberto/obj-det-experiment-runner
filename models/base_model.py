from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)