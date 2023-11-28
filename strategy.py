from abc import ABC, abstractmethod


class ModelStrategy(ABC):
    @abstractmethod
    def classify_objects(input_video):
        pass


