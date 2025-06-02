from abc import ABC, abstractmethod

class BaseProblem(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def evaluate(self, individual):
        pass