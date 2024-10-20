from abc import ABC, abstractmethod
from typing import List, Type, TypeVar, Generic


class IOComponent(ABC):
    pass
I = TypeVar('I', bound=IOComponent)
O = TypeVar('O', bound=IOComponent)
class DetectionPass(ABC, Generic[I, O]):
    @abstractmethod
    def run(self, input_data: I) -> O:
        pass



class Pipeline:
    def __init__(self):
        self.passes:List[DetectionPass] = []

    def add_passes(self, *detection_pass_types: Type[DetectionPass]):
        for detection_pass_type in detection_pass_types:
            self.passes.append(detection_pass_type())
        return self

    def run(self, start_input:IOComponent):
        current_input = start_input
        for detection_pass in self.passes:
            current_input = detection_pass.run(current_input)
        


