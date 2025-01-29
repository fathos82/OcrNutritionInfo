from abc import ABC, abstractmethod
from copy import copy
from typing import List, Type, TypeVar, Generic, Union

class IOComponent(ABC):
    pass

I = TypeVar('I', bound=IOComponent)
O = TypeVar('O', bound=IOComponent)

class DetectionPass(ABC, Generic[I, O]):
    def __init__(self):
        self.original_image = None

    def set_original_image(self, image_data: IOComponent):
        self.original_image = getattr(image_data, "image", None)

    def get_original_image(self):
        return copy(self.original_image)

    @abstractmethod
    def run(self, input_data: I) -> O:
        pass

class Pipeline:
    def __init__(self):
        self.passes: List[DetectionPass] = []

    def add_passes(self, *detection_passes: Union[Type[DetectionPass], DetectionPass]):
        for detection_pass in detection_passes:
            if isinstance(detection_pass, type):
                self.passes.append(detection_pass())
            else:
                self.passes.append(detection_pass)
        return self

    def run(self, image_data: IOComponent) -> IOComponent:
        current_input = image_data
        original_image = image_data
        for detection_pass in self.passes:
            detection_pass.set_original_image(original_image)
            current_input = detection_pass.run(current_input)

        # self.passes.clear()  # Limpa os passes após execução
        return current_input
