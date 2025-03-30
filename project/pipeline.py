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
        self.original_image = image_data

    def get_original_image(self):
        return copy(getattr(self.original_image, 'image', None))

    @abstractmethod
    def run(self, input_data: I) -> O:
        pass

class Pipeline:
    def __init__(self):
        self.passes: List[DetectionPass] = []
        self.original_image=None


    def add_passes(self, *detection_passes: Union[Type[DetectionPass], DetectionPass]):
        for detection_pass in detection_passes:
            if isinstance(detection_pass, type):
                self.passes.append(detection_pass())
            else:
                self.passes.append(detection_pass)
        return self

    @profile
    def run(self, image_data: IOComponent) -> IOComponent:
        current_input = image_data
        self.original_image = image_data
        for detection_pass in self.passes:
            detection_pass.set_original_image(self.original_image)
            current_input = detection_pass.run(current_input)

        return current_input

