from abc import ABC, abstractmethod
from typing import List, Type, TypeVar, Generic

from project.utils.image_data import ImageData


class IOComponent(ABC):
    pass
I = TypeVar('I', bound=IOComponent)
O = TypeVar('O', bound=IOComponent)
class DetectionPass(ABC, Generic[I, O]):
    def __init__(self):
        self.original_image = None

    def set_original_image(self, image_data: ImageData):
        self.original_image = image_data.image
    def get_original_image(self):

        return self.get_original_image()
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

    def run(self, image_data:ImageData):
        current_input = image_data
        original_image = image_data
        for detection_pass in self.passes:
            detection_pass.set_original_image(original_image)
            current_input = detection_pass.run(current_input)

        #TODO: CLEAR PASSES
        


