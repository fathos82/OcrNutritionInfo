import os

import cv2

from structure.pandas_utils import contains_register, get_name_from_path, save_or_update_table
from structure.pipeline_factory import PipelineFactory, PipelineVariation
from structure.run_pipeline import run
from structure.runner import Runner


class AppRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.base_path= 'res/videos'
    def load_videos_path(self):
        video_paths = []
        paths = os.listdir(self.base_path)
        for video_file in paths:
            if not video_file.endswith('.mp4') or contains_register(get_name_from_path(video_file)):
                continue
            video_path = os.path.join(self.base_path, video_file)
            video_paths.append(video_path)
        return video_paths

    def run(self):
        video_paths = self.load_videos_path()
        pipeline = PipelineFactory.create_pipeline(PipelineVariation.PIPELINE_1)
        for video_path in video_paths:
            print('Processing video {}'.format(video_path))
            result = run(video_path=video_path,pipeline=pipeline)
            save_or_update_table(result, self.config.processing_name)
    def run_pipeline(self, video_path):
        cap = cv2.VideoCapture(video_path)

        # while True:

