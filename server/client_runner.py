import multiprocessing
import os
import tempfile
import json
import asyncio
import websockets
import cv2
import numpy as np

from project.pipeline_factory import PipelineFactory
from project.run_pipeline import run
from project.runner import Runner
from multiprocessing import Process
from project.runner_configuration import RunnerConfiguration


class ClientRunner(Runner):
    def __init__(self, configs: RunnerConfiguration):
        super().__init__(configs)
        self.pipeline = PipelineFactory.create_pipeline(self.config.pipeline_variation)
        self.queue = multiprocessing.Queue()
        self.process = None
        self.video_frames = []

    def run(self):
        asyncio.run(self.run_client())

    async def process_tasks(self, websocket):
        print("Conectado ao servidor")

        while True:
            if self.process and not self.process.is_alive():
                await self.handle_completed_process(websocket)
                continue

            frame_bytes = await websocket.recv()
            if isinstance(frame_bytes, str):
                await self.start_processing(frame_bytes)
            else:
                self.collect_frame(frame_bytes)

    async def handle_completed_process(self, websocket):
        self.process.join()
        result = self.queue.get()
        await websocket.send(json.dumps(result))
        print(f"Resultado enviado: {result}")
        self.reset_state()

    async def start_processing(self, task_name):
        print("Iniciando processamento de vídeo")
        video_path = self.save_video()
        self.process = Process(target=run, args=(self.pipeline,), kwargs={
            'video_path': video_path,
            'test_name': task_name,
            'queue': self.queue
        })
        self.process.start()

    def collect_frame(self, frame_bytes):
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.video_frames.append(frame)
        print(f"Frame recebido. Total: {len(self.video_frames)} frames")

    def save_video(self):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        height, width, _ = self.video_frames[0].shape
        writer = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        for frame in self.video_frames:
            writer.write(frame)
        writer.release()

        return temp_file.name

    def reset_state(self):
        self.video_frames.clear()
        self.process = None

    async def run_client(self):
        uri = f"ws://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8765')}"
        print(f"Conectando a {uri}")

        while True:
            try:
                async with websockets.connect(uri, ping_interval=5) as websocket:
                    await self.process_tasks(websocket)
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode, OSError) as e:
                print(f"⚠ Erro de conexão: {e}. Tentando novamente em 5 segundos...")
                await asyncio.sleep(5)