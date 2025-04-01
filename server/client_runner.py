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
    def __init__(self, configs:RunnerConfiguration):
        super().__init__(configs)
        # TODO: ISTO DEVE SER UMA CONFIG PADRÃO:
        self.pipeline = PipelineFactory.create_pipeline(self.config.pipeline_variation)
        self.queue = multiprocessing.Queue()
        self.can_run = False

    def run(self):
        asyncio.run(self.run_client())

    async def process_tasks(self, uri):
        async with websockets.connect(uri, ping_interval=5) as websocket:
            print("Conectado ao servidor")
            video = []
            result = None
            p = None

            while True:

                # Se houver um processo em execução, aguarde até que ele termine
                if p is not None:
                    if not p.is_alive():
                        print("Pronto")
                        p.join()  # Aguarda a finalização do processo
                        result = self.queue.get()  # Obtém o resultado da fila
                        print(f"Resultado do processamento: {result}")

                        # Enviar os dados para o servidor
                        json_data = json.dumps(result)
                        await websocket.send(json_data)
                        print(f"Resultado enviado:\n{json_data}")
                        # Resetar variáveis para processar um novo vídeo
                        video = []
                        self.can_run = False
                        p = None  # Resetar `p` para permitir um novo processo

                    continue  # Volta ao loop para continuar a execução

                # Coletar os frames do WebSocket
                frame_bytes = await websocket.recv()

                if not isinstance(frame_bytes, str):
                    if len(video) == 1:
                        print("Coletando dados da tarefa.")

                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    video.append(frame)

                else:
                    print("Dados coletados. Iniciando processamento para detecção dos rótulos")

                    # Criando um arquivo temporário para salvar o vídeo
                    with tempfile.NamedTemporaryFile('w+b', suffix=".mp4", delete=False) as temp_file:
                        height, width, _ = video[0].shape
                        video_writer = cv2.VideoWriter(
                            temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height)
                        )

                        if not self.can_run:
                            for frame in video:
                                video_writer.write(frame)
                            video_writer.release()
                            temp_file.seek(0)

                            # Iniciando um novo processo
                            p = Process(target=run, kwargs={
                                'pipeline': self.pipeline,
                                'video_path': temp_file.name,
                                'test_name': str(frame_bytes),
                                'queue': self.queue
                            })
                            p.start()

    async def run_client(self):
        host = os.getenv('HOST', 'localhost')
        port = os.getenv('PORT', '8765')
        uri = f"ws://{host}:{port}"  # Endereço do servidor WebSocket
        print(f"Conectando a {uri}")

        while True:
            try:
                await self.process_tasks(uri)
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode, OSError) as e:
                print(f"Erro de conexão: {e}, tentando novamente em 5 segundos...")
                await asyncio.sleep(5)
