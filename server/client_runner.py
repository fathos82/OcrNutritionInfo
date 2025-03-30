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


class ClientRunner(Runner):
    def run(self):
        asyncio.run(self.run_client())

    async def process_tasks(self, uri):
        async with websockets.connect(uri, ping_interval=None) as websocket:
            print("Conectado ao servidor")
            video = []

            while True:
                    # Recebe a tarefa do servidor
                frame_bytes = await websocket.recv()

                # Se os dados recebidos não forem uma string, processamos como um frame
                if not isinstance(frame_bytes, str):
                    if len(video) == 1:
                        print("Coletando dados da tarefa.")

                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    video.append(frame)

                else:
                    print("Dados coletados.")
                    print("Iniciando processamento para detecção dos rótulos")

                    # Criando um arquivo temporário para salvar o vídeo
                    with tempfile.NamedTemporaryFile('w+b', suffix=".mp4", delete=True) as temp_file:
                        height, width, _ = video[0].shape
                        video_writer = cv2.VideoWriter(
                            temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height)
                        )

                        for frame in video:
                            video_writer.write(frame)

                        video_writer.release()
                        temp_file.seek(0)

                        # Executando a pipeline de processamento
                        pipeline = PipelineFactory.create_pipeline(self.config.pipeline_variation)
                        data = await run(
                            pipeline=pipeline,
                            video_path=temp_file.name,
                            test_name=frame_bytes,
                            socket=websocket
                        )

                        print("Detecções finalizadas.\nEnviando dados para o servidor.")
                        json_data = json.dumps(data)
                        video = []

                        print("Enviando dados para o servidor.")
                        await websocket.send(json_data)
                        print(f"Resultado enviado:\n{json_data}")


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
