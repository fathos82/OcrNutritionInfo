from project.pandas_utils import *
from project.runner import Runner
import asyncio
import json
import os
from enum import verify

import cv2
import websockets

from project.runner_configuration import RunnerConfiguration


class ServerApplicationRunner(Runner):
    def __init__(self,  config: RunnerConfiguration):
        super().__init__(config)
        self.task_queue = asyncio.Queue()
        self.base_path = 'res/videos'

    def run(self):
        asyncio.run(self.start_server())

    # Função para agendar as tarefas (ler os vídeos e adicioná-los à fila)
    async def schedule_tasks(self):
        paths = os.listdir(self.base_path)
        for video_file in paths:
            if not video_file.endswith('.mp4') or contains_register(get_name_from_path(video_file)):
                continue
            video_path = os.path.join(self.base_path, video_file)
            await self.schedule_task(self.base_path + "/" + video_file)

    # Função para adicionar a tarefa à fila
    async def schedule_task(self,video_file):
        video_path = os.path.join(self.base_path, video_file)

        await self.task_queue.put(video_file)
        # print(f"Task added: {video_file}")

    async def start_server(self):
        host = os.environ.get("HOST", "0.0.0.0")
        port = os.getenv('PORT', 8765)
        print("Escalonando tarefas...")
        await self.schedule_tasks()
        print(f"{self.task_queue.qsize()} tarefas encontradas.")
        server = await websockets.serve(self.send_task, host, port, max_size=None)  # 10 MB
        print(f"Servidor iniciado em ws://{host}:{port}")
        await server.wait_closed()  # Aguarda até o servidor ser fechado
        asyncio.run(self.start_server())

    async def send_task(self, websocket):
        print(f"Cliente conectado: {websocket.remote_address}")
        task = None
        resolving_task = False
        try:
            while not self.task_queue.empty():
                task = await self.task_queue.get()
                print(f"Enviando task '{task}' para o cliente {websocket.remote_address}.")
                cap = cv2.VideoCapture(task)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Fim do vídeo

                    # Converte o frame para JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    # Envia o frame
                    await websocket.send(frame_bytes)
                    await asyncio.sleep(0.01)  # Pequeno delay para evitar congestionamento
                await websocket.send(task)
                print(f"Dados da tarefa task '{task} foram enviadas para cliente {websocket.remote_address}")
                resolving_task = True
                result = await asyncio.wait_for(websocket.recv(), timeout=60 * 10)
                print(self.config.processing_name)

                save_or_update_table(json.loads(result),file_name=self.config.processing_name)
                resolving_task = False
                print(f"Resposta recebida de {websocket.remote_address} referente a {task}.")
                self.task_queue.task_done()
        except (Exception, asyncio.TimeoutError) as e:
            print(f"Erro na conexão com {websocket.remote_address}: {e}")
        finally:
            if resolving_task:
                print(f"Infelizmente o cliente de endereço {websocket.remote_address} esta inapto a concluir a tarefa.")
                print(f"Reincluindo tarefa {task} para lista de tarefas")
                await self.task_queue.put(task)

            print(f"Conexão encerrada com {websocket.remote_address}")









