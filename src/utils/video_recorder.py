import cv2
import os
import numpy as np

class VideoRecorder:
    def __init__(self, frame_width=160, frame_height=144, fps=30, flush_every=10000, save_path="game&results/Pokemon_red_env/videos"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.flush_every = flush_every
        self.save_path = save_path

        self.screen_video_writer = None
        self.maps_video_writer = None

        self.frame_count = 0
        self.part_number = 0  # Para guardar en partes numeradas

        # Asegúrate de que la ruta exista
        os.makedirs(self.save_path, exist_ok=True)

    def _start_new_writers(self):
        screen_path = os.path.join(self.save_path, f"screen_video_{self.part_number:03}.mp4")
        maps_path = os.path.join(self.save_path, f"maps_video_{self.part_number:03}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.screen_video_writer = cv2.VideoWriter(screen_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.maps_video_writer = cv2.VideoWriter(maps_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def start_writing(self):
        self._start_new_writers()

    def _release_writers(self):
        if self.screen_video_writer:
            self.screen_video_writer.release()
            self.screen_video_writer = None
        if self.maps_video_writer:
            self.maps_video_writer.release()
            self.maps_video_writer = None

    def save_videos(self, screen, maps):
        if self.screen_video_writer is None or self.maps_video_writer is None:
            raise RuntimeError("Los videos no se han iniciado. Llama a start_writing() primero.")

        # Convertir screen a BGR si está en escala de grises (con un solo canal)
        if len(screen.shape) == 3 and screen.shape[2] == 1:
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
        elif len(screen.shape) == 2:  # Si screen tiene solo 2 dimensiones (escala de grises)
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)


        maps_uint8 = (maps * 255).astype(np.uint8)
        maps_uint8 = cv2.cvtColor(maps_uint8, cv2.COLOR_RGB2BGR)

        # Escribir los frames en los videos
        self.screen_video_writer.write(screen)
        self.maps_video_writer.write(maps_uint8)

        self.frame_count += 1
        if self.frame_count >= self.flush_every:
            self._release_writers()
            self.part_number += 1
            self._start_new_writers()
            self.frame_count = 0

    def stop_writing(self):
        self._release_writers()
