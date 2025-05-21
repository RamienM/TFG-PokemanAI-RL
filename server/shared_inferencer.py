import uuid

class SharedInferencer:
    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue

    def predict(self, np_image):
        # Cada llamada usa un UID único y su propia cola de respuesta
        uid = str(uuid.uuid4())
        # Enviar: (uid, imagen, cola_para_este_cliente)
        self.request_queue.put((uid, np_image, self.response_queue))
        # Esperar solo en esta cola
        result_uid, output = self.response_queue.get()
        return output
