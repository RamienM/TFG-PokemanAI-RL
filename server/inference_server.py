import multiprocessing as mp
from models.segmentation.STELLE_Pokemon_Segmentation.inference.inferencer import STELLEInferencer

def inference_process(request_queue):
    """
    Servidor de inferencia: un solo proceso, carga STELLEInferencer una vez.
    Espera tuplas (uid, np_image, response_queue), procesa y devuelve en response_queue.
    """
    print("[Inferencia] Cargando modelo...")
    inferencer = STELLEInferencer()
    print("[Inferencia] Modelo cargado.")

    while True:
        uid, image, resp_q = request_queue.get()
        # STOP: uid None
        if uid is None:
            break
        # run inference
        result = inferencer.predict(image)
        # devolver al cliente específico
        resp_q.put((uid, result))