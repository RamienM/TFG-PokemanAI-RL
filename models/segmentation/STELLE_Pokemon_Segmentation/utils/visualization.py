import numpy as np

COLOR_DICT = {
    'fondo': (0, 0, 0), 'captura': (0, 255, 0), 'leer': (255, 255, 0),
    'main': (255, 0, 0),'map': (125,125,125), 'mo': (0, 0, 255), 'no_caminable': (255, 255, 255),
    'npc': (255, 0, 255), 'poke_ball': (0, 255, 255), 'puerta': (255, 165, 0),
    'recuperar': (0, 128, 0), 'salto': (0, 0, 128), 'seleccionado': (139, 0, 139),
    'texto': (160, 82, 45),
}
CLASS_NAMES = list(COLOR_DICT.keys())
COLORS = np.array(list(COLOR_DICT.values())) / 255.0

def apply_overlay(np_image, prediction):
    mask_rgb = np.zeros((*prediction.shape, 3), dtype=np.float32)

    for class_id, color in enumerate(COLORS):
        if class_id == 0:
            continue
        mask_rgb[prediction == class_id] = color

    if np_image.ndim == 3 and np_image.shape[-1] == 1:
        np_image = np_image.squeeze(-1)
    img_rgb = np.stack([np_image / 255.0] * 3, axis=-1)

    alpha = 0.5
    return (1 - alpha) * img_rgb + alpha * mask_rgb
