"""Red neuronal para clasificar fotos de ropa en alta resolución.

Usa ResNet18 con pesos preentrenados en ImageNet (transfer learning) y le
cambia la última capa para que tenga una salida por cada clase de ropa.

>>> PARA CAMBIAR LA RESOLUCIÓN, edita TAM aquí y vuelve a entrenar. <<<
(224 es el tamaño estándar de ResNet; prueba después con 300.)
"""

from torch import nn
from torchvision import models

TAM = 224  # las imágenes se redimensionan a TAM x TAM píxeles (a color)

# Valores de normalización de ImageNet, con los que se preentrenó ResNet
MEDIA = (0.485, 0.456, 0.406)
DESVIACION = (0.229, 0.224, 0.225)


def crear_modelo(n_clases, preentrenado=True):
    """Crea ResNet18 con una capa final nueva de n_clases salidas.

    preentrenado=True descarga los pesos de ImageNet (solo hace falta al entrenar;
    el servidor usa False porque carga sus propios pesos ya entrenados).
    """
    pesos = models.ResNet18_Weights.DEFAULT if preentrenado else None
    modelo = models.resnet18(weights=pesos)
    modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)
    return modelo