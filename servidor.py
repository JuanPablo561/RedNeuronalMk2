"""Servidor web: recibe una foto, la pasa por la red y devuelve las probabilidades.

Uso: python servidor.py   y abre http://127.0.0.1:8000
Antes debes haber entrenado con: python entrenar.py
"""

import base64
import io
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from torchvision import transforms

from modelo import DESVIACION, MEDIA, crear_modelo

CARPETA = Path(__file__).parent
RUTA_MODELO = CARPETA / "ropa_hd.pt"

if not RUTA_MODELO.exists():
    raise SystemExit("No se encontró ropa_hd.pt. Primero ejecuta: python entrenar.py")

# El archivo guarda los pesos, los nombres de las clases y la resolución usada
punto = torch.load(RUTA_MODELO, map_location="cpu", weights_only=True)
CLASES = punto["clases"]
TAM = punto["tam"]

modelo = crear_modelo(len(CLASES), preentrenado=False)
modelo.load_state_dict(punto["estado"])
modelo.eval()

redimensionar = transforms.Resize((TAM, TAM))
a_tensor = transforms.ToTensor()
normalizar = transforms.Normalize(MEDIA, DESVIACION)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # fotos de hasta 15 MB


@app.errorhandler(413)
def demasiado_grande(_):
    return jsonify(error="La imagen es demasiado grande (máximo 15 MB)."), 413


@app.get("/")
def inicio():
    return send_from_directory(CARPETA, "index.html")


@app.post("/predecir")
def predecir():
    archivo = request.files.get("imagen")
    if archivo is None:
        return jsonify(error="Falta la imagen."), 400

    try:
        img = Image.open(archivo.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")  # respeta la rotación del celular
    except Exception:
        return jsonify(error="No se pudo leer la imagen."), 400

    img = redimensionar(img)

    # La misma imagen que ve la red, para mostrarla en la página
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    vista = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    x = normalizar(a_tensor(img))[None]  # añade la dimensión del lote: 1 x 3 x TAM x TAM
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(x), dim=1)[0]

    indice = int(probabilidades.argmax())
    return jsonify(
        indice=indice,
        clase=CLASES[indice],
        tam=TAM,
        vista=vista,
        probabilidades=[
            {"clase": c, "p": round(p, 4)} for c, p in zip(CLASES, probabilidades.tolist())
        ],
    )


if __name__ == "__main__":
    print(f"Modelo cargado ({len(CLASES)} clases, {TAM}x{TAM} píxeles)")
    print("Abre http://127.0.0.1:8000 en tu navegador")
    app.run(host="127.0.0.1", port=8000)