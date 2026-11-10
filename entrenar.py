"""Entrena el clasificador de ropa con transfer learning (ResNet18).

Requisito: tener la carpeta "clothing-dataset-small" junto a este archivo:
    git clone https://github.com/alexeygrigorev/clothing-dataset-small
con subcarpetas train/, validation/ (y opcionalmente test/), una carpeta por clase.

Uso: python entrenar.py
Guarda el mejor modelo en ropa_hd.pt
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import DESVIACION, MEDIA, TAM, crear_modelo

CARPETA = Path(__file__).parent
CARPETA_DATOS = CARPETA / "clothing-dataset-small"
RUTA_MODELO = CARPETA / "ropa_hd.pt"

EPOCAS_FASE1 = 3   # fase 1: solo se entrena la capa final
EPOCAS_FASE2 = 7   # fase 2: se afina toda la red
TAMANO_LOTE = 32


def main():
    for sub in ("train", "validation"):
        if not (CARPETA_DATOS / sub).is_dir():
            raise SystemExit(
                f"No encuentro {CARPETA_DATOS / sub}\n"
                "Descarga el dataset con:\n"
                "  git clone https://github.com/alexeygrigorev/clothing-dataset-small"
            )

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {dispositivo} | resolución: {TAM}x{TAM}")

    # Aumento de datos para fotos: recortes, espejo, giros leves y cambios de color
    transformar_entrenamiento = transforms.Compose([
        transforms.RandomResizedCrop(TAM, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIACION),
    ])
    transformar_evaluacion = transforms.Compose([
        transforms.Resize((TAM, TAM)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIACION),
    ])

    entrenamiento = datasets.ImageFolder(CARPETA_DATOS / "train", transformar_entrenamiento)
    validacion = datasets.ImageFolder(CARPETA_DATOS / "validation", transformar_evaluacion)
    clases = entrenamiento.classes
    print(f"Clases ({len(clases)}): {clases}")
    print(f"Imágenes: {len(entrenamiento)} entrenamiento, {len(validacion)} validación\n")

    cargador_ent = DataLoader(entrenamiento, batch_size=TAMANO_LOTE, shuffle=True)
    cargador_val = DataLoader(validacion, batch_size=64)

    modelo = crear_modelo(len(clases), preentrenado=True).to(dispositivo)
    funcion_perdida = nn.CrossEntropyLoss()

    def evaluar(cargador):
        modelo.eval()
        correctas = total = 0
        with torch.no_grad():
            for imagenes, etiquetas in cargador:
                imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
                correctas += (modelo(imagenes).argmax(dim=1) == etiquetas).sum().item()
                total += len(etiquetas)
        return correctas / total

    mejor = 0.0

    def correr_fase(nombre, epocas, optimizador, programador, congelada):
        nonlocal mejor
        for epoca in range(1, epocas + 1):
            inicio = time.time()
            modelo.train()
            if congelada:
                modelo.eval()      # la base queda fija (también sus BatchNorm)
                modelo.fc.train()
            for i, (imagenes, etiquetas) in enumerate(cargador_ent):
                imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
                perdida = funcion_perdida(modelo(imagenes), etiquetas)
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()
                if i % 25 == 0:
                    print(f"[{nombre}] época {epoca}/{epocas} | lote {i:3d} | error: {perdida.item():.4f}")
            if programador:
                programador.step()
            precision = evaluar(cargador_val)
            print(f"==> [{nombre}] época {epoca} | validación: {precision:.2%} | {time.time() - inicio:.0f} s")
            if precision > mejor:
                mejor = precision
                torch.save({"estado": modelo.state_dict(), "clases": clases, "tam": TAM}, RUTA_MODELO)
                print("    (nuevo mejor modelo guardado)")
            print()

    # ---- Fase 1: congelar la base, entrenar solo la capa final ----
    for p in modelo.parameters():
        p.requires_grad = False
    for p in modelo.fc.parameters():
        p.requires_grad = True
    opt1 = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)
    correr_fase("fase 1", EPOCAS_FASE1, opt1, None, congelada=True)

    # ---- Fase 2: descongelar todo y afinar con tasa baja ----
    for p in modelo.parameters():
        p.requires_grad = True
    opt2 = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    prog2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCAS_FASE2)
    correr_fase("fase 2", EPOCAS_FASE2, opt2, prog2, congelada=False)

    print(f"Mejor precisión en validación: {mejor:.2%}")

    # Evaluación final con el conjunto de prueba (imágenes que nunca influyeron)
    if (CARPETA_DATOS / "test").is_dir():
        prueba = datasets.ImageFolder(CARPETA_DATOS / "test", transformar_evaluacion)
        punto = torch.load(RUTA_MODELO, map_location=dispositivo, weights_only=True)
        modelo.load_state_dict(punto["estado"])
        print(f"Precisión en prueba (test): {evaluar(DataLoader(prueba, batch_size=64)):.2%}")

    print(f"\nModelo guardado en {RUTA_MODELO}")
    print("Ahora ejecuta: python servidor.py")


if __name__ == "__main__":
    main()