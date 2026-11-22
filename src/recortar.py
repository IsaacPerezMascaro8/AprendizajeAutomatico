"""
recortar.py
-----------
Diagnóstico: recorte automático de la mano con MediaPipe Hands (HandLandmarker, API Tasks).
NO modifica las fotos originales: los recortes se guardan en cache/.

Para cada foto (a resolución completa):
  1. Se detecta la mano (21 puntos clave). Si no se detecta con confianza 0,5, se reintenta con 0,2.
  2. Caja que contiene los 21 puntos -> cuadrado de lado = max(ancho, alto) * (1 + 2 * MARGEN), centrado en la mano.
     (MARGEN = 0,25 por cada lado). Si el cuadrado se sale de la foto, lo que falta se rellena de negro.
  3. Se redimensiona a 128x128 (LANCZOS, con antialias).

Genera:
  cache/recortes_128x128.npz                  X (uint8), rutas, detectado, caja
  results/recortes_deteccion.csv              una fila por foto (detectada, pasada, caja, mano)
  results/recortes_fallos_por_persona_letra.csv
  results/figuras/recortes_ejemplos.png       recortes buenos y fallos
  results/figuras/revision/recortes_fallos.jpg  todas las fotos sin detección (revisión a ojo)

Uso:  cd src && python recortar.py
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
import datos

MODELO_MANOS = config.DIR_MODELOS / "hand_landmarker.task"
MARGEN = 0.25
TAM_RECORTE = 128
CONFIANZAS = [0.5, 0.2]   # 1ª pasada y reintento


def crear_detector(confianza):
    from mediapipe.tasks.python import BaseOptions, vision
    opciones = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELO_MANOS)),
                                            num_hands=1, min_hand_detection_confidence=confianza,
                                            min_hand_presence_confidence=confianza)
    return vision.HandLandmarker.create_from_options(opciones)


def detectar(detector, imagen_np):
    """Devuelve (x_min, y_min, x_max, y_max) en píxeles y la mano ('Left'/'Right'), o None si no hay mano."""
    import mediapipe as mp
    resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_np))
    if not resultado.hand_landmarks:
        return None
    alto, ancho = imagen_np.shape[:2]
    xs = np.array([p.x for p in resultado.hand_landmarks[0]]) * ancho
    ys = np.array([p.y for p in resultado.hand_landmarks[0]]) * alto
    mano = resultado.handedness[0][0].category_name if resultado.handedness else ""
    return (xs.min(), ys.min(), xs.max(), ys.max()), mano


def caja_cuadrada(caja):
    x0, y0, x1, y1 = caja
    lado = max(x1 - x0, y1 - y0) * (1 + 2 * MARGEN)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - lado / 2)), int(round(cy - lado / 2)), int(round(lado))


def recortar(imagen, cuadro):
    x, y, lado = cuadro
    # PIL rellena de negro lo que queda fuera de la imagen
    return imagen.crop((x, y, x + lado, y + lado)).resize((TAM_RECORTE, TAM_RECORTE), Image.LANCZOS)


def figura_ejemplos(df, X, n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(config.SEMILLA)
    # Ejemplos buenos solo de David e Isaac (sin cara en el recorte), uno o dos por letra
    buenos = df[df["detectado"] & df["persona"].isin(["David", "Isaac"])]
    buenos = buenos.groupby("letra", observed=True).sample(2, random_state=config.SEMILLA)
    fallos = df[~df["detectado"]]
    fallos = fallos.sample(min(5, len(fallos)), random_state=config.SEMILLA) if len(fallos) else fallos

    fig, ejes = plt.subplots(3, n, figsize=(1.7 * n, 5.6))
    for j in range(n):
        for i in range(3):
            ejes[i, j].axis("off")
    for j, (i, fila) in enumerate(buenos.iterrows()):
        original = Image.open(fila["ruta"]).convert("RGB")
        dib = ImageDraw.Draw(original)
        x, y, lado = fila["caja_x"], fila["caja_y"], fila["caja_lado"]
        dib.rectangle([x, y, x + lado, y + lado], outline=(42, 120, 214), width=max(4, original.width // 200))
        ejes[0, j].imshow(original.reduce(4))
        ejes[0, j].set_title(f"{fila['persona']} {fila['letra']}", fontsize=8)
        ejes[1, j].imshow(X[i])
        ejes[1, j].set_title("recorte 128×128", fontsize=8)
    for j, (_, fila) in enumerate(fallos.iterrows()):
        ejes[2, j].imshow(Image.open(fila["ruta"]).convert("RGB").reduce(4))
        ejes[2, j].set_title(f"FALLO {fila['persona']} {fila['letra']} #{fila['orden']}", fontsize=8, color="#e34948")
    fig.suptitle("Recorte automático de la mano con MediaPipe: foto con la caja (fila 1), recorte (fila 2) y "
                 "fotos sin detección (fila 3)", fontsize=11)
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / "recortes_ejemplos.png"
    fig.savefig(ruta, dpi=150, facecolor="white")
    plt.close(fig)
    return ruta


def hoja_fallos(df):
    fallos = df[~df["detectado"]]
    if fallos.empty:
        return None
    columnas, ancho = 8, 240
    alto = ancho * 9 // 16
    filas = int(np.ceil(len(fallos) / columnas))
    hoja = Image.new("RGB", (columnas * (ancho + 4), filas * (alto + 18)), "white")
    dib = ImageDraw.Draw(hoja)
    for k, fila in enumerate(fallos.itertuples()):
        x, y = (k % columnas) * (ancho + 4), (k // columnas) * (alto + 18)
        hoja.paste(Image.open(fila.ruta).convert("RGB").resize((ancho, alto)), (x, y))
        dib.text((x + 2, y + alto + 2), f"{fila.persona} {fila.letra} #{fila.orden}", fill="black")
    ruta = config.DIR_FIGURAS / "revision" / "recortes_fallos.jpg"
    hoja.save(ruta, quality=85)
    return ruta


def main():
    df = datos.cargar_listado()
    detectores = {c: crear_detector(c) for c in CONFIANZAS}
    recortes, filas = [], []
    for fila in df.itertuples():
        imagen = Image.open(fila.ruta).convert("RGB")
        imagen_np = np.asarray(imagen)
        deteccion, pasada = None, None
        for k, c in enumerate(CONFIANZAS):
            deteccion = detectar(detectores[c], imagen_np)
            if deteccion:
                pasada = k + 1
                break
        if deteccion:
            (caja, mano) = deteccion
            cuadro = caja_cuadrada(caja)
            recortes.append(np.asarray(recortar(imagen, cuadro)))
            ancho_mano = caja[2] - caja[0]
            filas.append({"detectado": True, "pasada": pasada, "mano": mano, "caja_x": cuadro[0],
                          "caja_y": cuadro[1], "caja_lado": cuadro[2],
                          "ancho_mano_en_256px": round(ancho_mano / imagen.width * 256, 1)})
        else:
            recortes.append(np.zeros((TAM_RECORTE, TAM_RECORTE, 3), np.uint8))
            filas.append({"detectado": False, "pasada": None, "mano": "", "caja_x": -1, "caja_y": -1,
                          "caja_lado": -1, "ancho_mano_en_256px": np.nan})
    for detector in detectores.values():
        detector.close()
    info = pd.DataFrame(filas, index=df.index)
    df = pd.concat([df, info], axis=1)
    X = np.stack(recortes)

    config.DIR_CACHE.mkdir(exist_ok=True)
    np.savez(config.DIR_CACHE / f"recortes_{TAM_RECORTE}x{TAM_RECORTE}.npz", X=X,
             rutas=df["ruta"].to_numpy().astype(str), detectado=df["detectado"].to_numpy(),
             caja=df[["caja_x", "caja_y", "caja_lado"]].to_numpy())
    columnas = ["persona", "letra", "archivo", "orden", "detectado", "pasada", "mano", "caja_x", "caja_y",
                "caja_lado", "ancho_mano_en_256px"]
    df[columnas].to_csv(config.DIR_RESULTADOS / "recortes_deteccion.csv", index=False)

    tabla = df.pivot_table(index="persona", columns="letra", values="detectado",
                           aggfunc=lambda s: int((~s.astype(bool)).sum()), observed=True)
    tabla["Total"] = tabla.sum(axis=1)
    tabla.loc["Total"] = tabla.sum()
    tabla.to_csv(config.DIR_RESULTADOS / "recortes_fallos_por_persona_letra.csv")

    n_fallos = int((~df["detectado"]).sum())
    print(f"Detectadas: {int(df['detectado'].sum())} de {len(df)}; fallos: {n_fallos} ({n_fallos / len(df):.1%})")
    print("Detectadas en la 2ª pasada (confianza 0,2):", int((df["pasada"] == 2).sum()))
    print("\nFallos por persona y letra:\n", tabla, sep="")
    print("\nAncho de la mano (px en una imagen de 256 de ancho):\n",
          df.groupby("persona")["ancho_mano_en_256px"].describe()[["min", "25%", "50%", "75%", "max"]].round(1), sep="")
    print("\n", figura_ejemplos(df, X), sep="")
    print(hoja_fallos(df))


if __name__ == "__main__":
    main()
