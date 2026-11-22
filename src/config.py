"""
config.py
---------
Configuración común del proyecto: rutas, clases, tamaño de imagen, semilla e hiperparámetros por defecto.
El resto de módulos importan de aquí para que todos usen los mismos valores.
"""

import os
import random
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
# Carpeta raíz del proyecto (la que contiene src/, results/, ...)
RAIZ = Path(__file__).resolve().parent.parent

# Las fotos originales están directamente en la raíz: RAIZ/<Persona>/<Letra>/...
# (SOLO LECTURA: nunca se modifican, mueven ni borran)
DIR_DATOS = RAIZ

DIR_MODELOS = RAIZ / "models"
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
CSV_EXPERIMENTOS = DIR_RESULTADOS / "experimentos.csv"
DIR_DOCS = RAIZ / "docs"

# Carpetas de la raíz que NO son personas (para detectar las personas automáticamente)
CARPETAS_NO_PERSONA = {
    "src", "notebooks", "models", "results", "docs", "dataset_limpio",
    "__pycache__", ".git", ".ipynb_checkpoints", ".claude", "cache",
}

# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
# Orden fijo de las clases: índice 0 = A, 1 = E, ... (se usa en to_categorical)
CLASES = ["A", "E", "I", "O", "U"]
NUM_CLASES = len(CLASES)

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tamaño de entrada de la red (alto, ancho), RGB.
# Principal: 144x256, que respeta el 16:9 de las fotos (sin padding ni recorte).
# Comparación: 128x128 con padding (bandas negras arriba y abajo; imagen útil de 128x72).
# Siempre se redimensiona con tf.image.resize_with_pad: si la proporción coincide, no añade bandas.
ALTO_IMAGEN = 144
ANCHO_IMAGEN = 256
CANALES = 3
FORMA_ENTRADA = (ALTO_IMAGEN, ANCHO_IMAGEN, CANALES)
TAMANOS_ALTERNATIVOS = {"principal": (144, 256), "cuadrado_padding": (128, 128)}

# Caché en disco de las imágenes ya redimensionadas (carpeta ignorada por git)
DIR_CACHE = RAIZ / "cache"

# Grupos de ráfaga: fotos consecutivas (misma persona y letra) separadas por <= este nº de segundos
# pertenecen al mismo grupo. Justificado con el histograma de tiempos (Fase 2).
UMBRAL_RAFAGA_SEGUNDOS = 5

# Proporción de validación (sacada de las personas de entrenamiento, por grupos de ráfaga).
# StratifiedGroupKFold usa round(1 / PROPORCION_VALIDACION) particiones y se toma una: 1/6 ≈ 17 %.
PROPORCION_VALIDACION = 1 / 6

# Split aleatorio (SOLO para demostrar la fuga de información): proporciones de validación y test
PROPORCION_VAL_ALEATORIO = 0.15
PROPORCION_TEST_ALEATORIO = 0.15

# ----------------------------------------------------------------------------
# Entrenamiento (valores por defecto; cada experimento puede cambiarlos)
# ----------------------------------------------------------------------------
SEMILLA = 42
BATCH_SIZE = 32
EPOCAS = 50
LEARNING_RATE = 1e-3
PACIENCIA_EARLY_STOPPING = 8

# Data augmentation (solo entrenamiento). Justificación en docs/decisiones.md (Fase 2).
AUMENTO = {
    "rotacion": 10 / 360,      # ±10° (RandomRotation usa fracciones de vuelta completa)
    "zoom": 0.10,              # ±10 %
    "traslacion": 0.10,        # ±10 % del alto y del ancho
    "brillo": 0.20,            # ±0,2 sobre el rango [0, 1]
    "contraste": 0.20,         # factor de contraste en [0,8, 1,2]
}


def detectar_personas(dir_datos=DIR_DATOS):
    """Devuelve la lista ordenada de personas: carpetas que contienen al menos una carpeta de letra (A, E, I, O, U),
    sin distinguir mayúsculas. Así no hay que escribir a mano quién está en el dataset."""
    personas = []
    for carpeta in sorted(Path(dir_datos).iterdir()):
        if not carpeta.is_dir() or carpeta.name in CARPETAS_NO_PERSONA or carpeta.name.startswith("."):
            continue
        subcarpetas = {s.name.upper() for s in carpeta.iterdir() if s.is_dir()}
        if subcarpetas & set(CLASES):
            personas.append(carpeta.name)
    return personas


def fijar_semillas(semilla=SEMILLA):
    """Fija las semillas de Python, NumPy y TensorFlow para que los experimentos sean reproducibles.
    Nota: en GPU algunas operaciones siguen sin ser 100 % deterministas; enable_op_determinism lo fuerza
    a costa de algo de velocidad."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)

    import tensorflow as tf  # se importa aquí para que config.py se pueda usar sin cargar TensorFlow
    tf.random.set_seed(semilla)  # equivale a keras.utils.set_random_seed para TF
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


if __name__ == "__main__":
    print("Raíz del proyecto:", RAIZ)
    print("Clases:", CLASES)
    print("Personas detectadas:", detectar_personas())
