"""
datos.py
--------
Todo lo relacionado con los datos del proyecto:

  1. cargar_listado()      -> DataFrame con una fila por foto (ruta, persona, letra, hora, grupo de ráfaga...)
  2. cargar_imagenes()     -> matrices X (imágenes en [0, 1]) e Y (one-hot), con caché en disco
  3. splits_lopo()         -> divisiones leave-one-person-out (train / val / test), en índices
     split_aleatorio()     -> división aleatoria foto a foto (SOLO para demostrar la fuga de información)
     comprobar_split()     -> asserts de que no hay solapamiento
  4. capas_aumento()       -> data augmentation con capas de Keras (solo entrenamiento)
     crear_dataset()       -> tf.data.Dataset listo para model.fit

Las fotos originales están en <raíz>/<Persona>/<Letra>/ y son de SOLO LECTURA:
aquí solo se leen, nunca se modifican, mueven ni borran.
"""

import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import config


# =============================================================================
# 1. Listado de fotos
# =============================================================================
def orden_natural(nombre):
    """Clave de ordenación 'natural': 'foto 2' va antes que 'foto 10'.
    Así las fotos quedan en el orden en que se hicieron (nombres de la cámara)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nombre)]


def hora_captura(ruta):
    """Hora en que se hizo la foto.
    1º) del nombre de la cámara de Windows: WIN_AAAAMMDD_HH_MM_SS_Pro.jpg
    2º) si no, de los metadatos EXIF (DateTimeOriginal)
    3º) si tampoco hay, devuelve NaT (la foto formará un grupo de ráfaga ella sola)."""
    nombre = Path(ruta).name
    m = re.match(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})", nombre)
    if m:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    try:
        with Image.open(ruta) as img:
            fecha = img.getexif().get_ifd(0x8769).get(36867)  # DateTimeOriginal
        if fecha:
            return datetime.strptime(fecha, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return pd.NaT


def asignar_rafagas(df, umbral_segundos=config.UMBRAL_RAFAGA_SEGUNDOS):
    """Añade al DataFrame:
      - 'segundos_desde_anterior': tiempo desde la foto anterior de la misma persona y letra,
      - 'rafaga': id de grupo de ráfaga. Una foto empieza grupo nuevo si es la primera de su carpeta,
        si no tiene hora, o si han pasado MÁS de `umbral_segundos` desde la anterior.
    Las fotos de una misma ráfaga son casi iguales (mismo fondo, luz y postura), así que nunca
    deben repartirse entre entrenamiento y validación/test."""
    df = df.sort_values(["persona", "letra", "orden"]).copy()
    df["segundos_desde_anterior"] = (df.groupby(["persona", "letra"], observed=True)["hora"]
                                     .diff().dt.total_seconds())
    empieza_grupo = df["segundos_desde_anterior"].isna() | (df["segundos_desde_anterior"] > umbral_segundos)
    df["rafaga"] = empieza_grupo.cumsum() - 1   # ids 0, 1, 2, ...
    return df.sort_index()


def cargar_listado(dir_datos=config.DIR_DATOS, personas=None, umbral_rafaga=config.UMBRAL_RAFAGA_SEGUNDOS):
    """Recorre las carpetas de cada persona y devuelve un DataFrame con una fila por foto:

        ruta, persona, letra, archivo, orden, ancho, alto, hora, segundos_desde_anterior, rafaga

    - Las personas se detectan leyendo las carpetas (config.detectar_personas), así que cuando
      se añada Juan/ entra solo, sin tocar el código.
    - Las carpetas de letra se reconocen sin distinguir mayúsculas ('a/' o 'A/').
    - 'orden' es la posición de la foto dentro de su carpeta (0, 1, 2...) según el orden natural
      del nombre de archivo, es decir, el orden en que se hicieron.
    - El ancho y el alto se leen de la cabecera del archivo (sin decodificar la imagen entera).
    - El índice del DataFrame (0..N-1) es la posición de cada foto en X e Y (cargar_imagenes).
    """
    dir_datos = Path(dir_datos)
    if personas is None:
        personas = config.detectar_personas(dir_datos)

    filas = []
    for persona in personas:
        for carpeta_letra in sorted((dir_datos / persona).iterdir()):
            letra = carpeta_letra.name.upper()
            if not carpeta_letra.is_dir() or letra not in config.CLASES:
                continue
            archivos = [f for f in carpeta_letra.iterdir()
                        if f.is_file() and f.suffix.lower() in config.EXTENSIONES_IMAGEN]
            archivos.sort(key=lambda f: orden_natural(f.name))
            for orden, archivo in enumerate(archivos):
                with Image.open(archivo) as img:
                    ancho, alto = img.size
                filas.append({
                    "ruta": str(archivo),
                    "persona": persona,
                    "letra": letra,
                    "archivo": archivo.name,
                    "orden": orden,
                    "ancho": ancho,
                    "alto": alto,
                    "hora": hora_captura(archivo),
                })

    df = pd.DataFrame(filas)
    # Letra como categoría con el orden fijo A, E, I, O, U (útil para tablas, gráficas y etiquetas)
    df["letra"] = pd.Categorical(df["letra"], categories=config.CLASES, ordered=True)
    sin_hora = df["hora"].isna().sum()
    if sin_hora:
        warnings.warn(f"{sin_hora} fotos sin hora de captura: cada una será su propio grupo de ráfaga.")
    return asignar_rafagas(df, umbral_rafaga)


# =============================================================================
# 2. Imágenes y etiquetas
# =============================================================================
def _redimensionar(ruta, alto, ancho):
    """Lee una foto y la lleva a (alto, ancho) SIN recortar ni deformar: tf.image.resize_with_pad
    mantiene la proporción y, si no coincide, rellena con bandas negras. Para 144x256 (16:9, igual que las
    fotos) no añade bandas. antialias=True evita el aliasing al reducir mucho (las fotos de Rodrigo se
    reducen 10 veces)."""
    import tensorflow as tf
    imagen = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    imagen = tf.image.resize_with_pad(tf.cast(imagen, tf.float32), alto, ancho, antialias=True)
    return np.clip(np.round(imagen.numpy()), 0, 255).astype(np.uint8)


def cargar_imagenes(df, alto=config.ALTO_IMAGEN, ancho=config.ANCHO_IMAGEN, usar_cache=True):
    """Devuelve:
      X: array (N, alto, ancho, 3) float32 con los píxeles normalizados a [0, 1]
      Y: array (N, 5) one-hot (to_categorical), columnas en el orden A, E, I, O, U
    en el mismo orden que las filas de df.

    Caché: la primera vez guarda las imágenes redimensionadas (uint8) en cache/imagenes_<alto>x<ancho>.npz.
    Si el listado de fotos cambia (p. ej. llegan las de Juan), la caché se regenera sola."""
    from keras.utils import to_categorical

    rutas = df["ruta"].to_numpy()
    archivo_cache = config.DIR_CACHE / f"imagenes_{alto}x{ancho}.npz"

    X_uint8 = None
    if usar_cache and archivo_cache.exists():
        cache = np.load(archivo_cache, allow_pickle=False)
        if np.array_equal(cache["rutas"], rutas.astype(str)):
            X_uint8 = cache["X"]
        else:
            print("La lista de fotos ha cambiado: se regenera la caché.")

    if X_uint8 is None:
        print(f"Redimensionando {len(rutas)} fotos a {alto}x{ancho}...")
        X_uint8 = np.stack([_redimensionar(r, alto, ancho) for r in rutas])
        if usar_cache:
            config.DIR_CACHE.mkdir(exist_ok=True)
            np.savez(archivo_cache, X=X_uint8, rutas=rutas.astype(str))

    X = X_uint8.astype(np.float32) / 255.0                   # normalización a [0, 1]
    Y = to_categorical(df["letra"].cat.codes.to_numpy(), num_classes=config.NUM_CLASES).astype(np.float32)
    return X, Y


# =============================================================================
# 3. Divisiones (splits) en índices
# =============================================================================
def _desviacion_maxima(etiquetas, pos_val, objetivo):
    """Peor diferencia, entre los valores de `etiquetas`, entre la fracción que cae en validación y el objetivo."""
    return max(abs((etiquetas[pos_val] == v).sum() / (etiquetas == v).sum() - objetivo)
               for v in np.unique(etiquetas))


def _mejor_particion_validacion(estratos, grupos, proporcion_val, semilla, n_semillas=50):
    """StratifiedGroupKFold reparte GRUPOS enteros. Como hay pocos grupos por letra y de tamaños muy distintos
    (de 1 a 21 fotos), una partición concreta puede quedar desequilibrada (p. ej. casi sin fotos de una letra).
    Por eso se prueban todas las particiones de varias semillas y se elige la más equilibrada: la que minimiza
    (peor desviación por letra) + (peor desviación por persona) respecto a la proporción deseada.
    Es determinista: misma semilla -> misma partición."""
    n_splits = round(1 / proporcion_val)
    personas = np.array([e.rsplit("_", 1)[0] for e in estratos])
    letras = np.array([e.rsplit("_", 1)[1] for e in estratos])
    mejor, mejor_error = None, np.inf
    for s in range(semilla, semilla + n_semillas):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for pos_train, pos_val in sgkf.split(estratos, estratos, groups=grupos):
            error = (_desviacion_maxima(letras, pos_val, 1 / n_splits)
                     + _desviacion_maxima(personas, pos_val, 1 / n_splits))
            if error < mejor_error:
                mejor, mejor_error = (pos_train, pos_val), error
    return mejor


def splits_lopo(df, proporcion_val=config.PROPORCION_VALIDACION, semilla=config.SEMILLA):
    """Leave-one-person-out: un split por persona.
      - test:  todas las fotos de esa persona (nunca vista en entrenamiento),
      - val:   ~1/6 de las fotos de las demás personas, elegidas por GRUPOS DE RÁFAGA con StratifiedGroupKFold
               (estratificando por persona+letra, para que haya de todas las personas y letras),
      - train: el resto.
    Devuelve una lista de diccionarios {nombre, persona_test, train, val, test} con índices (posiciones en df)."""
    splits = []
    for persona in sorted(df["persona"].unique()):
        test = np.flatnonzero(df["persona"] == persona)
        resto = np.flatnonzero(df["persona"] != persona)

        estratos = (df["persona"].astype(str) + "_" + df["letra"].astype(str)).to_numpy()[resto]
        grupos = df["rafaga"].to_numpy()[resto]
        pos_train, pos_val = _mejor_particion_validacion(estratos, grupos, proporcion_val, semilla)

        splits.append({
            "nombre": f"lopo_{persona}",
            "persona_test": persona,
            "train": resto[pos_train],
            "val": resto[pos_val],
            "test": test,
        })
    return splits


def split_aleatorio(df, prop_val=config.PROPORCION_VAL_ALEATORIO, prop_test=config.PROPORCION_TEST_ALEATORIO,
                    semilla=config.SEMILLA):
    """División aleatoria estratificada por letra, FOTO A FOTO (mezcla personas y ráfagas).
    Solo se usa como comparación: tiene FUGA DE INFORMACIÓN (fotos casi iguales de la misma ráfaga y de la
    misma persona acaban a la vez en entrenamiento y test), así que su accuracy será artificialmente alta."""
    indices = np.arange(len(df))
    letras = df["letra"].astype(str).to_numpy()
    train_val, test = train_test_split(indices, test_size=prop_test, stratify=letras, random_state=semilla)
    train, val = train_test_split(train_val, test_size=prop_val / (1 - prop_test),
                                  stratify=letras[train_val], random_state=semilla)
    return {"nombre": "aleatorio", "persona_test": None, "train": np.sort(train), "val": np.sort(val),
            "test": np.sort(test)}


def comprobar_split(df, split, exigir_sin_fuga=True):
    """Comprueba con asserts que un split es correcto y devuelve una tabla de fotos por letra y conjunto.
      - Siempre: train, val y test no comparten fotos y entre los tres cubren todo el dataset.
      - Si exigir_sin_fuga: tampoco comparten grupos de ráfaga, y la persona de test no está en train ni val.
    Para el split aleatorio (exigir_sin_fuga=False) solo informa de cuántas ráfagas se reparten (la fuga)."""
    conjuntos = {c: set(split[c].tolist()) for c in ["train", "val", "test"]}
    assert not conjuntos["train"] & conjuntos["val"], "Fotos repetidas entre train y val"
    assert not conjuntos["train"] & conjuntos["test"], "Fotos repetidas entre train y test"
    assert not conjuntos["val"] & conjuntos["test"], "Fotos repetidas entre val y test"
    assert len(conjuntos["train"] | conjuntos["val"] | conjuntos["test"]) == len(df), "Faltan fotos en el split"

    rafagas = {c: set(df["rafaga"].to_numpy()[split[c]].tolist()) for c in conjuntos}
    personas = {c: set(df["persona"].to_numpy()[split[c]].tolist()) for c in conjuntos}
    compartidas = len((rafagas["train"] & rafagas["val"]) | (rafagas["train"] & rafagas["test"])
                      | (rafagas["val"] & rafagas["test"]))
    if exigir_sin_fuga:
        assert compartidas == 0, f"{compartidas} grupos de ráfaga repartidos entre conjuntos"
        assert personas["test"] == {split["persona_test"]}, "El test tiene fotos de otra persona"
        assert split["persona_test"] not in personas["train"] | personas["val"], "Persona de test vista en train/val"

    tabla = pd.DataFrame({c: df["letra"].iloc[split[c]].value_counts().reindex(config.CLASES)
                          for c in ["train", "val", "test"]}).T
    tabla["Total"] = tabla.sum(axis=1)
    tabla.attrs["rafagas_compartidas"] = compartidas
    return tabla


# =============================================================================
# 4. Data augmentation y tf.data
# =============================================================================
def capas_aumento(aumento=config.AUMENTO, semilla=config.SEMILLA):
    """Bloque de capas de Keras para el data augmentation. Se aplica SOLO al entrenamiento.
      - RandomFlip horizontal: equivale a hacer el gesto con la otra mano (en el dataset ya hay de ambas).
      - RandomRotation ±10°: la mano nunca sale perfectamente vertical; más giro podría cambiar el gesto.
      - RandomZoom ±10 % y RandomTranslation ±10 %: distancia y posición distintas respecto a la cámara.
      - RandomBrightness / RandomContrast ±20 %: luz natural frente a bombilla, webcams distintas.
    fill_mode='constant' (negro) en las transformaciones geométricas: el modo por defecto ('reflect') crearía
    una segunda mano reflejada cuando la mano está pegada al borde, como pasa en muchas fotos."""
    from keras import Sequential, layers

    return Sequential([
        layers.RandomFlip("horizontal", seed=semilla),
        layers.RandomRotation(aumento["rotacion"], fill_mode="constant", seed=semilla + 1),
        layers.RandomZoom(aumento["zoom"], fill_mode="constant", seed=semilla + 2),
        layers.RandomTranslation(aumento["traslacion"], aumento["traslacion"], fill_mode="constant",
                                 seed=semilla + 3),
        layers.RandomBrightness(aumento["brillo"], value_range=(0, 1), seed=semilla + 4),
        layers.RandomContrast(aumento["contraste"], value_range=(0, 1), seed=semilla + 5),
    ], name="aumento_datos")


def crear_dataset(X, Y, indices, entrenamiento, batch_size=config.BATCH_SIZE, aumento=None, pesos=None,
                  semilla=config.SEMILLA):
    """Crea un tf.data.Dataset con las fotos de `indices`.
      - entrenamiento=True: baraja en cada época y aplica `aumento` (capas_aumento()) si se pasa.
      - entrenamiento=False (val/test): ni baraja ni aumenta, para que la evaluación sea siempre la misma.
      - pesos: opcional, sample_weight por foto (p. ej. para equilibrar personas en la Fase 4)."""
    import tensorflow as tf

    tensores = (X[indices], Y[indices]) if pesos is None else (X[indices], Y[indices], pesos[indices])
    ds = tf.data.Dataset.from_tensor_slices(tensores)
    if entrenamiento:
        ds = ds.shuffle(len(indices), seed=semilla, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if entrenamiento and aumento is not None:
        ds = ds.map(lambda x, *resto: (aumento(x, training=True), *resto), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    df = cargar_listado()
    print(df.head())
    print(f"\n{len(df)} fotos de {df['persona'].nunique()} personas: {sorted(df['persona'].unique())}")
    print(f"{df['rafaga'].nunique()} grupos de ráfaga (umbral {config.UMBRAL_RAFAGA_SEGUNDOS} s)")
