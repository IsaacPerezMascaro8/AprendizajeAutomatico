"""
modelos.py
----------
Arquitecturas CNN del proyecto (API Sequential + model.add, como en los apuntes).

  M0  baseline de los apuntes: 2 bloques Conv2D+MaxPooling, Flatten, Dense(128), softmax
  M1  más profundo: 4 bloques [Conv-BN-ReLU] x2 + MaxPooling + Dropout, Flatten, Dense, Dropout, softmax
  M2  igual que M1 pero con GlobalAveragePooling2D en lugar de Flatten     <- modelo principal propuesto
  M3  M2 + regularización L2 en los pesos

Entrada: 144x256x3 (config.FORMA_ENTRADA), píxeles en [0, 1]. Salida: Dense(5, softmax) -> A, E, I, O, U.
El data augmentation NO está aquí: se aplica en el pipeline tf.data (datos.crear_dataset), solo en entrenamiento.
Justificación de cada decisión: docs/decisiones.md (Fase 3).
"""

from keras import Input, Sequential, layers, optimizers, regularizers

import config


# -----------------------------------------------------------------------------
# Bloque convolucional reutilizable
# -----------------------------------------------------------------------------
def anadir_bloque_conv(model, filtros, convs_por_bloque=2, dropout=0.2, l2=0.0, bn_momentum=0.9, batchnorm=True,
                       nombre="bloque"):
    """Añade al modelo un bloque:   [Conv2D 3x3 -> BatchNormalization -> ReLU] x convs_por_bloque
                                    -> MaxPooling2D 2x2 -> Dropout
    - Conv2D sin bias (use_bias=False): la BatchNormalization que va justo después ya suma su propio desplazamiento
      (beta), así que el bias de la convolución sería redundante.
    - BatchNormalization ANTES de la ReLU, como en el artículo original (Ioffe y Szegedy, 2015): normaliza la
      entrada de la no linealidad.
    - Dos convoluciones 3x3 seguidas ven lo mismo que una 5x5 (campo receptivo 5x5) con menos parámetros
      (2·9 = 18 frente a 25 por canal) y una ReLU más (estilo VGG).
    - BatchNormalization con momentum=0.9 (Keras usa 0.99 por defecto). Con ~20 batches por época, con 0.99 las
      medias/varianzas móviles (las que se usan en validación y test) tardan decenas de épocas en olvidar su valor
      inicial (media 0, varianza 1), y la red se comporta muy distinto en entrenamiento y en inferencia (comprobado en
      la Fase 3: val_loss inestable y el test de sobreajuste fallaba en inferencia). Con 0.9 basta ~2 épocas.
    - Dropout después del MaxPooling (recomendación de los apuntes).
    - batchnorm=False quita la BN (solo para el experimento de ablación: sin BN la red de 4 bloques no aprende)."""
    regularizador = regularizers.L2(l2) if l2 > 0 else None
    for i in range(convs_por_bloque):
        model.add(layers.Conv2D(filtros, kernel_size=(3, 3), padding="same", use_bias=not batchnorm,
                                kernel_regularizer=regularizador, name=f"{nombre}_conv{i + 1}"))
        if batchnorm:
            model.add(layers.BatchNormalization(momentum=bn_momentum, name=f"{nombre}_bn{i + 1}"))
        model.add(layers.Activation("relu", name=f"{nombre}_relu{i + 1}"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f"{nombre}_pool"))
    if dropout > 0:
        model.add(layers.Dropout(dropout, name=f"{nombre}_dropout"))


# -----------------------------------------------------------------------------
# M0 – baseline de los apuntes
# -----------------------------------------------------------------------------
def crear_m0(forma_entrada=config.FORMA_ENTRADA, filtros=(32, 64), unidades_densas=128):
    """Modelo mínimo, tal cual se ve en clase. Sirve de punto de partida y para mostrar el problema de Flatten con
    imágenes grandes: tras 2 poolings el mapa es 36x64x64 = 147 456 valores, y la Dense(128) que va detrás tiene
    ~18,9 millones de parámetros (casi todos los del modelo)."""
    model = Sequential(name="M0_baseline")
    model.add(Input(shape=forma_entrada))
    for i, f in enumerate(filtros):
        model.add(layers.Conv2D(f, kernel_size=(3, 3), activation="relu", padding="same", name=f"conv{i + 1}"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{i + 1}"))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(unidades_densas, activation="relu", name="densa"))
    model.add(layers.Dense(config.NUM_CLASES, activation="softmax", name="salida"))
    return model


# -----------------------------------------------------------------------------
# M1 / M2 / M3 – CNN de 4 bloques
# -----------------------------------------------------------------------------
def crear_cnn(nombre, forma_entrada=config.FORMA_ENTRADA, filtros=(32, 64, 128, 256), convs_por_bloque=2,
              dropout_conv=0.2, cabeza="gap", unidades_densas=128, dropout_denso=0.5, l2=0.0, batchnorm=True,
              bn_momentum=0.9):
    """CNN de bloques convolucionales + cabeza clasificadora.
      cabeza = "flatten": aplana el último mapa (posición fija: cada peso de la densa mira un sitio concreto).
      cabeza = "gap":     GlobalAveragePooling2D, media de cada mapa de características (invariante a la posición).
      cabeza = "gmp":     GlobalMaxPooling2D, máximo de cada mapa (variante probada en la Fase 3; ver decisiones.md).
    Todos los parámetros se pueden cambiar en la Fase 4 (filtros, dropout, unidades, L2...)."""
    model = Sequential(name=nombre)
    model.add(Input(shape=forma_entrada))
    for i, f in enumerate(filtros):
        anadir_bloque_conv(model, f, convs_por_bloque, dropout_conv, l2, bn_momentum=bn_momentum, batchnorm=batchnorm,
                           nombre=f"bloque{i + 1}")

    if cabeza == "flatten":
        model.add(layers.Flatten(name="flatten"))
    elif cabeza == "gap":
        model.add(layers.GlobalAveragePooling2D(name="gap"))
    elif cabeza == "gmp":
        model.add(layers.GlobalMaxPooling2D(name="gmp"))
    else:
        raise ValueError(f"cabeza desconocida: {cabeza}")

    regularizador = regularizers.L2(l2) if l2 > 0 else None
    model.add(layers.Dense(unidades_densas, activation="relu", kernel_regularizer=regularizador, name="densa"))
    model.add(layers.Dropout(dropout_denso, name="densa_dropout"))
    model.add(layers.Dense(config.NUM_CLASES, activation="softmax", name="salida"))
    return model


def crear_m1(**kwargs):
    """M1 – más profundo, con Flatten."""
    return crear_cnn("M1_profundo_flatten", cabeza="flatten", **kwargs)


def crear_m2(**kwargs):
    """M2 – como M1 pero con GlobalAveragePooling2D (modelo principal propuesto)."""
    return crear_cnn("M2_profundo_gap", cabeza="gap", **kwargs)


def crear_m3(l2=1e-4, **kwargs):
    """M3 – M2 con regularización L2 (weight decay) en convoluciones y capa densa."""
    return crear_cnn("M3_gap_l2", cabeza="gap", l2=l2, **kwargs)


MODELOS = {"M0": crear_m0, "M1": crear_m1, "M2": crear_m2, "M3": crear_m3}


def compilar(model, learning_rate=config.LEARNING_RATE):
    """Compilación de los apuntes: Adam + entropía cruzada categórica (etiquetas one-hot) + accuracy."""
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------------------------------------------------------
# Campo receptivo
# -----------------------------------------------------------------------------
def campo_receptivo(model):
    """Campo receptivo (en píxeles de la entrada) tras cada capa Conv2D / MaxPooling2D.
    Fórmula: r_nuevo = r + (k - 1) * salto ;  salto_nuevo = salto * stride
    (r = campo receptivo, salto = distancia en píxeles de entrada entre dos neuronas vecinas).
    Devuelve una lista de (nombre_capa, campo_receptivo, salto)."""
    r, salto, resultado = 1, 1, []
    for capa in model.layers:
        if isinstance(capa, layers.Conv2D):
            k, s = capa.kernel_size[0], capa.strides[0]
        elif isinstance(capa, layers.MaxPooling2D):
            k, s = capa.pool_size[0], capa.strides[0]
        else:
            continue
        r += (k - 1) * salto
        salto *= s
        resultado.append((capa.name, r, salto))
    return resultado


if __name__ == "__main__":
    for nombre, crear in MODELOS.items():
        m = crear()
        print(f"{nombre}: {m.count_params():,} parámetros; salida {m.output_shape}")
