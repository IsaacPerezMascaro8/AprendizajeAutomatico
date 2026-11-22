"""
fase3_arquitectura.py
---------------------
Fase 3: análisis y comprobación de las arquitecturas de modelos.py. Genera:

  results/modelos/summary_<M>.txt          model.summary() de cada modelo
  results/modelos_comparativa.csv          parámetros por tipo de capa, mapas tras cada bloque, campo receptivo
  results/campo_receptivo.csv              campo receptivo capa a capa (M0, M2 y M2 con 1 conv por bloque)
  results/fase3_sobreajuste.csv            test de sobreajuste con 20 fotos
  results/fase3_entrenamiento_corto.csv    entrenamiento corto (30 épocas, EarlyStopping) en el fold de Isaac
  results/fase3_historial_corto_<M>.csv    historial época a época
  results/figuras/arquitectura_M2.png      diagrama del modelo principal
  results/figuras/fase3_curvas_corto.png   curvas de loss y accuracy (train vs val)
  --ablacion: results/fase3_ablacion.csv   se quita una pieza cada vez (momentum BN, Dropout, BN, cabeza)
  --largo:    results/fase3_entrenamiento_largo.csv, fase3_historial_largo_<M>.csv, figuras/fase3_curvas_largo.png

Uso:  cd src && python fase3_arquitectura.py [--solo-analisis | --ablacion | --largo]
"""

import sys
import time

import keras
import matplotlib
import numpy as np
import pandas as pd
from keras import layers

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import config  # noqa: E402
import datos  # noqa: E402
import modelos  # noqa: E402

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SECUNDARIA, REJILLA, GRIS = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
DIR_MODELOS_TXT = config.DIR_RESULTADOS / "modelos"

EPOCAS_CORTO = 30
PACIENCIA_CORTO = 8
FOTOS_POR_LETRA_SOBREAJUSTE = 4      # 4 x 5 letras = 20 fotos
EPOCAS_MAX_SOBREAJUSTE = 200


# =============================================================================
# 1. Análisis de los modelos
# =============================================================================
def parametros_por_tipo(model):
    cuenta = {"conv": 0, "batchnorm_entrenables": 0, "batchnorm_no_entrenables": 0, "densa": 0}
    for capa in model.layers:
        if isinstance(capa, layers.Conv2D):
            cuenta["conv"] += capa.count_params()
        elif isinstance(capa, layers.BatchNormalization):
            n = capa.count_params()
            cuenta["batchnorm_entrenables"] += n // 2        # gamma, beta
            cuenta["batchnorm_no_entrenables"] += n // 2     # media y varianza móviles
        elif isinstance(capa, layers.Dense):
            cuenta["densa"] += capa.count_params()
    return cuenta


def resumen_texto(model):
    """Equivalente a model.summary() en texto plano (capa, tipo, forma de salida, parámetros)."""
    lineas = [f"Modelo: {model.name}", f"{'Capa':<22}{'Tipo':<24}{'Salida':<22}{'Parámetros':>12}", "-" * 80]
    for c in model.layers:
        forma = "x".join(str(d) for d in c.output.shape[1:])
        lineas.append(f"{c.name:<22}{type(c).__name__:<24}{forma:<22}{c.count_params():>12,}")
    p = parametros_por_tipo(model)
    lineas += ["-" * 80,
               f"Parámetros totales:      {model.count_params():>12,}",
               f"  no entrenables (medias/varianzas móviles de BN): {p['batchnorm_no_entrenables']:,}"]
    return "\n".join(lineas).replace(",", ".") + "\n"


def mapas_por_bloque(model):
    """Forma de la salida de cada MaxPooling (final de cada bloque) y de la entrada a la parte densa."""
    mapas = [f"{c.output.shape[1]}x{c.output.shape[2]}x{c.output.shape[3]}"
             for c in model.layers if isinstance(c, layers.MaxPooling2D)]
    entrada_densa = next(c for c in model.layers if isinstance(c, (layers.Flatten, layers.GlobalAveragePooling2D)))
    return mapas, int(entrada_densa.output.shape[-1])


def analizar_modelos():
    DIR_MODELOS_TXT.mkdir(parents=True, exist_ok=True)
    filas, filas_cr = [], []
    for nombre, crear in modelos.MODELOS.items():
        model = crear()
        (DIR_MODELOS_TXT / f"summary_{nombre}.txt").write_text(resumen_texto(model), encoding="utf-8")

        p = parametros_por_tipo(model)
        mapas, n_entrada_densa = mapas_por_bloque(model)
        cr = modelos.campo_receptivo(model)
        ultima_conv = [x for x in cr if "conv" in x[0]][-1]
        total = model.count_params()
        filas.append({
            "modelo": nombre, "nombre": model.name, "parametros_totales": total,
            "entrenables": total - p["batchnorm_no_entrenables"],
            "conv": p["conv"], "batchnorm": p["batchnorm_entrenables"] + p["batchnorm_no_entrenables"],
            "densa": p["densa"], "pct_densa": round(100 * p["densa"] / total, 1),
            "mapas_tras_cada_bloque": " -> ".join(mapas), "entrada_a_la_densa": n_entrada_densa,
            "campo_receptivo_ultima_conv_px": ultima_conv[1],
            "tamano_MB_float32": round(total * 4 / 1e6, 1),
        })
    # Campo receptivo capa a capa (también de la variante con 1 conv por bloque, para justificar las 2 convs)
    variantes = {"M0": modelos.crear_m0(), "M2 (2 conv/bloque)": modelos.crear_m2(),
                 "M2 con 1 conv/bloque": modelos.crear_m2(convs_por_bloque=1)}
    for nombre, model in variantes.items():
        for capa, r, salto in modelos.campo_receptivo(model):
            filas_cr.append({"modelo": nombre, "capa": capa, "campo_receptivo_px": r, "salto_px": salto})

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_RESULTADOS / "modelos_comparativa.csv", index=False)
    pd.DataFrame(filas_cr).to_csv(config.DIR_RESULTADOS / "campo_receptivo.csv", index=False)
    print(tabla.drop(columns=["nombre"]).to_string(index=False))
    print("\nCampo receptivo de la última conv: M2 con 1 conv/bloque =",
          [r for c, r, s in modelos.campo_receptivo(variantes["M2 con 1 conv/bloque"]) if "conv" in c][-1], "px")
    print(f"Resúmenes en {DIR_MODELOS_TXT}")


# =============================================================================
# 2. Diagrama del modelo principal (M2)
# =============================================================================
def diagrama_m2():
    model = modelos.crear_m2()
    etapas = [("Entrada", (144, 256, 3), "foto RGB\n[0, 1]", GRIS)]
    for i in range(4):
        pool = model.get_layer(f"bloque{i + 1}_pool")
        h, w, c = pool.output.shape[1:]
        etapas.append((f"Bloque {i + 1}", (h, w, c),
                       f"2× [Conv 3×3 ({c})\n→ BN → ReLU]\nMaxPool 2×2\nDropout 0,2", AZUL))
    etapas.append(("GAP", (1, 1, 256), "media de\ncada mapa", NARANJA))
    etapas.append(("Densa", (1, 1, 128), "Dense 128\nReLU\nDropout 0,5", NARANJA))
    etapas.append(("Salida", (1, 1, 5), "Dense 5\nsoftmax\nA E I O U", AQUA))

    fig, ax = plt.subplots(figsize=(15, 4.8))
    x = 0.0
    for titulo, (h, w, c), texto, color in etapas:
        alto = 0.6 + 2.4 * (h / 144) ** 0.5 if h > 1 else 0.25 + c / 256 * 2.2
        ancho = 0.35 + 0.9 * (c / 256) ** 0.5 if h > 1 else 0.35
        ax.add_patch(FancyBboxPatch((x, -alto / 2), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="white", alpha=0.9))
        forma = f"{h}×{w}×{c}" if h > 1 else f"{c}"
        ax.text(x + ancho / 2, 1.75, titulo, ha="center", va="bottom", fontsize=10, color=TINTA, weight="bold")
        ax.text(x + ancho / 2, 1.55, forma, ha="center", va="bottom", fontsize=9, color=TINTA)
        ax.text(x + ancho / 2, -1.65, texto, ha="center", va="top", fontsize=8.5, color=TINTA_SECUNDARIA)
        x_siguiente = x + ancho + 1.0
        if titulo != "Salida":
            ax.annotate("", xy=(x_siguiente - 0.08, 0), xytext=(x + ancho + 0.08, 0),
                        arrowprops=dict(arrowstyle="->", color=TINTA_SECUNDARIA, lw=1.2))
        x = x_siguiente
    ax.set_xlim(-0.3, x - 0.6)
    ax.set_ylim(-2.9, 2.2)
    ax.axis("off")
    ax.set_title(f"Arquitectura del modelo M2 ({model.count_params():,} parámetros)".replace(",", "."),
                 color=TINTA, loc="left", fontsize=12)
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / "arquitectura_M2.png"
    fig.savefig(ruta, dpi=200, facecolor="white")
    plt.close(fig)
    print("  ", ruta)


# =============================================================================
# 3. Comprobaciones
# =============================================================================
class TiempoPorEpoca(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["tiempo_s"] = time.time() - self._t


def test_sobreajuste(X, Y, df, split):
    """Cada modelo debe memorizar 20 fotos (sin augmentation). Si no llega a ~100 %, hay un error en el modelo,
    en las etiquetas o en la compilación."""
    rng = np.random.default_rng(config.SEMILLA)
    train = df.iloc[split["train"]]
    idx = np.concatenate([rng.choice(train.index[train["letra"] == l], FOTOS_POR_LETRA_SOBREAJUSTE, replace=False)
                          for l in config.CLASES])
    filas = []
    for nombre, crear in modelos.MODELOS.items():
        config.fijar_semillas()
        model = modelos.compilar(crear())
        historial = model.fit(X[idx], Y[idx], epochs=EPOCAS_MAX_SOBREAJUSTE, batch_size=len(idx), verbose=0)
        acc = np.array(historial.history["accuracy"])
        primera = int(np.argmax(acc >= 1.0)) + 1 if (acc >= 1.0).any() else None
        perdida_inf, acc_inf = model.evaluate(X[idx], Y[idx], verbose=0)
        filas.append({"modelo": nombre, "fotos": len(idx), "epocas": EPOCAS_MAX_SOBREAJUSTE,
                      "primera_epoca_100pct_train": primera,
                      "accuracy_final_train": round(float(acc[-1]), 3),
                      "loss_final_train": round(float(historial.history["loss"][-1]), 4),
                      "accuracy_inferencia": round(float(acc_inf), 3), "loss_inferencia": round(float(perdida_inf), 4)})
        print(f"  {nombre}: 100 % en la época {primera}; final train acc={acc[-1]:.3f}, "
              f"inferencia acc={acc_inf:.3f} loss={perdida_inf:.4f}")
        keras.backend.clear_session()
    pd.DataFrame(filas).to_csv(config.DIR_RESULTADOS / "fase3_sobreajuste.csv", index=False)


def entrenamiento_corto(X, Y, split, constructores=None, epocas=EPOCAS_CORTO, paciencia=PACIENCIA_CORTO,
                        sufijo="corto"):
    """Entrena cada modelo en el fold de Isaac, con augmentation. Con paciencia=None no hay EarlyStopping.
    Solo depuración: comprobar que entrenan y medir el tiempo. No se mira el test (se reserva para la Fase 5)."""
    constructores = constructores or modelos.MODELOS
    aumento = datos.capas_aumento()
    ds_val = datos.crear_dataset(X, Y, split["val"], entrenamiento=False)
    filas, historiales = [], {}
    for nombre, crear in constructores.items():
        config.fijar_semillas()
        ds_train = datos.crear_dataset(X, Y, split["train"], entrenamiento=True, aumento=aumento)
        model = modelos.compilar(crear())
        callbacks = [TiempoPorEpoca()]
        if paciencia:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                                           restore_best_weights=True))
        t0 = time.time()
        h = model.fit(ds_train, validation_data=ds_val, epochs=epocas, verbose=0, callbacks=callbacks)
        total = time.time() - t0
        hist = pd.DataFrame(h.history)
        hist.index = hist.index + 1
        hist.index.name = "epoca"
        hist.to_csv(config.DIR_RESULTADOS / f"fase3_historial_{sufijo}_{nombre}.csv")
        historiales[nombre] = hist
        mejor = int(hist["val_loss"].idxmin())
        fila = {"modelo": nombre, "fold": split["nombre"], "epocas_hechas": len(hist), "mejor_epoca": mejor,
                "val_loss_mejor": round(hist.loc[mejor, "val_loss"], 4),
                "val_accuracy_mejor": round(hist.loc[mejor, "val_accuracy"], 3),
                "train_accuracy_mejor": round(hist.loc[mejor, "accuracy"], 3),
                "train_accuracy_ultima": round(hist["accuracy"].iloc[-1], 3),
                "val_accuracy_max": round(hist["val_accuracy"].max(), 3),
                "val_accuracy_media_ultimas_10": round(hist["val_accuracy"].iloc[-10:].mean(), 3),
                "tiempo_1a_epoca_s": round(hist["tiempo_s"].iloc[0], 1),
                "tiempo_por_epoca_s": round(hist["tiempo_s"].iloc[1:].median(), 2),
                "tiempo_total_s": round(total, 1)}
        filas.append(fila)
        print(f"  {nombre}: {fila}")
        keras.backend.clear_session()
    pd.DataFrame(filas).to_csv(config.DIR_RESULTADOS / f"fase3_entrenamiento_{sufijo}.csv", index=False)
    figura_curvas(historiales, split["nombre"], epocas, paciencia, sufijo)


def figura_curvas(historiales, fold, epocas, paciencia, sufijo):
    fig, ejes = plt.subplots(2, len(historiales), figsize=(max(4 * len(historiales), 11), 6.4), sharey="row")
    for j, (nombre, h) in enumerate(historiales.items()):
        for i, (metrica, etiqueta) in enumerate([("loss", "Pérdida (entropía cruzada)"), ("accuracy", "Accuracy")]):
            ax = ejes[i, j]
            ax.plot(h.index, h[metrica], color=AZUL, linewidth=2, marker="o", markersize=3, label="entrenamiento")
            ax.plot(h.index, h[f"val_{metrica}"], color=NARANJA, linewidth=2, marker="o", markersize=3,
                    label="validación")
            mejor = h["val_loss"].idxmin()
            ax.axvline(mejor, color=GRIS, linestyle=":", linewidth=1)
            for lado in ["top", "right"]:
                ax.spines[lado].set_visible(False)
            ax.grid(True, color=REJILLA, linewidth=0.8)
            ax.tick_params(colors=TINTA_SECUNDARIA)
            if i == 0:
                ax.set_title(nombre, color=TINTA)
            if j == 0:
                ax.set_ylabel(etiqueta, color=TINTA)
            if i == 1:
                ax.set_xlabel("Época", color=TINTA)
                ax.set_ylim(0, 1.02)
    ejes[0, 0].legend(frameon=False)
    parada = f"EarlyStopping sobre val_loss (paciencia {paciencia})" if paciencia else "sin EarlyStopping"
    fig.suptitle(f"Entrenamiento de comprobación ({fold}, ≤ {epocas} épocas, {parada}; "
                 f"línea punteada = menor val_loss)", color=TINTA, fontsize=11)
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / f"fase3_curvas_{sufijo}.png"
    fig.savefig(ruta, dpi=170, facecolor="white")
    plt.close(fig)
    print("  ", ruta)


def ablacion(X, Y, df, split, epocas=12):
    """Depuración de por qué los modelos de 4 bloques entrenan despacio: se quita una pieza cada vez.
    Sin augmentation (para aislar la arquitectura), fold de Isaac, mismas semillas. Incluye el test de sobreajuste
    (20 fotos) en inferencia para ver el efecto del momentum de BatchNorm."""
    rng = np.random.default_rng(config.SEMILLA)
    train = df.iloc[split["train"]]
    idx20 = np.concatenate([rng.choice(train.index[train["letra"] == l], FOTOS_POR_LETRA_SOBREAJUSTE, replace=False)
                            for l in config.CLASES])
    variantes = {
        "M2 (BN momentum 0,99, valor por defecto de Keras)": dict(cabeza="gap", bn_momentum=0.99),
        "M2 (BN momentum 0,9)": dict(cabeza="gap"),
        "M2 con Dropout solo en la cabeza": dict(cabeza="gap", dropout_conv=0.0),
        "M2 sin Dropout": dict(cabeza="gap", dropout_conv=0.0, dropout_denso=0.0),
        "M2 sin BatchNorm ni Dropout": dict(cabeza="gap", batchnorm=False, dropout_conv=0.0, dropout_denso=0.0),
        "M2 con GlobalMaxPooling": dict(cabeza="gmp"),
        "M1 (Flatten)": dict(cabeza="flatten"),
    }
    ds_val = datos.crear_dataset(X, Y, split["val"], entrenamiento=False)
    filas = []
    for nombre, kw in variantes.items():
        config.fijar_semillas()
        model = modelos.compilar(modelos.crear_cnn("ablacion", **kw))
        model.fit(X[idx20], Y[idx20], epochs=150, batch_size=20, verbose=0)
        _, acc_sobre = model.evaluate(X[idx20], Y[idx20], verbose=0)
        keras.backend.clear_session()

        config.fijar_semillas()
        model = modelos.compilar(modelos.crear_cnn("ablacion", **kw))
        h = model.fit(datos.crear_dataset(X, Y, split["train"], entrenamiento=True), validation_data=ds_val,
                      epochs=epocas, verbose=0)
        h = h.history
        fila = {"variante": nombre, "sobreajuste_20_fotos_acc_inferencia": round(float(acc_sobre), 3),
                "epocas": epocas, "train_acc_final": round(h["accuracy"][-1], 3),
                "val_acc_final": round(h["val_accuracy"][-1], 3),
                "val_loss_min": round(min(h["val_loss"]), 3), "val_loss_max": round(max(h["val_loss"]), 3)}
        filas.append(fila)
        print(f"  {fila}")
        keras.backend.clear_session()
    pd.DataFrame(filas).to_csv(config.DIR_RESULTADOS / "fase3_ablacion.csv", index=False)


# =============================================================================
def main():
    print("##### 1. Análisis de los modelos")
    analizar_modelos()
    print("\n##### 2. Diagrama")
    diagrama_m2()
    if "--solo-analisis" in sys.argv:
        return

    df = datos.cargar_listado()
    X, Y = datos.cargar_imagenes(df)
    split = next(s for s in datos.splits_lopo(df) if s["persona_test"] == "Isaac")

    if "--largo" in sys.argv:
        print("\n##### Comprobación larga (60 épocas, sin EarlyStopping, con augmentation)")
        entrenamiento_corto(X, Y, split, epocas=60, paciencia=None, sufijo="largo", constructores={
            "M2": modelos.crear_m2,
            "M2 Dropout solo cabeza": lambda: modelos.crear_m2(dropout_conv=0.0)})
        return

    if "--ablacion" in sys.argv:
        print("\n##### Ablación (sin augmentation)")
        ablacion(X, Y, df, split)
        return

    print("\n##### 3a. Test de sobreajuste (20 fotos, sin augmentation)")
    test_sobreajuste(X, Y, df, split)
    print("\n##### 3b. Entrenamiento corto en el fold de Isaac")
    entrenamiento_corto(X, Y, split)


if __name__ == "__main__":
    main()
