"""
diagnostico.py
--------------
Diagnóstico del bajo rendimiento inicial de la Fase 3 (fold de Isaac). No cambia ninguna decisión tomada:
solo experimentos para entender el problema.

  --fallos    1. Descartar fallos: resumen de la Fase 3, alineación de etiquetas en tf.data, preprocesamiento de
                 train/val, distribución de las predicciones en validación, M1 sin augmentation con 2 learning rates.
  --gradcam   2. Grad-CAM del modelo M2 entrenado (¿mira la mano o el fondo?)
  --recortes  3. Entrenar M1 y M2 con los recortes de la mano (recortar.py) y comparar con las fotos completas.

Salidas en results/diagnostico_*.csv y results/figuras/diagnostico_*.png, gradcam_*.png.
Modelos entrenados en models/diag_*.keras.

Uso:  cd src && python diagnostico.py --fallos | --gradcam | --recortes
"""

import sys
import time

import keras
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import datos  # noqa: E402
import modelos  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_SECUNDARIA, REJILLA, ROJO, VERDE = "#0b0b0b", "#52514e", "#e4e3df", "#e34948", "#008300"
R = config.DIR_RESULTADOS
EPOCAS = 60   # misma configuración que la comprobación larga de la Fase 3


# =============================================================================
# Utilidades
# =============================================================================
def preparar_fold():
    df = datos.cargar_listado()
    X, Y = datos.cargar_imagenes(df)
    split = next(s for s in datos.splits_lopo(df) if s["persona_test"] == "Isaac")
    return df, X, Y, split


def entrenar(model, X, Y, split, epocas=EPOCAS, aumento=True, lr=config.LEARNING_RATE, train=None, val=None):
    """Entrena sin EarlyStopping (como la comprobación larga de la Fase 3) y devuelve el historial."""
    config.fijar_semillas()
    train = split["train"] if train is None else train
    val = split["val"] if val is None else val
    modelos.compilar(model, lr)
    ds_train = datos.crear_dataset(X, Y, train, entrenamiento=True,
                                   aumento=datos.capas_aumento() if aumento else None)
    ds_val = datos.crear_dataset(X, Y, val, entrenamiento=False)
    t0 = time.time()
    h = model.fit(ds_train, validation_data=ds_val, epochs=epocas, verbose=0)
    hist = pd.DataFrame(h.history)
    hist.index = hist.index + 1
    hist.index.name = "epoca"
    hist.attrs["tiempo_s"] = time.time() - t0
    return hist


def resumen_historial(nombre, hist, **extra):
    return {"experimento": nombre, **extra, "epocas": len(hist),
            "train_acc_final": round(hist["accuracy"].iloc[-1], 3),
            "train_loss_final": round(hist["loss"].iloc[-1], 3),
            "val_acc_final": round(hist["val_accuracy"].iloc[-1], 3),
            "val_loss_final": round(hist["val_loss"].iloc[-1], 3),
            "val_acc_max": round(hist["val_accuracy"].max(), 3),
            "val_acc_media_ult10": round(hist["val_accuracy"].iloc[-10:].mean(), 3),
            "val_loss_min": round(hist["val_loss"].min(), 3),
            "tiempo_min": round(hist.attrs.get("tiempo_s", np.nan) / 60, 1)}


def estilo(ax):
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.tick_params(colors=TINTA_SECUNDARIA)


def figura_curvas(historiales, titulo, ruta):
    n = len(historiales)
    fig, ejes = plt.subplots(2, n, figsize=(max(3.6 * n, 11), 6.2), sharey="row", squeeze=False)
    for j, (nombre, h) in enumerate(historiales.items()):
        for i, (m, etiqueta) in enumerate([("loss", "Pérdida"), ("accuracy", "Accuracy")]):
            ax = ejes[i, j]
            ax.plot(h.index, h[m], color=AZUL, linewidth=2, label="entrenamiento")
            ax.plot(h.index, h[f"val_{m}"], color=NARANJA, linewidth=2, label="validación")
            estilo(ax)
            if i == 0:
                ax.set_title(nombre, color=TINTA, fontsize=10)
            if j == 0:
                ax.set_ylabel(etiqueta, color=TINTA)
            if i == 1:
                ax.set_xlabel("Época", color=TINTA)
                ax.set_ylim(0, 1.02)
                ax.axhline(0.2, color=TINTA_SECUNDARIA, linestyle=":", linewidth=1)
    ejes[1, 0].text(1, 0.21, "azar (20 %)", fontsize=8, color=TINTA_SECUNDARIA)
    ejes[0, 0].legend(frameon=False)
    fig.suptitle(titulo, color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor="white")
    plt.close(fig)
    print("  ", ruta)


# =============================================================================
# 1. Descartar fallos
# =============================================================================
def resumen_fase3():
    filas = []
    sobre = pd.read_csv(R / "fase3_sobreajuste.csv").set_index("modelo")
    for m in ["M0", "M1", "M2", "M3"]:
        h = pd.read_csv(R / f"fase3_historial_corto_{m}.csv", index_col="epoca")
        filas.append({**resumen_historial(f"{m} corto (EarlyStopping)", h),
                      "sobreajuste_train_acc": sobre.loc[m, "accuracy_final_train"],
                      "sobreajuste_inferencia_acc": sobre.loc[m, "accuracy_inferencia"]})
    for m in ["M2", "M2 Dropout solo cabeza"]:
        h = pd.read_csv(R / f"fase3_historial_largo_{m}.csv", index_col="epoca")
        filas.append(resumen_historial(f"{m} largo (60 épocas)", h))
    tabla = pd.DataFrame(filas).drop(columns=["tiempo_min"])
    tabla.to_csv(R / "diagnostico_resumen_fase3.csv", index=False)
    print(tabla.to_string(index=False))


def comprobar_alineacion(df, X, Y, split):
    """Se pasa el índice de cada foto por el pipeline como si fuera el 'peso' de la muestra: así viaja pegado a la
    imagen a través de shuffle, batch y augmentation, y al final se comprueba que la etiqueta coincide con la carpeta."""
    codigos = df["letra"].cat.codes.to_numpy()
    indices_como_peso = np.arange(len(df), dtype=np.float32)
    resultado = {}
    for conjunto, entrenamiento in [("train", True), ("val", False)]:
        ds = datos.crear_dataset(X, Y, split[conjunto], entrenamiento=entrenamiento, pesos=indices_como_peso,
                                 aumento=datos.capas_aumento() if entrenamiento else None)
        vistos = []
        for xb, yb, ib in ds:
            ib = ib.numpy().astype(int)
            assert (yb.numpy().argmax(1) == codigos[ib]).all(), f"Etiquetas desalineadas en {conjunto}"
            vistos.append(ib)
        vistos = np.concatenate(vistos)
        assert sorted(vistos.tolist()) == sorted(split[conjunto].tolist()), f"Faltan o sobran fotos en {conjunto}"
        resultado[conjunto] = len(vistos)
    print(f"  Alineación OK: {resultado} fotos comprobadas una a una (etiqueta = carpeta de origen)")

    # Figura: 8 fotos de un batch de train (con augmentation) y 8 de val, ya preprocesadas, con su etiqueta.
    # Se omiten las fotos de Rodrigo (sale su cara) para que la figura pueda ir a la memoria.
    fig, ejes = plt.subplots(2, 8, figsize=(17, 3.9))
    for fila, (conjunto, entrenamiento) in enumerate([("train", True), ("val", False)]):
        ds = datos.crear_dataset(X, Y, split[conjunto], entrenamiento=entrenamiento, pesos=indices_como_peso,
                                 aumento=datos.capas_aumento() if entrenamiento else None, batch_size=64)
        xb, yb, ib = next(iter(ds))
        ib = ib.numpy().astype(int)
        elegidas = [k for k in range(len(ib)) if df["persona"].iloc[ib[k]] != "Rodrigo"][:8]
        for col, k in enumerate(elegidas):
            ax = ejes[fila, col]
            ax.imshow(np.clip(xb[k].numpy(), 0, 1))
            ax.set_title(f"etiqueta: {config.CLASES[int(yb[k].numpy().argmax())]}  "
                         f"(carpeta {df['letra'].iloc[ib[k]]})", fontsize=8.5, color=TINTA)
            ax.set_xticks([]); ax.set_yticks([])
        ejes[fila, 0].set_ylabel("train\n(con aumento)" if entrenamiento else "validación", fontsize=10,
                                 color=TINTA)
    fig.suptitle("Batches tal como llegan a la red (tf.data, ya preprocesados), con la etiqueta one-hot decodificada",
                 color=TINTA, fontsize=11)
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / "diagnostico_batch_etiquetado.png"
    fig.savefig(ruta, dpi=130, facecolor="white")
    plt.close(fig)
    print("  ", ruta)


def comprobar_preprocesamiento(df, X, split):
    filas = []
    for conjunto in ["train", "val", "test"]:
        x = X[split[conjunto]]
        filas.append({"conjunto": conjunto, "fotos": len(x), "forma": "x".join(map(str, x.shape[1:])),
                      "dtype": str(x.dtype), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3),
                      "media": round(float(x.mean()), 3), "desviacion": round(float(x.std()), 3)})
    aum = datos.crear_dataset(X, np.zeros((len(X), 5), np.float32), split["train"], entrenamiento=True,
                              aumento=datos.capas_aumento())
    xa = np.concatenate([b[0].numpy() for b in aum])
    filas.append({"conjunto": "train tras augmentation", "fotos": len(xa), "forma": "x".join(map(str, xa.shape[1:])),
                  "dtype": str(xa.dtype), "min": round(float(xa.min()), 3), "max": round(float(xa.max()), 3),
                  "media": round(float(xa.mean()), 3), "desviacion": round(float(xa.std()), 3)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(R / "diagnostico_preprocesamiento.csv", index=False)
    print(tabla.to_string(index=False))
    brillo = pd.DataFrame({"persona": df["persona"], "brillo_medio": X.mean(axis=(1, 2, 3))}).groupby("persona")
    print("  Brillo medio por persona:", brillo["brillo_medio"].mean().round(3).to_dict())


def distribucion_predicciones(df, X, Y, split):
    """Entrena M2 como en la comprobación larga (60 épocas, con augmentation), lo guarda y mira qué predice en val."""
    model = modelos.crear_m2()
    hist = entrenar(model, X, Y, split)
    model.save(config.DIR_MODELOS / "diag_M2_original.keras")
    hist.to_csv(R / "diagnostico_historial_M2_original.csv")
    pred = model.predict(X[split["val"]], verbose=0).argmax(1)
    real = Y[split["val"]].argmax(1)
    conf = pd.crosstab(pd.Categorical([config.CLASES[i] for i in real], config.CLASES),
                       pd.Categorical([config.CLASES[i] for i in pred], config.CLASES),
                       rownames=["real"], colnames=["predicha"], dropna=False)
    conf.to_csv(R / "diagnostico_confusion_val_M2_original.csv")
    print(f"  M2 (60 épocas): val acc = {(pred == real).mean():.3f}")
    print("  Predicciones en validación por clase:", dict(zip(config.CLASES, np.bincount(pred, minlength=5))))
    print("  Matriz de confusión (val):\n", conf, sep="")
    personas_val = df["persona"].to_numpy()[split["val"]]
    for p in np.unique(personas_val):
        m = personas_val == p
        print(f"  acc val {p}: {(pred[m] == real[m]).mean():.3f} ({m.sum()} fotos)")
    return resumen_historial("M2 fotos completas, con aumento, lr 1e-3", hist)


def m1_sin_aumento(X, Y, split):
    filas, historiales = [], {}
    for lr in [1e-3, 3e-4]:
        hist = entrenar(modelos.crear_m1(), X, Y, split, epocas=40, aumento=False, lr=lr)
        nombre = f"M1 sin aumento, lr {lr:g}"
        hist.to_csv(R / f"diagnostico_historial_M1_sinaumento_lr{lr:g}.csv")
        historiales[nombre] = hist
        filas.append(resumen_historial(nombre, hist))
        print("  ", filas[-1])
        keras.backend.clear_session()
    figura_curvas(historiales, "Diagnóstico: M1 sin data augmentation (fold de Isaac)",
                  config.DIR_FIGURAS / "diagnostico_curvas_M1_sinaumento.png")
    return filas


def paso_fallos():
    df, X, Y, split = preparar_fold()
    print("##### 1a. Resumen de la Fase 3"); resumen_fase3()
    print("\n##### 1b. Alineación de etiquetas"); comprobar_alineacion(df, X, Y, split)
    print("\n##### 1c. Preprocesamiento train / val / test"); comprobar_preprocesamiento(df, X, split)
    print("\n##### 1d. Distribución de predicciones en validación (M2, 60 épocas)")
    filas = [distribucion_predicciones(df, X, Y, split)]
    keras.backend.clear_session()
    print("\n##### 1e. M1 sin augmentation con lr 1e-3 y 3e-4 (40 épocas)")
    filas += m1_sin_aumento(X, Y, split)
    pd.DataFrame(filas).to_csv(R / "diagnostico_entrenamientos.csv", index=False)


# =============================================================================
# 2. Grad-CAM
# =============================================================================
def grad_cam(model, imagenes, capa="bloque4_relu2"):
    """Grad-CAM (Selvaraju et al., 2017): pondera cada mapa de la última capa convolucional por la media del gradiente
    de la clase predicha respecto a ese mapa, suma, aplica ReLU y normaliza a [0, 1]."""
    # Se recorren las capas del Sequential una a una y se guarda la salida de la capa elegida
    with tf.GradientTape() as cinta:
        x = tf.convert_to_tensor(imagenes)
        for c in model.layers:
            x = c(x, training=False)
            if c.name == capa:
                mapas = x
        pred = x
        clase = tf.argmax(pred, axis=1)
        puntuacion = tf.gather(pred, clase, axis=1, batch_dims=1)
    grads = cinta.gradient(puntuacion, mapas)
    pesos = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(pesos * mapas, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], imagenes.shape[1:3]).numpy()[..., 0]
    return cam, pred.numpy()


def figura_gradcam(model, X, Y, df, indices, titulo, ruta):
    cam, pred = grad_cam(model, X[indices])
    n = len(indices)
    columnas = 5
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3.6 * columnas, 2.6 * filas * X.shape[1] / X.shape[2] + 0.9),
                             squeeze=False)
    for k, i in enumerate(indices):
        ax = ejes.flat[k]
        ax.imshow(X[i])
        ax.imshow(cam[k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        real, predicha = config.CLASES[Y[i].argmax()], config.CLASES[pred[k].argmax()]
        ax.set_title(f"real {real} · predice {predicha} ({pred[k].max():.0%})", fontsize=9,
                     color=VERDE if real == predicha else ROJO)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(ejes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(titulo, color=TINTA, fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, facecolor="white")
    plt.close(fig)
    print("  ", ruta)


def fotos_gradcam(df, split, detectado=None):
    """2 fotos de validación por letra, de David (sin cara), elegidas al azar con semilla fija."""
    val = df.iloc[split["val"]]
    val = val[val["persona"] == "David"]
    if detectado is not None:
        val = val[detectado[val.index]]
    return val.groupby("letra", observed=True).sample(2, random_state=config.SEMILLA).index.to_numpy()


def paso_gradcam():
    df, X, Y, split = preparar_fold()
    model = keras.models.load_model(config.DIR_MODELOS / "diag_M2_original.keras")
    figura_gradcam(model, X, Y, df, fotos_gradcam(df, split),
                   "Grad-CAM de M2 (fotos completas 144×256, 60 épocas): zonas que más influyen en la predicción "
                   "(validación, fold de Isaac)", config.DIR_FIGURAS / "gradcam_M2_original.png")


# =============================================================================
# 3. Recortes
# =============================================================================
def cargar_recortes(df):
    cache = np.load(config.DIR_CACHE / "recortes_128x128.npz")
    assert np.array_equal(cache["rutas"], df["ruta"].to_numpy().astype(str)), "Recortes desalineados: rehacer recortar.py"
    return cache["X"].astype(np.float32) / 255.0, cache["detectado"]


def paso_recortes():
    df, X, Y, split = preparar_fold()
    Xr, detectado = cargar_recortes(df)
    # Mismo split que con las fotos completas, quitando solo las fotos donde MediaPipe no detectó la mano
    train_r = split["train"][detectado[split["train"]]]
    val_r = split["val"][detectado[split["val"]]]
    print(f"  Recortes: train {len(train_r)}/{len(split['train'])}, val {len(val_r)}/{len(split['val'])} "
          f"(se excluyen las fotos sin detección)")

    filas, historiales = [], {}
    previo = pd.read_csv(R / "diagnostico_entrenamientos.csv")
    # M2 con fotos completas ya entrenado en --fallos; M1 con fotos completas y aumento se entrena aquí
    h = pd.read_csv(R / "diagnostico_historial_M2_original.csv", index_col="epoca")
    historiales["M2 foto completa"] = h
    filas.append({**previo.iloc[0].to_dict(), "experimento": "M2 foto completa 144×256"})
    for nombre, crear, Xu, tr, va, forma in [
            ("M2 recorte 128×128", modelos.crear_m2, Xr, train_r, val_r, (128, 128, 3)),
            ("M1 recorte 128×128", modelos.crear_m1, Xr, train_r, val_r, (128, 128, 3)),
            ("M1 foto completa 144×256", modelos.crear_m1, X, split["train"], split["val"], config.FORMA_ENTRADA)]:
        model = crear(forma_entrada=forma)
        hist = entrenar(model, Xu, Y, split, train=tr, val=va)
        hist.to_csv(R / f"diagnostico_historial_{nombre.replace(' ', '_').replace('×', 'x')}.csv")
        historiales[nombre.replace(" 144×256", "").replace(" 128×128", "")] = hist
        filas.append(resumen_historial(nombre, hist, dispositivo=tf.config.list_logical_devices()[-1].device_type))
        print("  ", filas[-1], flush=True)
        pd.DataFrame(filas).to_csv(R / "diagnostico_recortes.csv", index=False)   # se guarda tras cada modelo
        if nombre == "M2 recorte 128×128":
            model.save(config.DIR_MODELOS / "diag_M2_recorte.keras")
            figura_gradcam(model, Xr, Y, df, fotos_gradcam(df, split, detectado),
                           "Grad-CAM de M2 con recortes de la mano (128×128, 60 épocas; validación, fold de Isaac)",
                           config.DIR_FIGURAS / "gradcam_M2_recorte.png")
            pred = model.predict(Xr[val_r], verbose=0).argmax(1)
            print("  Predicciones en val (recorte):", dict(zip(config.CLASES, np.bincount(pred, minlength=5))))
        keras.backend.clear_session()
    tabla = pd.DataFrame(filas)
    tabla.to_csv(R / "diagnostico_recortes.csv", index=False)
    print(tabla.to_string(index=False))
    orden = ["M1 foto completa", "M2 foto completa", "M1 recorte", "M2 recorte"]
    figura_curvas({k: historiales[k] for k in orden},
                  f"Fotos completas frente a recortes de la mano (fold de Isaac, {EPOCAS} épocas, con augmentation)",
                  config.DIR_FIGURAS / "diagnostico_curvas_recortes.png")


if __name__ == "__main__":
    if "--fallos" in sys.argv:
        paso_fallos()
    if "--gradcam" in sys.argv:
        paso_gradcam()
    if "--recortes" in sys.argv:
        paso_recortes()
