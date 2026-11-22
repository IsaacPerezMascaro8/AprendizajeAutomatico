"""
preparar_datos.py
-----------------
Fase 2: preprocesamiento, grupos de ráfaga, splits y data augmentation. Genera:

  results/rafagas_umbral.csv               nº de grupos de ráfaga según el umbral (sensibilidad)
  results/rafagas_por_persona_letra.csv    grupos de ráfaga por persona y letra (umbral elegido)
  results/splits_resumen.csv               fotos por letra en train/val/test de cada split
  results/splits.json                      fotos de cada conjunto (rutas relativas), para reproducibilidad
  results/figuras/rafagas_histograma.png
  results/figuras/preprocesado_tamanos.png
  results/figuras/aumento_ejemplos.png
  cache/imagenes_144x256.npz, cache/imagenes_128x128.npz

Uso:  cd src && python preparar_datos.py
"""

import json
import time

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import datos  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_SECUNDARIA, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

# Fotos de ejemplo para las figuras (David: en sus fotos nunca sale la cara)
FOTO_AUMENTO = ("David", "U", 32)
FOTO_TAMANOS = ("David", "I", 17)


def estilo_ejes(ax):
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_SECUNDARIA)
    ax.tick_params(colors=TINTA_SECUNDARIA, length=0)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


# -----------------------------------------------------------------------------
# 1. Ráfagas
# -----------------------------------------------------------------------------
def analizar_rafagas(df):
    tiempos = df["segundos_desde_anterior"].dropna()
    umbral = config.UMBRAL_RAFAGA_SEGUNDOS

    # Histograma: segundos 0..20 uno a uno y una última barra "> 20"
    maximo = 20
    valores = np.minimum(tiempos.round().astype(int), maximo + 1)
    cuentas = valores.value_counts().reindex(range(maximo + 2), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colores = [AZUL if s <= umbral else NARANJA for s in cuentas.index]
    ax.bar(cuentas.index, cuentas.values, width=0.8, color=colores)
    ax.axvline(umbral + 0.5, color=TINTA, linestyle="--", linewidth=1)
    ax.text(umbral + 0.7, cuentas.max() * 0.92, f"umbral = {umbral} s", color=TINTA, fontsize=10)
    ax.set_xticks(range(maximo + 2), [str(s) for s in range(maximo + 1)] + [f">{maximo}"])
    ax.set_xlabel("Segundos desde la foto anterior (misma persona y letra)", color=TINTA)
    ax.set_ylabel("Número de fotos", color=TINTA)
    ax.set_title(f"Tiempo entre fotos consecutivas ({len(tiempos)} parejas)", color=TINTA, loc="left")
    ax.bar([0], [0], color=AZUL, label="misma ráfaga")
    ax.bar([0], [0], color=NARANJA, label="empieza ráfaga nueva")
    ax.legend(frameon=False, loc="upper right")
    estilo_ejes(ax)
    fig.tight_layout()
    ruta_fig = config.DIR_FIGURAS / "rafagas_histograma.png"
    fig.savefig(ruta_fig, dpi=200, facecolor="white")
    plt.close(fig)

    # Sensibilidad al umbral
    filas = []
    for u in [1, 2, 3, 5, 10, 30, 60, 300]:
        d = datos.asignar_rafagas(df, u)
        por_pl = d.groupby(["persona", "letra"], observed=True)["rafaga"].nunique()
        tam = d["rafaga"].value_counts()
        filas.append({"umbral_s": u, "pct_tiempos_dentro": round(100 * float((tiempos <= u).mean()), 1),
                      "grupos": d["rafaga"].nunique(), "tamano_mediano": float(tam.median()),
                      "tamano_max": int(tam.max()), "min_grupos_persona_letra": int(por_pl.min())})
    sensibilidad = pd.DataFrame(filas)
    sensibilidad.to_csv(config.DIR_RESULTADOS / "rafagas_umbral.csv", index=False)

    tabla = df.pivot_table(index="persona", columns="letra", values="rafaga", aggfunc="nunique", observed=True)
    tabla["Total"] = df.groupby("persona")["rafaga"].nunique()
    tabla.loc["Total"] = tabla.sum()
    tabla.to_csv(config.DIR_RESULTADOS / "rafagas_por_persona_letra.csv")
    tam = df["rafaga"].value_counts()

    print(f"Tiempos entre fotos consecutivas <= {umbral} s: {(tiempos <= umbral).mean():.1%}")
    print("\nSensibilidad al umbral:\n", sensibilidad.to_string(index=False), sep="")
    print(f"\nGrupos de ráfaga por persona y letra (umbral {umbral} s):\n", tabla, sep="")
    print(f"Tamaño de los grupos: mediana {tam.median():.0f}, media {tam.mean():.1f}, máximo {tam.max()}; "
          f"grupos de 1 foto: {(tam == 1).sum()}")
    print("  ", ruta_fig)


# -----------------------------------------------------------------------------
# 2. Figuras de preprocesado y aumento
# -----------------------------------------------------------------------------
def ruta_de(df, persona, letra, orden):
    fila = df[(df["persona"] == persona) & (df["letra"] == letra) & (df["orden"] == orden)]
    return fila["ruta"].iloc[0]


def figura_tamanos(df):
    ruta = ruta_de(df, *FOTO_TAMANOS)
    original = Image.open(ruta).convert("RGB")
    paneles = [(np.asarray(original), f"Original ({original.width}×{original.height})")]
    for alto, ancho in [(144, 256), (128, 128)]:
        img = datos._redimensionar(ruta, alto, ancho)
        texto = "sin bandas" if alto * 16 == ancho * 9 else "con padding (útil 128×72)"
        paneles.append((img, f"{alto}×{ancho} {texto}"))
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [16 / 9, 16 / 9, 1]})
    for ax, (img, titulo) in zip(ejes, paneles):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(titulo, color=TINTA, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Tamaño de entrada: la mano ocupa poco del encuadre (píxeles reales, sin suavizar)",
                 color=TINTA, fontsize=11)
    fig.tight_layout()
    ruta_fig = config.DIR_FIGURAS / "preprocesado_tamanos.png"
    fig.savefig(ruta_fig, dpi=160, facecolor="white")
    plt.close(fig)
    print("  ", ruta_fig)


def figura_aumento(X, df):
    import tensorflow as tf
    i = df.index[(df["persona"] == FOTO_AUMENTO[0]) & (df["letra"] == FOTO_AUMENTO[1])
                 & (df["orden"] == FOTO_AUMENTO[2])][0]
    aumento = datos.capas_aumento()
    lote = tf.repeat(X[i:i + 1], 8, axis=0)
    aumentadas = aumento(lote, training=True).numpy()

    fig, ejes = plt.subplots(3, 3, figsize=(12, 7.2))
    imagenes = [X[i]] + list(aumentadas)
    for k, (ax, img) in enumerate(zip(ejes.flat, imagenes)):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Original (144×256)" if k == 0 else f"Aumentada {k}", color=TINTA, fontsize=10)
        if k == 0:
            for s in ax.spines.values():
                s.set_color(AZUL); s.set_linewidth(3)
    a = config.AUMENTO
    fig.suptitle(f"Data augmentation: volteo horizontal, rotación ±{a['rotacion'] * 360:.0f}°, "
                 f"zoom ±{a['zoom']:.0%}, desplazamiento ±{a['traslacion']:.0%}, "
                 f"brillo ±{a['brillo']:.0%}, contraste ±{a['contraste']:.0%}", color=TINTA, fontsize=11)
    fig.tight_layout()
    ruta_fig = config.DIR_FIGURAS / "aumento_ejemplos.png"
    fig.savefig(ruta_fig, dpi=130, facecolor="white")
    plt.close(fig)
    print("  ", ruta_fig)


# -----------------------------------------------------------------------------
# 3. Splits
# -----------------------------------------------------------------------------
def preparar_splits(df):
    filas, guardar = [], {}
    splits = datos.splits_lopo(df) + [datos.split_aleatorio(df)]
    for s in splits:
        sin_fuga = s["nombre"] != "aleatorio"
        tabla = datos.comprobar_split(df, s, exigir_sin_fuga=sin_fuga)
        n = len(df)
        print(f"\n=== {s['nombre']} ===  (asserts OK"
              + ("" if sin_fuga else f"; ráfagas repartidas entre conjuntos = {tabla.attrs['rafagas_compartidas']}")
              + ")")
        if sin_fuga:
            for c in ["train", "val"]:
                print(f"  personas en {c}: {sorted(df['persona'].iloc[s[c]].unique())}")
        print(tabla.assign(**{"%": (100 * tabla["Total"] / n).round(1)}).to_string())
        for conjunto, fila in tabla.iterrows():
            filas.append({"split": s["nombre"], "conjunto": conjunto, **fila.to_dict(),
                          "porcentaje": round(100 * fila["Total"] / n, 1),
                          "rafagas_compartidas": tabla.attrs["rafagas_compartidas"]})
        guardar[s["nombre"]] = {c: [str(p.relative_to(config.RAIZ)) for p in map(config.Path, df["ruta"].iloc[s[c]])]
                                for c in ["train", "val", "test"]}
    pd.DataFrame(filas).to_csv(config.DIR_RESULTADOS / "splits_resumen.csv", index=False)
    with open(config.DIR_RESULTADOS / "splits.json", "w", encoding="utf-8") as f:
        json.dump(guardar, f, ensure_ascii=False, indent=1)
    return splits


# -----------------------------------------------------------------------------
def main():
    config.fijar_semillas()
    df = datos.cargar_listado()
    print(f"{len(df)} fotos, personas: {sorted(df['persona'].unique())}\n")

    print("##### 1. Grupos de ráfaga")
    analizar_rafagas(df)

    print("\n##### 2. Carga de imágenes (con caché)")
    for alto, ancho in config.TAMANOS_ALTERNATIVOS.values():
        for intento in ["1ª carga", "2ª carga (caché)"]:
            t = time.time()
            X, Y = datos.cargar_imagenes(df, alto, ancho)
            print(f"  {alto}x{ancho} {intento}: {time.time() - t:.1f} s  X={X.shape} {X.dtype} "
                  f"[{X.min():.2f}, {X.max():.2f}]  Y={Y.shape}")
    X, Y = datos.cargar_imagenes(df)
    assert (Y.argmax(1) == df["letra"].cat.codes.to_numpy()).all()
    print("  orden de las columnas de Y:", config.CLASES)
    figura_tamanos(df)

    print("\n##### 3. Splits")
    splits = preparar_splits(df)

    print("\n##### 4. Data augmentation")
    figura_aumento(X, df)

    print("\n##### 5. Prueba de tf.data con el primer fold")
    s = splits[0]
    aumento = datos.capas_aumento()
    ds_train = datos.crear_dataset(X, Y, s["train"], entrenamiento=True, aumento=aumento)
    ds_val = datos.crear_dataset(X, Y, s["val"], entrenamiento=False)
    ds_test = datos.crear_dataset(X, Y, s["test"], entrenamiento=False)
    for nombre, ds, idx in [("train", ds_train, s["train"]), ("val", ds_val, s["val"]), ("test", ds_test, s["test"])]:
        print(f"  {nombre:5}: {len(idx)} fotos, {len(ds)} batches, element_spec={ds.element_spec}")
    xb, yb = next(iter(ds_train))
    print(f"  batch de train: x {xb.shape} {xb.dtype} min={float(xb.numpy().min()):.3f} "
          f"max={float(xb.numpy().max()):.3f} media={float(xb.numpy().mean()):.3f}")
    print(f"  y {yb.shape} {yb.dtype}, fotos por letra en el batch: "
          f"{dict(zip(config.CLASES, yb.numpy().sum(0).astype(int)))}")
    xv, _ = next(iter(ds_val))
    xv2, _ = next(iter(ds_val))
    assert np.array_equal(xv.numpy(), xv2.numpy()), "La validación debe ser determinista (sin aumento ni barajado)"
    print("  validación determinista: OK")


if __name__ == "__main__":
    main()
