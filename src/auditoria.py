"""
auditoria.py
------------
Fase 1: auditoría del dataset (no modifica NINGUNA foto, solo las lee).

Genera:
  results/recuento_persona_letra.csv      recuento por persona y letra (con totales)
  results/resoluciones.csv                resoluciones por persona
  results/casi_identicas.csv              parejas de fotos casi idénticas (misma persona y letra)
  results/auditoria_phash_umbral.csv      nº de parejas por umbral de pHash (misma letra vs letra distinta)
  results/auditoria_resumen.json          números clave para la memoria
  results/figuras/auditoria_recuento.png
  results/figuras/auditoria_ejemplos_<persona>.png   ejemplos aleatorios por letra
  results/figuras/auditoria_casi_identicas.png
  results/figuras/revision/<persona>_<letra>.jpg     hojas de contacto con TODAS las fotos (revisión a ojo)

Uso:  cd src && python auditoria.py
"""

import itertools
import json

import imagehash
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import datos  # noqa: E402

# Colores por persona (paleta categórica validada, orden fijo; el color sigue a la persona)
COLORES_PERSONAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"

# Detección de casi idénticas
UMBRAL_PHASH_CANDIDATAS = 12   # distancia de Hamming máxima (sobre 64 bits) para considerar una pareja candidata
UMBRALES_PHASH_INFORME = [0, 2, 4, 6, 8, 10]
UMBRAL_DIF_PIXEL = 0.5         # diferencia (en desviaciones típicas) para contar un píxel como "cambiado"

N_EJEMPLOS_POR_LETRA = 8


def estilo_ejes(ax):
    """Ejes discretos: sin marco superior/derecho, rejilla horizontal suave."""
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_SECUNDARIA)
    ax.tick_params(colors=TINTA_SECUNDARIA, length=0)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# 1. Integridad, recuento y resoluciones
# ----------------------------------------------------------------------------
def comprobar_integridad(df):
    """Intenta decodificar cada foto entera. Devuelve la lista de rutas que no se pueden abrir."""
    corruptas = []
    for ruta in df["ruta"]:
        try:
            with Image.open(ruta) as img:
                img.load()
        except Exception:
            corruptas.append(ruta)
    return corruptas


def recuento(df):
    tabla = pd.crosstab(df["persona"], df["letra"], margins=True, margins_name="Total")
    tabla.to_csv(config.DIR_RESULTADOS / "recuento_persona_letra.csv")
    return tabla


def resoluciones(df):
    tabla = (df.groupby(["persona", "ancho", "alto"]).size().reset_index(name="fotos"))
    tabla["proporcion"] = (tabla["ancho"] / tabla["alto"]).round(3)
    tabla.to_csv(config.DIR_RESULTADOS / "resoluciones.csv", index=False)
    return tabla


def grafica_recuento(df):
    personas = sorted(df["persona"].unique())
    tabla = pd.crosstab(df["letra"], df["persona"]).reindex(columns=personas)
    totales = df["persona"].value_counts().reindex(personas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})
    fig.patch.set_facecolor("white")

    # (a) fotos por letra, una barra por persona
    n = len(personas)
    ancho = 0.8 / n
    x = np.arange(len(config.CLASES))
    for i, persona in enumerate(personas):
        ax1.bar(x + (i - (n - 1) / 2) * ancho, tabla[persona], width=ancho * 0.9,
                color=COLORES_PERSONAS[i], label=persona)
    ax1.set_xticks(x, config.CLASES)
    ax1.set_xlabel("Vocal", color=TINTA)
    ax1.set_ylabel("Número de fotos", color=TINTA)
    ax1.set_title("(a) Fotos por vocal y persona", color=TINTA, loc="left")
    ax1.legend(frameon=False, ncol=n, loc="upper left", bbox_to_anchor=(0, -0.13))
    estilo_ejes(ax1)

    # (b) total por persona
    barras = ax2.bar(personas, totales, width=0.6, color=COLORES_PERSONAS[:n])
    for b, v in zip(barras, totales):
        ax2.text(b.get_x() + b.get_width() / 2, v + 5, f"{v} ({v / len(df):.0%})",
                 ha="center", va="bottom", color=TINTA, fontsize=9)
    ax2.set_ylabel("Número de fotos", color=TINTA)
    ax2.set_title("(b) Total por persona", color=TINTA, loc="left")
    ax2.set_ylim(0, totales.max() * 1.15)
    estilo_ejes(ax2)

    fig.suptitle(f"Dataset: {len(df)} fotos de {n} personas", color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / "auditoria_recuento.png"
    fig.savefig(ruta, dpi=200, facecolor="white")
    plt.close(fig)
    return ruta


# ----------------------------------------------------------------------------
# 2. Ejemplos por persona y hojas de contacto
# ----------------------------------------------------------------------------
def miniatura(ruta, ancho=320):
    with Image.open(ruta) as img:
        img = img.convert("RGB")
        return img.resize((ancho, round(ancho * img.height / img.width)))


def cuadricula_ejemplos(df, persona, rng):
    sub = df[df["persona"] == persona]
    fig, ejes = plt.subplots(len(config.CLASES), N_EJEMPLOS_POR_LETRA,
                             figsize=(2.0 * N_EJEMPLOS_POR_LETRA, 1.35 * len(config.CLASES)))
    for i, letra in enumerate(config.CLASES):
        fotos = sub[sub["letra"] == letra]
        elegidas = fotos.sample(min(N_EJEMPLOS_POR_LETRA, len(fotos)), random_state=rng.integers(1e9))
        elegidas = elegidas.sort_values("orden")
        for j in range(N_EJEMPLOS_POR_LETRA):
            ax = ejes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(elegidas):
                fila = elegidas.iloc[j]
                ax.imshow(miniatura(fila["ruta"]))
                ax.set_title(f"#{fila['orden']}", fontsize=7, color=TINTA_SECUNDARIA, pad=2)
        ejes[i, 0].set_ylabel(letra, rotation=0, fontsize=16, labelpad=18, va="center", color=TINTA)
    fig.suptitle(f"{persona}: ejemplos aleatorios de cada vocal (nº = orden de captura dentro de la carpeta)",
                 color=TINTA, fontsize=11)
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / f"auditoria_ejemplos_{persona}.png"
    fig.savefig(ruta, dpi=130, facecolor="white")
    plt.close(fig)
    return ruta


def hojas_de_contacto(df, columnas=10, ancho=192):
    """Una imagen por persona y letra con TODAS sus fotos y el nº de orden, para revisar las etiquetas a ojo."""
    carpeta = config.DIR_FIGURAS / "revision"
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for (persona, letra), grupo in df.groupby(["persona", "letra"], observed=True):
        grupo = grupo.sort_values("orden")
        alto = round(ancho * 9 / 16)
        filas = int(np.ceil(len(grupo) / columnas))
        hoja = Image.new("RGB", (columnas * (ancho + 4), filas * (alto + 18) + 30), "white")
        dibujo = ImageDraw.Draw(hoja)
        dibujo.text((6, 6), f"{persona} - {letra} ({len(grupo)} fotos)", fill="black")
        for k, fila in enumerate(grupo.itertuples()):
            x = (k % columnas) * (ancho + 4)
            y = 30 + (k // columnas) * (alto + 18)
            hoja.paste(miniatura(fila.ruta, ancho).resize((ancho, alto)), (x, y))
            dibujo.text((x + 2, y + alto + 2), f"#{fila.orden}", fill="black")
        ruta = carpeta / f"{persona}_{letra}.jpg"
        hoja.save(ruta, quality=85)
        rutas.append(ruta)
    return rutas


# ----------------------------------------------------------------------------
# 3. Fotos casi idénticas
# ----------------------------------------------------------------------------
def calcular_huellas(df):
    """Para cada foto: hash perceptual (pHash, 64 bits) y una versión pequeña en gris con el contraste
    normalizado (media 0, desviación 1), que sirve para medir cuántos píxeles cambian entre dos fotos."""
    hashes, pequenas = [], []
    for ruta in df["ruta"]:
        with Image.open(ruta) as img:
            img = img.convert("RGB")
            hashes.append(imagehash.phash(img))
            gris = img.convert("L").resize((160, 90)).filter(ImageFilter.GaussianBlur(1))
            g = np.asarray(gris, dtype=np.float32)
            pequenas.append((g - g.mean()) / (g.std() + 1e-6))
    return hashes, pequenas


def comparar_parejas(df, hashes, pequenas):
    """Compara todas las parejas de fotos de la misma persona (de la misma letra y de letras distintas).
    Las parejas de letras distintas sirven de CONTROL: nunca pueden ser la misma foto."""
    filas = []
    for persona, grupo in df.groupby("persona"):
        for a, b in itertools.combinations(grupo.index, 2):
            d_hash = hashes[a] - hashes[b]
            fila = {
                "persona": persona,
                "letra_1": df.at[a, "letra"], "orden_1": df.at[a, "orden"], "archivo_1": df.at[a, "archivo"],
                "letra_2": df.at[b, "letra"], "orden_2": df.at[b, "orden"], "archivo_2": df.at[b, "archivo"],
                "misma_letra": df.at[a, "letra"] == df.at[b, "letra"],
                "dist_phash": int(d_hash),
                "frac_pixeles_cambiados": np.nan,
            }
            if d_hash <= UMBRAL_PHASH_CANDIDATAS:
                fila["frac_pixeles_cambiados"] = float(
                    np.mean(np.abs(pequenas[a] - pequenas[b]) > UMBRAL_DIF_PIXEL))
            filas.append(fila)
    return pd.DataFrame(filas)


def figura_casi_identicas(df, parejas, control):
    """Fila superior: parejas casi idénticas confirmadas. Fila inferior: parejas de letras DISTINTAS que el pHash
    considera muy parecidas (muestra que el pHash de la imagen completa mide sobre todo el fondo)."""
    indice = df.set_index(["persona", "letra", "orden"])
    seleccion = [("Casi idénticas (misma letra)", fila) for _, fila in parejas.head(4).iterrows()]
    seleccion += [("Control: letras distintas, pHash muy parecido", fila) for _, fila in control.head(2).iterrows()]

    fig, ejes = plt.subplots(len(seleccion), 2, figsize=(7.5, 2.25 * len(seleccion)))
    for i, (tipo, fila) in enumerate(seleccion):
        for j, k in enumerate(["1", "2"]):
            ruta = indice.loc[(fila["persona"], fila[f"letra_{k}"], fila[f"orden_{k}"]), "ruta"]
            ax = ejes[i, j]
            ax.imshow(miniatura(ruta, 480))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{fila['persona']} – {fila[f'letra_{k}']} #{fila[f'orden_{k}']}", fontsize=9, color=TINTA)
        ejes[i, 0].set_ylabel(f"pHash = {fila['dist_phash']}\npíxeles\ncambiados\n= {fila['frac_pixeles_cambiados']:.1%}",
                              rotation=0, fontsize=8, labelpad=38, va="center", color=TINTA_SECUNDARIA)
        if i in (0, len(parejas.head(4))):
            ejes[i, 0].text(0, 1.28, tipo, transform=ejes[i, 0].transAxes, fontsize=10, color=TINTA, weight="bold")
    fig.tight_layout()
    ruta = config.DIR_FIGURAS / "auditoria_casi_identicas.png"
    fig.savefig(ruta, dpi=150, facecolor="white")
    plt.close(fig)
    return ruta


def grupos_de_casi_identicas(parejas):
    """Une las parejas en grupos (si A≈B y B≈C, {A,B,C} es un grupo). Fotos 'redundantes' = las que sobrarían
    si se dejara solo una por grupo."""
    padre = {}

    def raiz(x):
        while padre.setdefault(x, x) != x:
            x = padre[x]
        return x

    for _, f in parejas.iterrows():
        a = (f["persona"], f["letra_1"], f["orden_1"])
        b = (f["persona"], f["letra_2"], f["orden_2"])
        padre[raiz(a)] = raiz(b)
    grupos = {}
    for x in list(padre):
        grupos.setdefault(raiz(x), []).append(x)
    return list(grupos.values())


# ----------------------------------------------------------------------------
def main():
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEMILLA)
    resumen = {}

    df = datos.cargar_listado()
    personas = sorted(df["persona"].unique())
    print(f"Personas detectadas: {personas}")
    resumen["personas"] = personas
    resumen["total_fotos"] = len(df)

    corruptas = comprobar_integridad(df)
    print(f"Fotos que no se pueden abrir: {len(corruptas)}")
    resumen["fotos_corruptas"] = len(corruptas)

    tabla = recuento(df)
    print("\nRecuento por persona y letra:\n", tabla, sep="")
    resumen["recuento"] = {p: {str(k): int(v) for k, v in fila.items()} for p, fila in tabla.iterrows()}

    res = resoluciones(df)
    print("\nResoluciones por persona:\n", res.to_string(index=False), sep="")
    resumen["resoluciones"] = res.to_dict(orient="records")

    # Ráfagas: fotos hechas en el mismo segundo o en segundos consecutivos (según el nombre del archivo)
    # (datos.cargar_listado ya calcula la hora de captura y los segundos desde la foto anterior)
    rafagas = df.groupby("persona")["segundos_desde_anterior"].agg(
        mediana_segundos="median",
        mismo_segundo=lambda s: int((s == 0).sum()),
        a_1_segundo_o_menos=lambda s: int((s <= 1).sum()))
    print("\nTiempo entre fotos consecutivas (misma persona y letra):\n", rafagas, sep="")
    resumen["rafagas"] = rafagas.reset_index().to_dict(orient="records")

    print("\nGráficas de recuento y ejemplos...")
    print(" ", grafica_recuento(df))
    for persona in personas:
        print(" ", cuadricula_ejemplos(df, persona, rng))
    hojas = hojas_de_contacto(df)
    print(f"  {len(hojas)} hojas de contacto en {config.DIR_FIGURAS / 'revision'}")

    print("\nCalculando hashes perceptuales...")
    hashes, pequenas = calcular_huellas(df)
    todas = comparar_parejas(df, hashes, pequenas)
    misma = todas[todas["misma_letra"]]
    distinta = todas[~todas["misma_letra"]]

    # (a) Recuento por umbral de pHash: misma letra frente al control (letras distintas)
    filas_umbral = []
    for u in UMBRALES_PHASH_INFORME:
        filas_umbral.append({"umbral_phash": u,
                             "parejas_misma_letra": int((misma["dist_phash"] <= u).sum()),
                             "parejas_letra_distinta": int((distinta["dist_phash"] <= u).sum())})
    tabla_umbral = pd.DataFrame(filas_umbral)
    tabla_umbral.to_csv(config.DIR_RESULTADOS / "auditoria_phash_umbral.csv", index=False)
    print("\nParejas con distancia pHash <= umbral (sobre 64 bits):\n", tabla_umbral.to_string(index=False), sep="")
    resumen["phash_por_umbral"] = filas_umbral
    resumen["parejas_comparadas"] = {"misma_letra": len(misma), "letra_distinta": len(distinta)}

    # (b) Casi idénticas confirmadas: más parecidas que CUALQUIER pareja de letras distintas
    umbral_pixeles = float(distinta["frac_pixeles_cambiados"].min())
    casi = (misma[misma["frac_pixeles_cambiados"] < umbral_pixeles]
            .sort_values("frac_pixeles_cambiados").reset_index(drop=True))
    casi.to_csv(config.DIR_RESULTADOS / "casi_identicas.csv", index=False)
    grupos = grupos_de_casi_identicas(casi)
    redundantes = sum(len(g) - 1 for g in grupos)
    print(f"\nUmbral de píxeles cambiados (mínimo entre letras distintas): {umbral_pixeles:.2%}")
    print(f"Parejas casi idénticas confirmadas: {len(casi)} -> {len(grupos)} grupos, "
          f"{redundantes} fotos redundantes ({redundantes / len(df):.1%} del total)")
    if len(casi):
        print(casi[["persona", "letra_1", "orden_1", "orden_2", "dist_phash", "frac_pixeles_cambiados"]]
              .to_string(index=False))
    resumen["casi_identicas"] = {
        "umbral_frac_pixeles": umbral_pixeles,
        "parejas": len(casi), "grupos": len(grupos), "fotos_redundantes": redundantes,
        "por_persona": casi["persona"].value_counts().to_dict(),
        "consecutivas": int((abs(casi["orden_1"] - casi["orden_2"]) == 1).sum()),
    }

    control = distinta.sort_values("dist_phash").head(2)
    print(" ", figura_casi_identicas(df, casi, control))

    with open(config.DIR_RESULTADOS / "auditoria_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    print("\nResumen guardado en", config.DIR_RESULTADOS / "auditoria_resumen.json")


if __name__ == "__main__":
    main()
