# Notas para el informe – Proyecto P3 (LSE, vocales con CNN)

Material para redactar la memoria, apuntado al terminar cada fase: qué se hizo, por qué, resultados y números
concretos, figuras (con su ruta), problemas encontrados y cómo se resolvieron, e ideas para la memoria.
Todos los números deben salir de `results/` o de comprobaciones reproducibles.

---

## Fase 0 – Entorno y estructura

### Entorno experimental (apartado de Metodología)

Todos los experimentos se han ejecutado en local en el siguiente equipo:

| Componente | Detalle |
|---|---|
| Equipo | Portátil Lenovo LOQ 15ARP9 |
| Sistema operativo | Ubuntu 22.04.5 LTS (núcleo Linux 6.8.0-149-generic) |
| CPU | AMD Ryzen 7 7435HS (8 núcleos / 16 hilos) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB de VRAM (arquitectura Ada Lovelace, *compute capability* 8.9) |
| Controlador NVIDIA | 580.178.04 (admite hasta CUDA 13.0) |
| RAM | 24 GB (23 GiB útiles) |

Software (entorno conda `lse`, dependencias exactas en `requirements.txt`):

| Paquete | Versión |
|---|---|
| Python | 3.11.18 |
| TensorFlow | 2.21.0 (`tensorflow[and-cuda]`) |
| Keras | 3.15.1 |
| CUDA | 12.9 en tiempo de ejecución (`nvidia-cuda-runtime-cu12` 12.9.79); TensorFlow compilado con CUDA 12.5.1 |
| cuDNN | 9.28.1 (`nvidia-cudnn-cu12`); TensorFlow compilado con cuDNN 9 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| OpenCV (`opencv-python`) | 5.1.0 |
| pandas / matplotlib / seaborn | 3.1.1 / 3.11.2 / 0.13.2 |
| Pillow / ImageHash | 13.0.0 / 4.3.2 |

Texto propuesto para la memoria:
> Los modelos se implementaron en Python 3.11 con TensorFlow 2.21 y Keras 3 (API `Sequential`). El entrenamiento se
> realizó en una GPU NVIDIA GeForce RTX 4060 Laptop (8 GB) con CUDA 12.9 y cuDNN 9, en un portátil con un AMD Ryzen 7
> 7435HS y 24 GB de RAM bajo Ubuntu 22.04. Para el preprocesamiento y la evaluación se usaron NumPy, OpenCV, pandas,
> scikit-learn, matplotlib y seaborn. Todas las semillas aleatorias (Python, NumPy y TensorFlow) se fijaron a 42 y se
> activó el determinismo de operaciones de TensorFlow para que los experimentos sean reproducibles.

### Qué se hizo
- Se creó el entorno conda `lse` (Python 3.11) sin tocar `base`, con TensorFlow para GPU y las librerías del proyecto.
- Se comprobó que TensorFlow usa la GPU: un producto de matrices 4000×4000 repetido 10 veces tarda **0,16 s en GPU
  frente a 1,61 s en CPU** (unas 10 veces más rápido), y las capas `Conv2D` se ejecutan en la GPU.
- Estructura de carpetas: `src/`, `notebooks/`, `models/`, `results/figuras/`, `docs/memoria/`.
- `src/config.py`: rutas, clases (A, E, I, O, U), tamaño de imagen 128, semilla 42, hiperparámetros por defecto,
  `detectar_personas()` (detecta las personas leyendo las carpetas: hoy David, Isaac y Rodrigo; Juan entrará solo)
  y `fijar_semillas()`.
- `.gitignore` (excluye fotos, `models/` y temporales), `requirements.txt`, `docs/decisiones.md` y este archivo.

### Problemas encontrados y solución
- **TensorFlow 2.21 no detectaba la GPU** aunque `tensorflow[and-cuda]` instala todas las librerías de CUDA.
  Causa: un fallo de empaquetado de esa versión. Las rutas de búsqueda de librerías (RUNPATH) de TensorFlow no incluyen
  la carpeta de cuSOLVER, así que no encontraba `libcusolver.so.11` y desactivaba la GPU.
  Solución, solo dentro del entorno `lse`:
  1. Enlace simbólico de `libcusolver.so.11` (y `libcusolverMg.so.11`) en `site-packages/nvidia/cusparse/lib/`, que
     sí está en el RUNPATH. Así funciona aunque no se active el entorno (Jupyter, IDE).
  2. Script en `$CONDA_PREFIX/etc/conda/activate.d/` que añade la carpeta de cuSOLVER a `LD_LIBRARY_PATH` al activar `lse`.
  Si se reinstala TensorFlow, hay que repetir el arreglo.

### Ideas para la memoria
- Justificar el uso de la GPU: el leave-one-person-out obliga a entrenar un modelo por persona (y varios experimentos
  de ajuste), lo que en CPU sería mucho más lento.
- Comentar la reproducibilidad: semillas fijas, determinismo de TF, versiones exactas en `requirements.txt`.

---

## Fase 1 – Captura y auditoría del dataset

### Protocolo de captura
- Dataset capturado por el propio grupo (no se ha descargado ninguno). Cada persona fotografió las cinco vocales
  A, E, I, O, U del alfabeto dactilológico de la Lengua de Signos Española con la **webcam de su portátil**
  (aplicación Cámara de Windows; los nombres `WIN_AAAAMMDD_HH_MM_SS_Pro.jpg` guardan la hora de captura).
- Por persona y letra, las fotos se hicieron en **4 bloques**: luz natural / luz de bombilla × fondo liso / fondo con
  objetos. Dentro de cada bloque se varió el ángulo, la distancia a la cámara, la posición de la mano en el encuadre
  y, a veces, la mano usada (derecha/izquierda).
- Se buscaba que solo saliera la mano. **En la práctica, en bastantes fotos de Rodrigo sale su cara y en algunas de
  Isaac el borde de la cara** (ver `results/figuras/auditoria_ejemplos_*.png`). No afecta a la clase (la cara aparece en
  todas las letras), pero hay que tenerlo en cuenta al elegir las figuras de la memoria (privacidad).
- Las fotos se guardan en `<Persona>/<Letra>/` y se tratan como **solo lectura**. La carga (`src/datos.py`,
  `cargar_listado()`) detecta las personas leyendo las carpetas, así que las fotos de Juan se añadirán sin tocar el código.

### Recuento (3 personas; Juan pendiente) – `results/recuento_persona_letra.csv`
| Persona | A | E | I | O | U | Total | % |
|---|---|---|---|---|---|---|---|
| David | 86 | 103 | 111 | 91 | 104 | 495 | 49 % |
| Isaac | 47 | 36 | 42 | 48 | 52 | 225 | 22 % |
| Rodrigo | 49 | 55 | 60 | 58 | 59 | 281 | 28 % |
| **Total** | 182 | 194 | 213 | 197 | 215 | **1001** | |

- Clases equilibradas: la menor (A, 182) tiene un 85 % de las fotos de la mayor (U, 215).
- Personas desequilibradas: David aporta el 49 % de las fotos.
- **0 fotos corruptas** (las 1001 se decodifican enteras). Todas en JPG y sin rotación EXIF.

### Resoluciones – `results/resoluciones.csv`
| Persona | Resolución | Fotos |
|---|---|---|
| David | 1280×720 | 495 |
| Isaac | 1280×720 | 225 |
| Rodrigo | 2560×1440 | 281 |

Todas en proporción 16:9 (1,778), así que se unifican al redimensionar. Rodrigo tiene el doble de resolución
(otra webcam): conviene comprobar en la Fase 5 si su fold se comporta distinto.

### Fotos casi idénticas – `results/casi_identicas.csv`, `results/auditoria_phash_umbral.csv`
Método (`src/auditoria.py`):
1. **Hash perceptual (pHash, 64 bits, `imagehash`)** de cada foto y distancia de Hamming entre todas las parejas de la
   misma persona: 37 285 parejas de la misma letra y 149 520 de letras distintas.
2. **Control:** las parejas de **letras distintas** no pueden ser la misma foto, así que sirven para medir qué detecta
   realmente el pHash.

| Umbral pHash (≤) | Parejas misma letra | Parejas letra distinta (control) |
|---|---|---|
| 0 | 4 | 0 |
| 2 | 17 | 20 |
| 4 | 78 | 99 |
| 6 | 222 | 276 |
| 8 | 480 | 586 |
| 10 | 925 | 1168 |

**Hallazgo:** a partir de distancia 2 hay tantas parejas "parecidas" entre letras distintas como dentro de la misma
letra. El pHash de la imagen completa mide sobre todo **la escena (fondo e iluminación)**: la mano ocupa una parte
pequeña del encuadre. Por eso, un umbral de pHash por sí solo no sirve para detectar fotos repetidas.

3. **Confirmación:** para las parejas con pHash ≤ 12 se calcula la **fracción de píxeles que cambian**. Se usa una versión
   160×90 en gris, suavizada y con el contraste normalizado (media 0, desviación 1) para que las fotos oscuras con
   bombilla no engañen, y se cuenta un píxel como cambiado si la diferencia es > 0,5 desviaciones. El umbral sale
   de los datos: una pareja es "casi idéntica" si se parece más que **cualquier** pareja de letras distintas del dataset
   (mínimo observado entre letras distintas: **2,08 %** de píxeles cambiados).

**Resultado: 4 parejas casi idénticas** (4 fotos redundantes, **0,4 %** del dataset): Rodrigo U #27–#28 (0,9 %),
David A #50–#55 (1,3 %), David E #9–#10 (1,5 %), David A #56–#62 (1,9 %). No se ha borrado ninguna.

### Ráfagas (según la hora del nombre de archivo)
Tiempo entre fotos consecutivas de la misma persona y letra:

| Persona | Mediana | Mismo segundo | ≤ 1 s (de las parejas consecutivas) |
|---|---|---|---|
| David | 1 s | 29 | 248 de 490 (51 %) |
| Isaac | 2 s | 0 | 64 de 220 (29 %) |
| Rodrigo | 2 s | 3 | 118 de 276 (43 %) |

Las fotos se hicieron a ritmo de ~1 por segundo: fotos consecutivas son **muy parecidas aunque no idénticas** (mismo
fondo, luz y postura, con la mano un poco desplazada). David hizo más fotos sobre todo porque disparó más rápido, así que
parte de su ventaja en número no es variedad real.

### Figuras generadas
- `results/figuras/auditoria_recuento.png`: (a) fotos por vocal y persona, (b) total por persona. **Para la memoria.**
- `results/figuras/auditoria_ejemplos_David.png`, `..._Isaac.png`, `..._Rodrigo.png`: 8 ejemplos aleatorios por letra
  (semilla 42). Para la memoria (ojo: en la de Rodrigo sale su cara).
- `results/figuras/auditoria_casi_identicas.png`: 4 parejas casi idénticas y 2 parejas de control (letras distintas con
  pHash = 2). Buena figura para explicar por qué el pHash global no basta.
- `results/figuras/revision/<Persona>_<Letra>.jpg`: hojas de contacto con **todas** las fotos (para revisar las etiquetas
  a ojo; no son para la memoria).
- Números clave en `results/auditoria_resumen.json`.

### Problemas encontrados y solución
- Había una copia antigua del dataset (`dataset_limpio/`, 1001 fotos, mismos recuentos) y su script. Tras comprobar que
  los originales estaban intactos (1001 fotos, recuentos idénticos a los del CLAUDE.md), se borraron. Desde ahora se lee
  directamente de las carpetas originales, sin copias intermedias.
- La carpeta de la O de Isaac se llama `o/` (minúscula): se lee sin distinguir mayúsculas.
- Un primer intento de confirmar las casi idénticas con un umbral fijo de nivel de gris fallaba con las fotos oscuras
  (luz de bombilla): no veía la mano. Se solucionó normalizando el contraste de cada foto antes de comparar.

### Ideas para la memoria
- La mano ocupa poco del encuadre (en muchas fotos de David, ~10 % del ancho). Al reducir a 128×128 con padding,
  la imagen útil queda en 128×72 y la mano en ~15 píxeles. Es un argumento para discutir el tamaño de entrada (Fase 2).
- Las fotos con bombilla de David son muy oscuras y con poco contraste: justifican el aumento de brillo/contraste.
- El fondo y la luz se repiten en las cinco letras de cada persona (los 4 bloques), así que la red no puede acertar la
  clase "por el fondo". Es un punto a favor del protocolo de captura.
- La correlación temporal (ráfagas) es el verdadero riesgo de fuga de información si se mezclan fotos de la misma
  persona entre entrenamiento y validación/test, más que las fotos idénticas (solo 4).

---

## Fase 2 – Preprocesamiento y preparación del dataset

### Tamaño de entrada: 144×256 (alto × ancho), RGB
- Las fotos son 16:9 (1280×720 o 2560×1440). Se reducen a **144×256**, que mantiene exactamente esa proporción:
  **sin recortar** (la mano está a menudo en un lateral o pegada al borde) **ni deformar**.
- Motivo: la mano ocupa poco del encuadre. Con 128×128 y padding, la imagen útil queda en 128×72 y la mano en unos pocos
  píxeles. Con 144×256 tiene el doble de resolución en cada eje (ver `results/figuras/preprocesado_tamanos.png`).
- 128×128 con padding (bandas negras arriba y abajo) se mantiene como **experimento de comparación** (Fase 4).
- No se recorta la mano con un detector (p. ej. MediaPipe): se propone como mejora futura.
- Implementación: `tf.image.resize_with_pad(..., antialias=True)` para los dos tamaños; con 144×256 no añade bandas. El
  antialias evita el aliasing al reducir 10 veces las fotos de Rodrigo, que si no tendrían un "ruido" distinto
  que la red podría aprender como rasgo de la persona.

### Normalización y etiquetas
- Píxeles divididos entre 255 → **[0, 1]** (float32). Etiquetas one-hot con `to_categorical`, columnas en el orden
  **A, E, I, O, U**.
- Caché en disco (`cache/imagenes_144x256.npz`, uint8): la primera carga tarda ~18 s y las siguientes ~0,2 s. Se regenera
  sola si cambia la lista de fotos.

### Data augmentation (solo en entrenamiento)
Capas de Keras aplicadas en el pipeline `tf.data` a cada batch de entrenamiento (nunca a validación ni test):

| Capa | Rango | Justificación |
|---|---|---|
| `RandomFlip` | horizontal | Equivale a hacer el gesto con la otra mano (en el dataset ya hay de ambas). |
| `RandomRotation` | ±10° | Inclinaciones naturales de la mano; más giro podría alterar el gesto. |
| `RandomZoom` | ±10 % | Distintas distancias a la cámara; moderado para no sacar la mano del encuadre. |
| `RandomTranslation` | ±10 % | La mano puede estar en cualquier zona del encuadre. |
| `RandomBrightness` | ±0,2 | Luz natural / bombilla, webcams distintas. |
| `RandomContrast` | 0,8-1,2 | Fotos con bombilla muy apagadas. |

- Las zonas que quedan vacías al girar o desplazar se rellenan de negro (`fill_mode='constant'`). El modo por defecto
  (`reflect`) duplicaría la mano en espejo cuando está pegada al borde.
- Brillo y contraste con `value_range=(0, 1)` (por defecto Keras supone [0, 255]).
- Figura: `results/figuras/aumento_ejemplos.png` (una foto original sin cara y 8 versiones aumentadas).

### Grupos de ráfaga
- **Definición:** fotos consecutivas de la misma persona y letra separadas por **≤ 5 s** (la hora sale del nombre del
  archivo, `WIN_AAAAMMDD_HH_MM_SS_Pro.jpg`). Una pausa de más de 5 s empieza una ráfaga nueva.
- **Justificación del umbral** (`results/figuras/rafagas_histograma.png`, `results/rafagas_umbral.csv`): de los 986
  intervalos entre fotos consecutivas, el 87,4 % dura ≤ 5 s (pico en 1-2 s: se disparaba a ~1 foto/s). Por encima de
  5 s la distribución se aplana en una cola larga, y los cambios de bloque superan los 20 s.

| Umbral (s) | % intervalos dentro | Grupos | Tamaño mediano | Mín. grupos por persona-letra |
|---|---|---|---|---|
| 1 | 43,6 | 571 | 1 | 24 |
| 2 | 73,4 | 277 | 2 | 9 |
| 3 | 83,0 | 183 | 3 | 4 |
| **5** | **87,4** | **139** | **5** | **4** |
| 10 | 91,4 | 100 | 10 | 3 |
| 30 | 94,4 | 70 | 12,5 | 3 |
| 300 | 96,0 | 54 | 16 | 3 |

- **Grupos con 5 s** (`results/rafagas_por_persona_letra.csv`): 139 en total (mediana 5 fotos, máximo 21; 25 de una foto).

| Persona | A | E | I | O | U | Total |
|---|---|---|---|---|---|---|
| David | 24 | 17 | 9 | 16 | 9 | 75 |
| Isaac | 14 | 5 | 8 | 5 | 6 | 38 |
| Rodrigo | 8 | 6 | 4 | 4 | 4 | 26 |
| **Total** | 46 | 28 | 21 | 25 | 19 | **139** |

### Esquema de splits
- **Principal: leave-one-person-out** (un fold por persona). Test = todas las fotos de una persona; validación ≈ 1/6 de
  las fotos de las otras personas, **por grupos de ráfaga** (`StratifiedGroupKFold`, estratos = persona+letra); el resto,
  entrenamiento. Como con pocos grupos una partición cualquiera puede quedar desequilibrada (la primera dejaba 4 fotos de
  I y 27 de O en la validación del fold de David), se prueban 50 semillas × 6 particiones y se elige la más equilibrada
  por letra y persona.
- **Comprobaciones automáticas (asserts):** ninguna foto ni grupo de ráfaga se repite entre train, val y test; la persona
  de test no aparece en train ni val; entre los tres conjuntos están todas las fotos.
- **Tamaños** (`results/splits_resumen.csv`; fotos concretas de cada conjunto en `results/splits.json`):

| Fold (test) | Conjunto | A | E | I | O | U | Total | Personas |
|---|---|---|---|---|---|---|---|---|
| David | train | 85 | 79 | 81 | 88 | 87 | 420 | Isaac 188, Rodrigo 232 |
| | val | 11 | 12 | 21 | 18 | 24 | 86 | Isaac 37, Rodrigo 49 |
| | test | 86 | 103 | 111 | 91 | 104 | 495 | David |
| Isaac | train | 118 | 138 | 138 | 120 | 130 | 644 | David 411, Rodrigo 233 |
| | val | 17 | 20 | 33 | 29 | 33 | 132 | David 84, Rodrigo 48 |
| | test | 47 | 36 | 42 | 48 | 52 | 225 | Isaac |
| Rodrigo | train | 110 | 116 | 130 | 113 | 131 | 600 | David 413, Isaac 187 |
| | val | 23 | 23 | 23 | 26 | 25 | 120 | David 82, Isaac 38 |
| | test | 49 | 55 | 60 | 58 | 59 | 281 | Rodrigo |
| Aleatorio | train | 127 | 136 | 149 | 137 | 151 | 700 | todas |
| | val | 27 | 29 | 32 | 30 | 32 | 150 | todas |
| | test | 28 | 29 | 32 | 30 | 32 | 151 | todas |

- El fold de David es el más difícil: se entrena con solo 420 fotos (dos personas con menos datos) y se testea con 495.

### Fuga de información (para el apartado de metodología)
- **Qué es:** el modelo se evalúa con datos que, directa o indirectamente, ha visto al entrenar, y la métrica sale
  optimista: mide memorización y no generalización.
- **En este dataset hay dos fuentes:**
  1. **La persona:** misma mano, piel, fondos, iluminación y webcam. Si una persona está en train y en test, el modelo
     puede acertar por reconocer su "estilo". El objetivo real es reconocer el gesto de una persona **nueva**, y eso es
     lo que mide leave-one-person-out.
  2. **Las ráfagas:** fotos hechas con 1-2 s de diferencia son casi iguales. En el **split aleatorio foto a foto,
     100 de las 139 ráfagas quedan repartidas entre train, val y test**: el test contiene "casi copias" de fotos de
     entrenamiento.
- Por eso la validación (que decide cuándo parar con EarlyStopping) también se saca por ráfagas: si no, también saldría
  optimista.
- El split aleatorio se mantiene **solo como comparación** (Fase 5): se espera una accuracy claramente mayor que en LOPO,
  y esa diferencia cuantifica la fuga.

### Prueba del pipeline (fold de David)
- `tf.data`: train 420 fotos (13 batches de 32, barajado + aumento), val 86 (3 batches), test 495 (16 batches).
- Batch: x `(32, 144, 256, 3)` float32 en [0, 1]; y `(32, 5)` float32. La validación es determinista (dos pasadas iguales).

### Figuras generadas
- `results/figuras/rafagas_histograma.png`: tiempo entre fotos consecutivas y umbral de 5 s. **Para la memoria.**
- `results/figuras/preprocesado_tamanos.png`: original frente a 144×256 y 128×128 con padding. **Para la memoria.**
- `results/figuras/aumento_ejemplos.png`: original y 8 versiones aumentadas (foto de David, sin cara). **Para la memoria.**

### Problemas encontrados y solución
- Valores por defecto peligrosos de Keras 3: `RandomBrightness`/`RandomContrast` suponen [0, 255] (se pasa
  `value_range=(0, 1)`) y las capas geométricas rellenan con `reflect` (se cambia a `constant`).
- `resize_with_pad` no usa antialias por defecto: se activa.
- Validación desequilibrada por letras con la primera partición de `StratifiedGroupKFold`: se elige la más equilibrada
  entre varias semillas.
- `to_categorical` devuelve float64: se convierte a float32.
- TF muestra un aviso inofensivo al usar `map` en `tf.data` ("NodeDef mentions attribute use_unbounded_threadpool").

### Ideas para la memoria
- El histograma de ráfagas es una forma visual y cuantitativa de justificar la validación por grupos.
- La comparación LOPO frente a aleatorio de la Fase 5 se apoyará en el dato de las 100 ráfagas repartidas.
- Al llegar Juan: repetir `auditoria.py` y `preparar_datos.py`; habrá 4 folds y la caché se regenerará sola.

---

## Fase 3 – Arquitectura CNN

Código: `src/modelos.py` (API `Sequential` + `model.add`, como en los apuntes) y `src/fase3_arquitectura.py` (análisis y
comprobaciones). Entrada 144×256×3 en [0, 1]; salida `Dense(5, softmax)`. El data augmentation va en el pipeline
`tf.data` (Fase 2), no dentro del modelo.

### Modelos
| Modelo | Descripción |
|---|---|
| **M0** baseline de los apuntes | Conv2D(32, 3×3, relu, same) → MaxPool 2×2 → Conv2D(64) → MaxPool → Flatten → Dense(128, relu) → Dense(5, softmax) |
| **M1** más profundo | 4 bloques [Conv2D 3×3 → BN → ReLU]×2 → MaxPool 2×2 → Dropout(0,2), con 32-64-128-256 filtros → Flatten → Dense(128, relu) → Dropout(0,5) → Dense(5, softmax) |
| **M2** (principal) | Igual que M1 pero con **GlobalAveragePooling2D** en lugar de Flatten |
| **M3** | M2 + regularización L2 (λ = 1e-4) en convoluciones y densa |

Todos se crean con funciones con parámetros (filtros, convoluciones por bloque, dropout, unidades, L2, cabeza,
momentum de BN) para ajustarlos en la Fase 4.

### Parámetros y mapas de características (`results/modelos_comparativa.csv`, `results/modelos/summary_M*.txt`)
| Modelo | Parámetros totales | Convolucionales | BatchNorm | Densos | % densos | Mapas tras cada bloque | Entrada a la densa |
|---|---|---|---|---|---|---|---|
| M0 | 18 894 533 | 19 392 | 0 | 18 875 141 | 99,9 % | 72×128×32 → 36×64×64 | 147 456 |
| M1 | 5 894 501 | 1 171 296 | 3 840 | 4 719 365 | 80,1 % | 72×128×32 → 36×64×64 → 18×32×128 → 9×16×256 | 36 864 |
| M2 | 1 208 677 | 1 171 296 | 3 840 | 33 541 | 2,8 % | (igual que M1) | 256 |
| M3 | 1 208 677 | 1 171 296 | 3 840 | 33 541 | 2,8 % | (igual que M1) | 256 |

- De los 3 840 parámetros de BN, 1 920 son entrenables (γ, β) y 1 920 no (medias y varianzas móviles).
- **El problema de Flatten con imágenes grandes:** en M0, tras solo 2 poolings el mapa es 36×64×64 = 147 456 valores.
  La `Dense(128)` que va detrás tiene 147 456 × 128 + 128 = **18,87 M** pesos, el 99,9 % del modelo, mientras que la
  parte que "ve" la imagen (las convoluciones) son 19 k. Con 4 bloques (M1) baja a 4,7 M; con GAP (M2), a 33 k.
- Cálculo de ejemplo (para la memoria): una `Conv2D` 3×3 con C_in canales de entrada y C_out filtros tiene
  3·3·C_in·C_out pesos (+ C_out bias si no va seguida de BN). Bloque 4, segunda conv: 3·3·256·256 = 589 824.

### Campo receptivo (`results/campo_receptivo.csv`)
Fórmula: r ← r + (k − 1)·salto ; salto ← salto·stride (empezando en r = 1, salto = 1).

| Modelo | Campo receptivo de la última conv | Salto (stride acumulado) |
|---|---|---|
| M0 | **8 px** | 2 px |
| M1/M2/M3 con 1 conv por bloque | **38 px** | 8 px |
| M1/M2/M3 con 2 conv por bloque (elegido) | **68 px** | 8 px |

- La mano mide unos 30 px de ancho en las fotos donde está lejos y unos 65 px en las más cercanas (medido a ojo en
  `results/figuras/preprocesado_tamanos.png`), sobre una imagen de 256 px de ancho.
- **¿"Ve" cada neurona la mano entera?** En M0, no: 8 px es un trozo de dedo, y toda la composición queda para la densa
  (que memoriza posiciones). Con 1 conv por bloque, 38 px apenas cubre una mano lejana. Con 2 conv por bloque, 68 px
  cubre la mano entera en casi todas las fotos. Es la justificación principal de las dos convoluciones por bloque.
- Matiz para la memoria: el campo receptivo **efectivo** es menor que el teórico, porque los píxeles del centro pesan
  mucho más que los del borde (Luo et al., 2016).

### Justificación teórica de cada componente (relacionada con los apuntes)
- **Convolución (Conv2D 3×3, padding 'same'):** filtros pequeños que se deslizan por la imagen. **Comparten pesos**
  (el mismo detector de "dedo extendido" sirve en cualquier posición) y tienen **conectividad local**, con muchísimos
  menos parámetros que una densa. Son equivariantes a traslaciones. 'same' mantiene el tamaño espacial, así que solo
  el pooling lo reduce, y no se pierden los bordes (donde a menudo está la mano). Los filtros se duplican en cada
  bloque (32→256) mientras la resolución se divide por 2: se cambia resolución espacial por riqueza de rasgos.
- **ReLU:** no linealidad max(0, x). Sin ella, varias capas equivaldrían a una sola transformación lineal. No se satura
  para x > 0 (evita el desvanecimiento del gradiente de sigmoide/tanh) y es barata de calcular.
- **MaxPooling2D 2×2:** reduce a la mitad alto y ancho (×4 menos cálculo en el bloque siguiente), duplica el campo
  receptivo de las capas siguientes y aporta algo de invariancia a pequeños desplazamientos.
- **BatchNormalization:** normaliza cada canal con la media y la varianza del batch y después reescala con γ y β, que
  se aprenden. Estabiliza y acelera el entrenamiento (Ioffe y Szegedy, 2015). Va entre la Conv (sin bias) y la ReLU.
  En inferencia usa medias móviles: se fija momentum 0,9 (ver comprobaciones).
- **Dropout:** en entrenamiento apaga neuronas al azar, lo que obliga a no depender de rasgos concretos y equivale a
  promediar muchas subredes (Srivastava et al., 2014). Va tras el MaxPooling (como recomiendan los apuntes): 0,2 en los
  bloques convolucionales y 0,5 en la cabeza densa, que es donde más se sobreajusta.
- **Flatten frente a GlobalAveragePooling:** Flatten conserva *dónde* está cada rasgo y lo pasa a la densa (muchos
  parámetros, atados a posiciones). GAP hace la media de cada mapa: un valor por filtro, "cuánto aparece este rasgo en
  la imagen", sin importar dónde (Lin et al., 2014, *Network in Network*). Aquí la posición de la mano no aporta nada a
  la clase, así que GAP encaja mejor y reduce la densa de 4,7 M a 33 k pesos.
- **Softmax + entropía cruzada categórica:** softmax convierte las 5 salidas en probabilidades que suman 1. La entropía
  cruzada con etiquetas one-hot es −log(p de la clase correcta). **Adam** (Kingma y Ba, 2015): descenso por gradiente
  con momento y tasa de aprendizaje adaptativa por parámetro, lr = 1e-3.
- **L2 (M3):** suma λ·Σw² a la pérdida, empujando los pesos hacia valores pequeños (modelo más "suave").

### Diagrama
`results/figuras/arquitectura_M2.png`: diagrama de bloques de M2 con la forma de cada mapa (dibujado con matplotlib;
`plot_model` necesita graphviz, que no está instalado). **Para la memoria.**

### Comprobaciones (depuración, no ajuste)
Todas en el **fold de Isaac** (train = David + Rodrigo, 644 fotos; val = 132 fotos por ráfagas). **No se mira el test**:
se reserva para la Fase 5.

**1. Test de sobreajuste** (`results/fase3_sobreajuste.csv`): 20 fotos (4 por letra), sin augmentation, batch = 20,
200 épocas. Accuracy medida en modo entrenamiento (con Dropout) y en inferencia.

| Modelo | 1ª época con 100 % (modo entrenamiento) | Acc. final (entrenamiento) | Acc. en inferencia |
|---|---|---|---|
| M0 | 21 | 1,00 | 1,00 |
| M1 | no llega | 0,70 | 0,85 |
| M2 | 82 | 0,95 | 0,40 |
| M3 | 68 | 1,00 | 0,85 |

Todos los modelos son capaces de ajustar los datos (no hay errores de etiquetas ni de compilación). La diferencia entre
entrenamiento e inferencia en los modelos con BN se debe a las estadísticas móviles de BN con tan pocos datos (un solo
batch de 20). En la ablación, el mismo M2 dio 0,75 en inferencia con 150 épocas: con 20 fotos, la cifra en inferencia
es **inestable** y no debe interpretarse como una medida de calidad.

**2. Ablación** (`results/fase3_ablacion.csv`): M2 sin augmentation, 12 épocas, quitando una pieza cada vez.

| Variante | Sobreajuste 20 fotos (inferencia) | Train acc (época 12) | Val acc (época 12) | val_loss mín–máx |
|---|---|---|---|---|
| M2, BN momentum 0,99 (defecto de Keras) | 0,20 | 0,312 | 0,235 | 1,62–4,98 |
| M2, BN momentum 0,9 | 0,75 | 0,312 | 0,242 | 1,48–1,91 |
| M2, Dropout solo en la cabeza | 0,60 | 0,371 | 0,311 | 1,64–3,09 |
| M2 sin Dropout | 1,00 | 0,494 | 0,205 | 1,72–4,58 |
| M2 sin BatchNorm ni Dropout | 0,25 | 0,214 | 0,250 | 1,60–1,61 |
| M2 con GlobalMaxPooling | 0,60 | 0,267 | 0,280 | 1,46–1,87 |
| M1 (Flatten) | 0,75 | 0,220 | 0,303 | 1,56–1,65 |

Conclusiones: (a) **momentum 0,9** arregla la inestabilidad de BN en inferencia; (b) **sin BN la red no aprende**
(predice una clase); (c) el **Dropout frena** el ajuste; (d) GMP no aporta frente a GAP; (e) M1 (Flatten) no aprende.
La val_accuracy de 12 épocas no es significativa (la validación tiene ~21 ráfagas y la accuracy salta mucho).

**3. Entrenamiento corto** (`results/fase3_entrenamiento_corto.csv`, `results/figuras/fase3_curvas_corto.png`): cada
modelo con augmentation, hasta 30 épocas, EarlyStopping(val_loss, paciencia 8, restaura los mejores pesos).

| Modelo | Épocas | Mejor época | val_loss | val_acc | Tiempo/época |
|---|---|---|---|---|---|
| M0 | 9 | 1 | 1,582 | 0,258 | 3,4 s |
| M1 | 11 | 3 | 1,596 | 0,303 | 6,1 s |
| M2 | 9 | 1 | 1,606 | 0,152 | 6,2 s |
| M3 | 14 | 6 | 1,706 | 0,159 | 6,1 s |

Con augmentation, ningún modelo despega en las primeras ~10 épocas. La val_loss es tan ruidosa que EarlyStopping
para demasiado pronto (mejor época = 1 en M0 y M2). Lección para la Fase 4: más épocas y más paciencia.

**4. Comprobación larga** (`results/fase3_entrenamiento_largo.csv`, `results/figuras/fase3_curvas_largo.png`): M2 y M2
con Dropout solo en la cabeza, 60 épocas, con augmentation, sin EarlyStopping (~6,1 s/época, ~6 min por modelo).

| Variante | Train acc (época 60) | Val acc máx. | Val acc media (últimas 10) |
|---|---|---|---|
| M2 | 0,526 | 0,424 | 0,295 |
| M2, Dropout solo en la cabeza | 0,408 | 0,432 | 0,215 |

- M2 **sí aprende**: la accuracy de entrenamiento sube de forma sostenida (0,18 → 0,53) sin estancarse.
- Pero **lentamente y generalizando poco**: tras 60 épocas, la validación (fotos de **las mismas personas**, en otras
  ráfagas) apenas supera el azar (20 %) y la val_loss **sube** desde la época ~17. La red empieza a aprender rasgos del
  fondo y de la ráfaga, no del gesto.

### Problemas encontrados y solución
- **BatchNorm inestable en inferencia** con el momentum por defecto (0,99) → momentum 0,9.
- **`model.summary()` de Keras 3 cortaba la columna de parámetros** al guardarlo en texto → tabla propia en texto plano.
- **15 épocas no bastaban** para ver nada → entrenamiento corto de 30 épocas, ablación y comprobación larga de 60.
- **Aprendizaje lento y poca generalización (sin resolver, para la Fase 4):** la mano ocupa una parte pequeña de una
  imagen con fondos muy variados, y hay ~650 fotos para aprender desde cero. Ver ideas.

### Ideas para la memoria
- La tabla de parámetros M0 → M1 → M2 es un argumento muy visual a favor de GAP (18,9 M → 5,9 M → 1,2 M).
- El cálculo del campo receptivo (8 → 38 → 68 px frente a una mano de 30-65 px) justifica la profundidad y las dos
  convoluciones por bloque con números.
- La ablación justifica BN (sin ella no aprende) y explica un detalle práctico poco conocido: el momentum de BN.
- El aprendizaje lento es un resultado en sí: aprender un gesto pequeño desde cero con ~650 fotos es difícil. Motiva
  la comparación con transfer learning y la mejora futura del recorte de la mano.

---

## Diagnóstico del bajo rendimiento inicial

**Contexto:** al terminar la Fase 3, M2 apenas superaba el azar en validación (fold de Isaac). Antes de ajustar
hiperparámetros (Fase 4) se investigó por qué, sin cambiar las decisiones tomadas. Código: `src/diagnostico.py`
(`--fallos`, `--gradcam`, `--recortes`) y `src/recortar.py`. Todo en el fold de Isaac (train = David + Rodrigo).
**No se miró el test.**

### 1. Descartar fallos de implementación
| Comprobación | Resultado | Archivo |
|---|---|---|
| ¿Memoriza o no aprende? (resumen de la Fase 3) | **No aprende ni siquiera el entrenamiento**: train acc final 0,23-0,35 en los entrenamientos cortos y 0,53 en el largo de 60 épocas, frente a val 0,06-0,36. No es sobreajuste clásico: el modelo no consigue extraer el gesto. En el test de sobreajuste (20 fotos) todos llegan a ~100 % en modo entrenamiento, así que la red *puede* ajustar datos. | `results/diagnostico_resumen_fase3.csv` |
| Etiquetas alineadas tras barajar y tf.data | **OK.** Se pasó el índice de cada foto por el pipeline como `sample_weight` y se comprobó una a una (644 de train y 132 de val) que la etiqueta coincide con la carpeta. | `results/figuras/diagnostico_batch_etiquetado.png` |
| Mismo preprocesamiento en train/val/test | **OK.** Mismo array y misma función; todos float32 en [0, 1] y 144×256×3. Medias 0,495 / 0,501 / 0,528 (el test, de Isaac, tiene menos contraste: desviación 0,145 frente a 0,234, que viene de las fotos, no del código). | `results/diagnostico_preprocesamiento.csv` |
| Distribución de predicciones en val (M2, 60 épocas) | Sesgada: predice E en 62 de 132 fotos y U en 34. **Ninguna I, O ni U acertada.** Acc val 0,144 (David 0,214; Rodrigo 0,021). | `results/diagnostico_confusion_val_M2_original.csv` |
| M1 sin augmentation, lr 1e-3 y 3e-4 (40 épocas) | **No aprende con ninguno**: train acc 0,214 y loss 1,607 ≈ ln 5 = 1,609, que es la pérdida de predecir siempre 1/5 para cada clase. Ni el augmentation ni el lr son la causa. | `results/diagnostico_entrenamientos.csv`, `results/figuras/diagnostico_curvas_M1_sinaumento.png` |

**Variabilidad entre repeticiones:** M2 con la misma configuración (60 épocas, augmentation) dio en la Fase 3
train 0,526 y val media de las 10 últimas épocas 0,295, y en el diagnóstico train 0,298 y val 0,134. Con tan pocos
datos útiles el resultado depende mucho de la ejecución. Esto es un argumento más de que el modelo no estaba aprendiendo
algo estable.

**Hipótesis sobre M1 (pendiente de confirmar):** muerte de las ReLU de la capa densa. M1 empieza con una pérdida de
entrenamiento de ~13 en la 1ª época (Flatten con 36 864 entradas → Dense 128), y un gradiente tan grande puede dejar
todas las neuronas de la densa con salida 0 para siempre. Eso explicaría una pérdida clavada en ln 5 (la salida solo
depende del bias). Antes de entrenar, el 74 % de las neuronas de la densa se activan con alguna foto; la medida tras
1-3 épocas no se pudo completar (falta de memoria en la GPU, ver problemas), así que **no hay archivo de resultados**:
el 74 % inicial se vio en la consola y no está en `results/`; no citarlo en la memoria hasta repetir la medida.

### 2. ¿Dónde mira la red? (Grad-CAM)
- **Fotos completas** (`results/figuras/gradcam_M2_original.png`, 10 fotos de validación de David, 2 por letra):
  **la red mira el fondo**: la pancarta, la estantería, la cama y las esquinas de la pared, casi nunca la mano. Las
  probabilidades de la clase predicha son 22-33 %, poco más que repartir el 20 % entre las 5 clases. **Figura para la memoria.**
- **Recortes** (`results/figuras/gradcam_M2_recorte.png`): **la red mira la mano** (nudillos, puntas de los dedos). Los
  errores son entre gestos de puño (E→A, I→A), que se parecen de verdad. **Figura para la memoria.**
- Método: Grad-CAM (Selvaraju et al., 2017) sobre la última capa convolucional (`bloque4_relu2`) para la clase predicha.

**Conclusión:** el problema es que **la mano es pequeña**. MediaPipe midió el ancho de la mano en las 958 fotos
detectadas: mediana de **30,4 px (David), 32,6 px (Isaac) y 24,1 px (Rodrigo)** sobre una imagen de 256 px de ancho,
con mínimos de 8 px (`results/recortes_deteccion.csv`). Con ~650 fotos de entrenamiento y fondos muy variados, los
rasgos del fondo y de la ráfaga son más fáciles de aprender que el gesto, y la red se queda con ellos.

### 3. Experimento: recorte automático de la mano con MediaPipe
- **Método** (`src/recortar.py`): MediaPipe Hands (`HandLandmarker`, API Tasks, mediapipe 1.2.3; modelo
  `models/hand_landmarker.task`) sobre cada foto a **resolución completa**. La caja de los 21 puntos de la mano se
  convierte en un cuadrado de lado = máx(ancho, alto) × 1,5 (margen del 25 % por cada lado), centrado en la mano. Lo que
  se sale de la foto se rellena de negro y el recorte se reduce a **128×128** (LANCZOS). Si no se detecta la mano con
  confianza 0,5, se reintenta con 0,2. Recortes en `cache/recortes_128x128.npz`; **los originales no se tocan**.
- mediapipe se instaló en el entorno `lse` sin romper TensorFlow: no cambió numpy ni protobuf; solo añadió
  `opencv-contrib-python` (misma versión que el OpenCV que ya había) y `sounddevice`.
- **Detección: 958 de 1001 fotos (95,7 %)**; 46 de ellas en el reintento con confianza 0,2. Se revisaron a ojo los 46
  recortes del reintento y **todos son manos** (sin falsos positivos): `results/figuras/revision/recortes_2a_pasada.jpg`.
- **Fallos: 43 fotos (4,3 %)** (`results/recortes_fallos_por_persona_letra.csv`):

| Persona | A | E | I | O | U | Total |
|---|---|---|---|---|---|---|
| David | 4 | 16 | 4 | 4 | 1 | 29 |
| Isaac | 0 | 1 | 0 | 1 | 0 | 2 |
| Rodrigo | 2 | 3 | 3 | 0 | 4 | 12 |
| **Total** | 6 | 20 | 7 | 5 | 5 | **43** |

  La mano se ve en casi todos los fallos. Son sobre todo **puños (E) de David sobre la pared lisa con luz de bombilla**
  (poco contraste) y **manos de Rodrigo sobreexpuestas** (blancas por exceso de luz). Hoja con todos los fallos:
  `results/figuras/revision/recortes_fallos.jpg`. Ejemplos de recortes buenos y de fallos:
  `results/figuras/recortes_ejemplos.png` (los buenos solo de David e Isaac, sin cara). **Figura para la memoria.**
- **Comparación con y sin recorte** (fold de Isaac, 60 épocas, augmentation, sin EarlyStopping; mismo split, quitando
  del recorte las fotos sin detección: train 613 de 644, val 122 de 132) (`results/diagnostico_recortes.csv`,
  `results/figuras/diagnostico_curvas_recortes.png`):

| Modelo | Entrada | Train acc final | Val acc final | Val acc media (últ. 10) | Val acc máx. | Val loss mín. |
|---|---|---|---|---|---|---|
| M2 | foto completa 144×256 (diagnóstico) | 0,298 | 0,144 | 0,134 | 0,311 | 1,553 |
| M2 | foto completa 144×256 (Fase 3, misma config.) | 0,526 | 0,356 | 0,295 | 0,424 | 1,592 |
| **M2** | **recorte 128×128** | **0,869** | **0,508** | **0,589** | **0,697** | **0,805** |
| M1 | recorte 128×128 | 0,220 | 0,238 | 0,238 | 0,246 | 1,603 |
| M1 | foto completa 144×256 | no se repitió con 60 épocas* | | | | |

  \* Para M1 con fotos completas se citan los resultados ya existentes, todos clavados en el azar:
  entrenamiento corto de la Fase 3 (con augmentation, 11 épocas con EarlyStopping): train 0,230, val 0,250
  (`results/fase3_entrenamiento_corto.csv`); y diagnóstico sin augmentation, 40 épocas: train 0,214, loss 1,607,
  val 0,250, tanto con lr 1e-3 como con 3e-4 (`results/diagnostico_entrenamientos.csv`). Se paró antes de empezar
  para no gastar ~45 min de CPU en un resultado ya conocido.
- **Lectura:** con el recorte, M2 pasa de no aprender a aprender: train 0,87 y val ~0,59 de media (máx. 0,70), con
  la val_loss mínima a la mitad (0,81 frente a 1,55-1,59). El modelo y el pipeline estaban bien; lo que fallaba era la
  entrada. M1 no aprende ni con recortes: su problema es de la arquitectura (Flatten + densa grande), no de la entrada.
- En la última época, M2 con recorte predice en validación A 50, E 3, I 5, O 21, U 43: confunde E e I con A (los tres
  gestos de puño). Es el siguiente punto a mejorar.
- Los modelos con recorte se entrenaron en la **CPU** (~18 min cada uno; en GPU serían ~3 min) por el problema de memoria
  de GPU descrito abajo. M2 con foto completa se entrenó en GPU. Pequeñas diferencias numéricas CPU/GPU no cambian la conclusión.

### Problemas encontrados y solución
- **El `Sequential` cargado de disco no tiene `model.output` en Keras 3** → Grad-CAM recorriendo las capas una a una.
- **Falta de memoria en la GPU:** otro programa del usuario (`scripts/play.py`, ajeno al proyecto) se lanza cada ~35 s
  y ocupa hasta ~4,9 GB de los 8 GB de la GPU. Los entrenamientos que coincidían fallaban por falta de memoria. Se
  entrenó en la CPU (`CUDA_VISIBLE_DEVICES=""`) con el acuerdo del usuario.
- **MediaPipe 1.2.3 ya no trae `mp.solutions`** → API Tasks (`HandLandmarker`) con el modelo descargado de
  storage.googleapis.com/mediapipe-models.

### Ideas para la memoria
- Esta sección cuenta la historia **problema → diagnóstico → solución**: comprobaciones que descartan fallos de
  código, Grad-CAM que muestra que la red mira el fondo, medida del tamaño de la mano y recorte que lo arregla (val
  0,13-0,30 → 0,59). Las dos figuras de Grad-CAM, una junto a la otra, son la prueba más visual.
- Relacionar con la teoría: aprendizaje por atajos (*shortcut learning*, Geirhos et al., 2020). La red aprende la
  señal más fácil que separa las clases en entrenamiento (fondo/ráfaga), aunque no sea la que interesa.
- El recorte es un paso de preprocesamiento (localización de la mano); la clasificación sigue siendo la CNN diseñada
  por el grupo.
