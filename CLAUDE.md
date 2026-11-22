# Proyecto P3 – Reconocimiento de vocales en Lengua de Signos Española (CNN)

Proyecto de la asignatura **Aprendizaje Automático** (profesor: Joaquín Taverner).
Grupo de 4 personas: David, Isaac, Rodrigo y un cuarto compañero (sus fotos aún no han llegado).
Comunícate siempre en **español**: respuestas, comentarios del código, nombres de gráficas y textos de resultados.

## Enunciado (resumen)

Diseñar, implementar y entrenar una **CNN** que reconozca las cinco vocales **A, E, I, O, U** hechas con la mano
en Lengua de Signos Española. El dataset lo **captura el propio grupo** (prohibido descargar uno).
Fases: captura y etiquetado → preprocesamiento → arquitectura CNN → entrenamiento y ajuste → evaluación y análisis → memoria.

### Reparto de la nota (prioriza en consecuencia)
| Parte | Peso |
|---|---|
| Captura y etiquetado de datos | 10 % |
| Preprocesamiento y preparación del dataset | 5 % |
| **Implementación de la arquitectura CNN** | **30 %** |
| **Entrenamiento y ajuste del modelo** | **20 %** |
| **Evaluación y análisis de resultados** | **25 %** |
| Referencias bibliográficas y fundamentación teórica | 10 % |

Se valora sobre todo **justificar cada decisión**. Toda decisión relevante se apunta en `docs/decisiones.md`
(qué se decidió, alternativas, por qué), porque luego se usa para redactar la memoria.

## Lo que se ha visto en clase (usar este estilo)

Los apuntes usan **TensorFlow/Keras** (`tensorflow.keras`), con la **API `Sequential`** y `model.add(...)`:
- Capas: `Conv2D` (filters, kernel_size, activation='relu', padding='same', input_shape), `MaxPooling2D(pool_size=(2,2))`,
  `BatchNormalization`, `Dropout` (recomendado tras el MaxPooling), `Flatten`, `Dense`, salida `Dense(n, activation='softmax')`.
- `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`, `to_categorical`.
- Callbacks: `EarlyStopping`, `ModelCheckpoint`. Regularización: `regularizers` (L2), Dropout, data augmentation.
- Datos (tema U1.3): train/validación/test (60-80 / 10-20 / 10-20 %), división estratificada, **conjuntos sin solapamiento
  para evitar fugas de información**, validación cruzada k-fold, desbalanceo de clases.
- Métricas: accuracy, precision, recall, F1, matriz de confusión.
- Librerías vistas: numpy, pandas, matplotlib, scikit-learn.

El código debe ser **entendible para un estudiante** y parecerse a lo visto en clase: `Sequential`, nada de trucos exóticos.
Se puede añadir algo más avanzado (p. ej. transfer learning con MobileNetV2) **solo como comparación**, nunca como modelo principal:
el enunciado pide diseñar la CNN.

## Dataset

### Ubicación y estructura (fotos originales = SOLO LECTURA)
```
~/AprendizajeAutomatico/
├── David/   A/ E/ I/ O/ U/
├── Isaac/   A/ E/ I/ o/ U/
├── Rodrigo/ A/ E/ I/ O/ U/
└── (cuarto compañero, pendiente)
```
- **NUNCA modificar, mover, renombrar ni borrar las fotos originales.** Cualquier transformación se hace en copias o en memoria.
- Las carpetas de letra pueden venir en minúscula (`a/`): tratarlas sin distinguir mayúsculas.
- Los nombres de archivo son los de la cámara (orden natural = orden en que se hicieron).
- Cada persona hizo las fotos en 4 bloques (luz natural/bombilla × fondo liso/fondo con cosas), variando ángulo, distancia,
  posición y usando a veces la otra mano. Webcam del portátil (cámara de Windows: el nombre `WIN_AAAAMMDD_HH_MM_SS_Pro.jpg`
  da la hora de captura). Se buscaba que solo saliera la mano, pero **a Rodrigo se le ve la cara en muchas fotos y a Isaac
  en algunas el borde**: cuidado al elegir figuras para la memoria.
- Las fotos de Juan llegarán más adelante (carpeta Juan/). El código no debe dar por hecho qué personas hay: las detecta
  leyendo las carpetas. Cuando lleguen, se repite la auditoría (Fase 1) y la evaluación final (Fase 5) se hace con las 4 personas.

### Recuento actual (3 personas, 1001 fotos válidas, sin corruptas ni duplicadas)
| Persona | A | E | I | O | U | Total | Resolución |
|---|---|---|---|---|---|---|---|
| David | 86 | 103 | 111 | 91 | 104 | 495 | 1280×720 |
| Isaac | 47 | 36 | 42 | 48 | 52 | 225 | 1280×720 |
| Rodrigo | 49 | 55 | 60 | 58 | 59 | 281 | 2560×1440 |
| **Total** | 182 | 194 | 213 | 197 | 215 | 1001 | |

Observaciones:
- **Clases bastante equilibradas** (182–215): no hace falta remuestrear; como mucho `class_weight`.
- **Personas desequilibradas**: David aporta casi la mitad. Tenerlo en cuenta en la evaluación por persona y comentarlo en la memoria.
- Dos resoluciones distintas pero misma proporción 16:9: se unifican al redimensionar.

## Decisiones de diseño ya tomadas

1. **Evaluación por persona (sin fuga de información):** el test nunca contiene fotos de una persona vista en entrenamiento.
   Esquema principal: **leave-one-person-out** (un fold por persona; se entrena con las demás, se testea con la que queda fuera)
   y se reportan las métricas por fold y la media ± desviación **sin ponderar** (cada persona cuenta igual).
   La validación (para EarlyStopping/ajuste) sale de las personas de entrenamiento, **por grupos de ráfaga** (fotos a ≤ 5 s;
   `StratifiedGroupKFold`), nunca foto a foto.
   Como comparación, se puede mostrar que una división aleatoria da una accuracy artificialmente alta (fuga de información).
2. **Tamaño de entrada:** **144×256** (alto × ancho) RGB, respeta el 16:9 sin padding ni recorte (la mano es pequeña y
   puede estar en un lateral). 128×128 con `resize_with_pad` como experimento de comparación. Sin recorte automático de
   la mano (MediaPipe = mejora futura).
3. **Normalización:** píxeles a [0, 1].
4. **Data augmentation solo en entrenamiento:** rotación pequeña, zoom, desplazamiento, brillo/contraste.
   **Volteo horizontal permitido**: equivale a hacer el gesto con la otra mano, y en el dataset ya hay fotos con ambas manos.
5. **Todas las fotos se usan** (no se recorta a David); `sample_weight` por persona solo como experimento de la Fase 4.
6. **Figuras de la memoria:** solo fotos sin cara.
7. **Reproducibilidad:** fijar semillas (python, numpy, tensorflow) y guardar la configuración de cada experimento.

## Estructura del proyecto (a crear)
```
~/AprendizajeAutomatico/
├── CLAUDE.md
├── David/ Isaac/ Rodrigo/ ...   # fotos originales (solo lectura)
├── src/
│   ├── config.py           # rutas, tamaño de imagen, semillas, hiperparámetros
│   ├── datos.py            # carga del dataset (con persona y letra), splits por persona, augmentation
│   ├── modelos.py          # arquitecturas CNN (Sequential)
│   ├── entrenar.py         # entrenamiento de un experimento
│   ├── evaluar.py          # métricas, matriz de confusión, errores
│   └── demo_webcam.py      # (opcional) demo en tiempo real
├── notebooks/              # notebook final de entrega (si se pide)
├── models/                 # pesos .keras (no subir a git si pesan mucho)
├── results/
│   ├── experimentos.csv    # una fila por experimento: config + métricas
│   └── figuras/            # curvas, matrices de confusión, ejemplos mal clasificados
└── docs/
    ├── decisiones.md       # registro de decisiones justificadas
    └── memoria/            # borrador de la memoria
```

## Entorno
- Ubuntu 22.04.5, portátil Lenovo LOQ 15ARP9: AMD Ryzen 7 7435HS (8n/16h), 24 GB RAM,
  GPU NVIDIA RTX 4060 Laptop 8 GB (driver 580, CUDA 13.0). Detalle completo en `docs/notas_informe.md`.
- TF 2.21 no encontraba `libcusolver.so.11` (fallo de su RUNPATH). Ya está arreglado en `lse` con un enlace simbólico en
  `nvidia/cusparse/lib/` y un script en `etc/conda/activate.d/`. **Si se reinstala TensorFlow, hay que repetirlo.**
- Hay Anaconda instalado (`(base)`). **No instalar nada en `base`**: usar un entorno conda propio llamado `lse`
  con Python 3.11 y `tensorflow[and-cuda]` (si hay GPU NVIDIA), opencv-python, scikit-learn, matplotlib, pandas, seaborn, pillow.
- Ejecutar siempre con el entorno activado: `conda activate lse`.

## Normas de trabajo
- Trabajar por **fases** (ver abajo). Al terminar una fase: actualizar la sección *Estado*, y anotar las decisiones en `docs/decisiones.md`.
- Cada experimento de entrenamiento añade una fila a `results/experimentos.csv` y guarda sus gráficas en `results/figuras/`
  con un nombre que identifique el experimento.
- Gráficas con títulos y ejes en español, listas para pegar en la memoria.
- Antes de entrenar mucho rato, hacer una prueba corta (pocas épocas) para comprobar que todo funciona.
- No inventar resultados: cualquier número de la memoria debe salir de `results/`.
- Al terminar cada fase, apuntar en `docs/notas_informe.md` todo lo que necesitaremos para redactar el informe: qué se hizo,
  por qué, resultados y números concretos, figuras generadas (con su ruta), problemas encontrados y cómo se resolvieron,
  e ideas para la memoria. Escribirlo de forma clara para que luego se pueda pasar casi directamente al informe.

## Fases y estado
- [x] **Fase 0** – Entorno (conda `lse`, TensorFlow, comprobar GPU) y estructura de carpetas.
- [x] **Fase 1** – Auditoría y carga del dataset (recuento, cuadrícula de ejemplos por persona/letra, revisar fotos casi idénticas).
- [x] **Fase 2** – Preprocesamiento, data augmentation y splits por persona.
- [x] **Fase 3** – Arquitectura CNN: modelo base y variantes (BatchNorm, Dropout, más bloques).
- [~] **Diagnóstico** (entre Fase 3 y 4) – casi terminado, ver Estado.
- [ ] **Fase 4** – Entrenamiento y ajuste de hiperparámetros (tabla de experimentos).
- [ ] **Fase 5** – Evaluación final leave-one-person-out y análisis de errores.
- [ ] **Fase 6** – (Opcional) Demo en tiempo real con la webcam.
- [ ] **Fase 7** – Memoria: fundamentación teórica, referencias, resultados y conclusiones.

## Estado
(Actualizar al terminar cada fase: qué se hizo, archivos creados, resultados clave y pendientes.)

### Fase 0 – completada (2026-11-22)
- Entorno conda `lse` (Python 3.11.18): TensorFlow 2.21.0 + Keras 3.15.1 con GPU (CUDA 12.9, cuDNN 9.28), numpy 2.4.6,
  scikit-learn 1.9.1, OpenCV 5.1.0, pandas, matplotlib, seaborn, pillow, imagehash, jupyter. GPU comprobada
  (matmul ~10× más rápido que en CPU; Conv2D en GPU). Hubo que arreglar la carga de cuSOLVER (ver *Entorno*).
- Creados: `src/config.py` (rutas, `CLASES`, `TAM_IMAGEN=128`, `SEMILLA=42`, `detectar_personas()`, `fijar_semillas()`),
  `requirements.txt`, `.gitignore`, `docs/decisiones.md`, `docs/notas_informe.md` (con *Entorno experimental*),
  carpetas `src/ notebooks/ models/ results/figuras/ docs/memoria/`.
- `detectar_personas()` encuentra ahora: David, Isaac, Rodrigo.
- (Resuelto en la Fase 1: la copia antigua `dataset_limpio/` se borró; se lee de los originales.)

### Fase 1 – completada (2026-11-22)
- Limpieza: borrados `dataset_limpio/` (copia antigua) y `organizar_dataset.py`; `PROMPTS_FASES.md` → `docs/`.
  Originales comprobados antes: 1001 fotos, recuentos idénticos.
- `src/datos.py`: `cargar_listado()` → DataFrame (ruta, persona, letra, archivo, orden, ancho, alto); personas detectadas
  por carpetas, letra sin distinguir mayúsculas, orden natural del nombre = orden de captura.
- `src/auditoria.py` (`cd src && python auditoria.py`): recuento, resoluciones, ráfagas, ejemplos, hojas de contacto y
  casi idénticas. Salidas en `results/` (`recuento_persona_letra.csv`, `resoluciones.csv`, `casi_identicas.csv`,
  `auditoria_phash_umbral.csv`, `auditoria_resumen.json`) y `results/figuras/auditoria_*.png`, `results/figuras/revision/`.
- Resultados: 0 corruptas; 1280×720 (David, Isaac) y 2560×1440 (Rodrigo), todas 16:9. El pHash global mide el fondo, no la
  mano (tantas parejas parecidas entre letras distintas como dentro de la misma letra). Solo **4 parejas casi idénticas**
  confirmadas (0,4 %). Ráfagas: ~1 foto/s (51 % de las fotos consecutivas de David están a ≤ 1 s).
- **Pendiente de decisión del grupo** (propuestas en `docs/decisiones.md`, P1.1–P1.3): desequilibrio de personas,
  qué hacer con casi idénticas/ráfagas y tamaño de entrada. Revisar a ojo las hojas de `results/figuras/revision/`.

### Fase 2 – completada (2026-11-22)
- `src/config.py`: `ALTO_IMAGEN=144`, `ANCHO_IMAGEN=256`, `DIR_CACHE`, `UMBRAL_RAFAGA_SEGUNDOS=5`, `PROPORCION_VALIDACION=1/6`,
  `AUMENTO` (rangos). `.gitignore` excluye `cache/`.
- `src/datos.py`: `cargar_listado()` (+ hora y `rafaga`), `cargar_imagenes(df, alto, ancho)` → X float32 [0,1], Y one-hot
  float32 (A,E,I,O,U), con caché `.npz`; `splits_lopo(df)` / `split_aleatorio(df)` (dicts de índices train/val/test);
  `comprobar_split()` (asserts); `capas_aumento()`; `crear_dataset(X, Y, idx, entrenamiento, aumento=, pesos=)`.
- `src/preparar_datos.py` (`cd src && python preparar_datos.py`): ráfagas, cachés, splits + asserts, figuras y prueba tf.data.
  Salidas: `results/rafagas_umbral.csv`, `rafagas_por_persona_letra.csv`, `splits_resumen.csv`, `splits.json`,
  `results/figuras/rafagas_histograma.png`, `preprocesado_tamanos.png`, `aumento_ejemplos.png`.
- Resultados: 139 ráfagas (87,4 % de los tiempos ≤ 5 s). Folds LOPO sin solapamiento de fotos, ráfagas ni personas
  (val ≈ 17 % de las personas de entrenamiento). Split aleatorio: 100 de las 139 ráfagas repartidas entre conjuntos (fuga).
- Aviso inofensivo de TF al usar `map` en tf.data: "NodeDef mentions attribute use_unbounded_threadpool" (se puede ignorar).

### Fase 3 – completada (2026-11-22)
- `src/modelos.py`: `crear_m0()` (baseline de clase, Flatten, 18,9 M parámetros), `crear_m1()` (4 bloques
  [Conv-BN-ReLU]×2 + MaxPool + Dropout 0,2, Flatten, 5,9 M), `crear_m2()` (igual con GAP, 1,21 M, **principal propuesto**),
  `crear_m3()` (M2 + L2 1e-4); todos sobre `crear_cnn(...)` con parámetros (filtros, convs_por_bloque, dropout_conv,
  dropout_denso, unidades_densas, l2, cabeza flatten/gap/gmp, batchnorm, bn_momentum); `compilar(model, lr)`;
  `campo_receptivo(model)`.
- **BN con momentum 0,9** (el 0,99 de Keras hacía inestable la inferencia: comprobado en la ablación).
- `src/fase3_arquitectura.py` [`--solo-analisis` | `--ablacion` | `--largo`]: summaries (`results/modelos/`),
  `modelos_comparativa.csv`, `campo_receptivo.csv`, `fase3_sobreajuste.csv`, `fase3_ablacion.csv`,
  `fase3_entrenamiento_{corto,largo}.csv`, figuras `arquitectura_M2.png`, `fase3_curvas_{corto,largo}.png`.
- Resultados clave: campo receptivo de la última conv 8 px (M0), 38 px (1 conv/bloque) y 68 px (2 conv/bloque), frente a
  una mano de ~30-65 px. Sin BN la red no aprende. ~6 s/época (M1-M3) y 3,4 s (M0) en la RTX 4060.
- **Aviso para la Fase 4:** M2 aprende despacio. Con 60 épocas y augmentation: train 0,53 y val ~0,30 (máx. 0,42) en el
  fold de Isaac, con la val_loss subiendo desde la época ~17. Hacen falta más épocas, más paciencia en EarlyStopping y
  ajustar lr/dropout. La val_loss es muy ruidosa (validación ≈ 21 ráfagas).

### Diagnóstico del bajo rendimiento – EN CURSO (lo dejamos el 2026-11-22 por la tarde)
**Dónde lo dejamos:** se hicieron los pasos 1-3 del diagnóstico pedido tras la Fase 3, excepto lo indicado como pendiente.
Todo documentado en `docs/notas_informe.md` (sección "Diagnóstico del bajo rendimiento inicial") y las propuestas en
`docs/decisiones.md` (PD1-PD5, **pendientes de OK**). No hay ningún proceso de entrenamiento en marcha.
- Código nuevo: `src/diagnostico.py` (`--fallos`, `--gradcam`, `--recortes`) y `src/recortar.py` (MediaPipe → `cache/recortes_128x128.npz`,
  `results/recortes_deteccion.csv`). Modelos guardados: `models/diag_M2_original.keras`, `models/diag_M2_recorte.keras`,
  `models/hand_landmarker.task` (modelo de MediaPipe).
- mediapipe 1.2.3 está instalado en `lse` (TF sigue funcionando con GPU). Solo tiene la API Tasks (no `mp.solutions`).
- **Conclusión:** no hay errores de código; con fotos completas la red mira el fondo (Grad-CAM) porque la mano es pequeña
  (mediana ~24-33 px sobre 256). Con recorte de la mano, M2 llega a val ~0,59 de media (máx. 0,70) frente a ~0,13-0,30
  sin recorte. M1 (Flatten) no aprende en ningún caso (loss = ln 5).
- MediaPipe detecta la mano en 958/1001 fotos (43 fallos: 29 David, 12 Rodrigo, 2 Isaac; 20 son E).
**Qué falta del diagnóstico:**
1. Medir las ReLU muertas de M1 (prueba corta de 3 épocas; falló por falta de memoria de GPU). Script improvisado, no está en `src/`.
2. M1 con foto completa y 60 épocas: **no se hizo a propósito** (los resultados previos ya lo muestran clavado en el azar).
3. Decidir PD1-PD5 con el grupo.
**Siguiente paso mañana:** (1) que el grupo dé el OK a PD1 (recorte como entrada principal) y PD2 (qué hacer con los 43
fallos); (2) si se acepta, integrar el recorte en `datos.py`/`config.py` (entrada 128×128 desde `cache/recortes_128x128.npz`),
rehacer las figuras de preprocesado/aumento con recortes y actualizar decisiones D1.6 y la Fase 3 (parámetros/campo receptivo
a 128×128); (3) empezar la Fase 4 con M2 + recortes.
**Ojo con la GPU:** `scripts/play.py` (otro programa del usuario) ocupa hasta ~4,9 GB cada ~35 s y provoca falta de memoria.
Pararlo antes de entrenar en GPU o usar `CUDA_VISIBLE_DEVICES=""` (CPU, ~18 min por modelo de 60 épocas con recortes).
