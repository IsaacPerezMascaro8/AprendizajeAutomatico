# Registro de decisiones – Proyecto P3 (LSE, vocales con CNN)

Cada decisión relevante: **qué se decidió**, **alternativas consideradas** y **por qué**. Se usa para redactar la memoria.

---

## Fase 0 – Entorno y estructura

### D0.1 – Framework: TensorFlow/Keras con API `Sequential`
- **Decisión:** TensorFlow 2 + Keras (`tensorflow.keras`), modelos con `Sequential` y `model.add(...)`.
- **Alternativas:** PyTorch; API funcional de Keras.
- **Por qué:** es lo que se usa en los apuntes de la asignatura; el código queda más fácil de seguir y de defender.

### D0.2 – Entorno conda propio (`lse`, Python 3.11) con soporte GPU
- **Decisión:** entorno aislado `lse` con `tensorflow[and-cuda]`, sin tocar `base`.
- **Alternativas:** instalar en `base`; entrenar solo en CPU; Google Colab.
- **Por qué:** el portátil tiene una GPU NVIDIA RTX 4060 (8 GB), lo que acelera mucho el entrenamiento y permite hacer
  leave-one-person-out y varios experimentos en local. Un entorno aislado evita conflictos de versiones y es reproducible
  (`requirements.txt`).

### D0.3 – Personas detectadas automáticamente
- **Decisión:** `config.detectar_personas()` lee las carpetas de la raíz que contienen subcarpetas de letra.
- **Por qué:** las fotos de Juan llegarán más tarde; así no hay que cambiar código cuando aparezca `Juan/`.

### D0.4 – Semilla fija (42) para Python, NumPy y TensorFlow
- **Decisión:** `config.fijar_semillas()` al inicio de cada experimento, con determinismo de operaciones activado.
- **Por qué:** reproducibilidad de los experimentos y comparaciones justas entre configuraciones.

---

## Fase 1 – Auditoría y carga del dataset

### D1.1 – Leer directamente de las carpetas originales (sin copia intermedia)
- **Decisión:** borrar `dataset_limpio/` y `organizar_dataset.py`. `datos.cargar_listado()` lee `<Persona>/<Letra>/`.
- **Alternativas:** mantener una copia renombrada y uniforme.
- **Por qué:** una copia puede desincronizarse de los originales (p. ej., cuando lleguen las fotos de Juan) y ocupa disco.
  Leer los originales en modo solo lectura es más simple y la carpeta de letra se reconoce sin distinguir mayúsculas.

### D1.2 – Detección de casi idénticas: pHash + confirmación por píxeles, con umbral calibrado con un control
- **Decisión:** pHash (64 bits) para buscar candidatas. Una pareja de la misma persona y letra es "casi idéntica" si cambia
  menos píxeles (contraste normalizado) que la pareja de letras distintas más parecida del dataset (2,08 %).
- **Alternativas:** solo un umbral de pHash (p. ej. ≤ 4 o ≤ 5, habitual en la literatura); hash MD5 (solo duplicados exactos).
- **Por qué:** el control con parejas de letras distintas demostró que el pHash de la imagen completa mide el fondo y no la
  mano (99 parejas de letras distintas con pHash ≤ 4, frente a 78 de la misma letra). Usar parejas que no pueden ser
  duplicadas como referencia da un umbral objetivo, sacado de los propios datos.

### D1.3 – Revisión de etiquetas con hojas de contacto completas
- **Decisión:** además de la cuadrícula aleatoria (para la memoria), generar una hoja con todas las fotos de cada persona y
  letra (`results/figuras/revision/`).
- **Por qué:** una muestra aleatoria de 8 por letra no garantiza encontrar una foto en la carpeta equivocada.

### D1.4 – Desequilibrio entre personas (P1.1) – ACEPTADA
- **Decisión:** se usan todas las fotos (no se recorta a David). Las métricas se dan por fold y como **media sin ponderar**
  (cada persona cuenta igual). En la Fase 4 se probará `sample_weight` por persona como experimento.
- **Alternativas:** recortar David a ~250 fotos; pesos por persona desde el principio.
- **Por qué:** con 1001 fotos, descartar datos duele más que el desequilibrio. En leave-one-person-out el test de cada fold
  es una persona, así que el desequilibrio solo afecta a qué "estilo" de mano domina el entrenamiento, no a la medida.
  Además, parte de la ventaja numérica de David se debe a que disparó más rápido (ráfagas), no a más variedad.

### D1.5 – Validación por grupos de ráfaga (P1.2) – ACEPTADA
- **Decisión:** no se borra ninguna foto casi idéntica (4 parejas, 0,4 %). La validación se saca por **grupos de ráfaga**
  con `StratifiedGroupKFold`, nunca al azar foto a foto.
- **Por qué:** fotos hechas con 1-2 s de diferencia son casi iguales. Si unas caen en entrenamiento y otras en validación,
  la validación mide memorización: sale optimista y EarlyStopping para tarde.

### D1.6 – Tamaño de entrada (P1.3) – ACEPTADA CON CAMBIO
- **Decisión:** tamaño principal **144×256** (alto × ancho), que respeta el 16:9 sin padding ni recorte. Como experimento
  de comparación, 128×128 con `resize_with_pad`. **No** se hace recorte automático de la mano (p. ej. MediaPipe): se cita
  como mejora futura.
- **Alternativas:** 128×128 con padding (la imagen útil queda en 128×72); 128×128 estirado (deforma la mano);
  recortar la mano con un detector.
- **Por qué:** la mano ocupa poco del encuadre. Con 144×256 tiene ~2 veces más píxeles por lado que con 128×72, y el coste
  sigue siendo asumible en la GPU. El recorte con un detector externo añadiría una dependencia y se alejaría de "diseñar la CNN".

### D1.7 – Fotos con cara
- **Decisión:** se mantienen; no afectan al modelo (la cara aparece en todas las letras). En las figuras de la memoria se
  usan fotos sin cara.

---

## Fase 2 – Preprocesamiento, data augmentation y splits

### D2.1 – Grupos de ráfaga con umbral de 5 s
- **Decisión:** dos fotos consecutivas (misma persona y letra) son de la misma ráfaga si las separan ≤ 5 s
  (`config.UMBRAL_RAFAGA_SEGUNDOS`). La hora sale del nombre del archivo (o del EXIF si no lo tiene).
- **Alternativas:** 2-3 s (grupos más pequeños, 183-277); 30-300 s (grupo = bloque entero, solo 54-70 grupos).
- **Por qué:** el histograma (`results/figuras/rafagas_histograma.png`) muestra que el 87,4 % de los tiempos está entre 0
  y 5 s (ritmo de disparo de ~1 foto/s) y que después la distribución se aplana en una cola larga (pausas para cambiar
  de postura) hasta los cambios de bloque (> 20 s). Con 5 s salen 139 grupos (mediana 5 fotos) y al menos 4 por persona
  y letra, suficientes para `StratifiedGroupKFold`. Un umbral menor partiría ráfagas reales en trozos casi iguales (fuga).
  Uno mayor dejaría tan pocos grupos que la validación no se podría equilibrar.

### D2.2 – Validación: StratifiedGroupKFold (1/6) eligiendo la partición más equilibrada
- **Decisión:** para cada fold LOPO, validación = 1 de 6 particiones de `StratifiedGroupKFold` (grupos = ráfagas,
  estratos = persona+letra) sobre las personas de entrenamiento. De 50 semillas × 6 particiones se elige la que minimiza
  la peor desviación por letra + la peor desviación por persona. Resultado: ~17 % de las fotos de entrenamiento.
- **Alternativas:** la primera partición de `StratifiedGroupKFold`; `GroupShuffleSplit`.
- **Por qué:** con pocos grupos por letra y de tamaños muy distintos (1-21 fotos), la primera partición dejaba la
  validación del fold de David con solo 4 fotos de la I frente a 27 de la O. Elegir la más equilibrada es determinista
  y mantiene la regla de no partir ráfagas.

### D2.3 – Redimensionado con `resize_with_pad` y antialias
- **Decisión:** siempre `tf.image.resize_with_pad(..., antialias=True)`. Para 144×256 (misma proporción) no añade bandas;
  para 128×128 añade bandas negras arriba y abajo. Píxeles normalizados a [0, 1] (÷255).
- **Por qué:** una sola función para los dos tamaños y nunca se deforma ni se recorta la mano. El antialias evita el
  aliasing al reducir 10 veces las fotos de Rodrigo (2560 px): sin él, sus fotos tendrían un ruido distinto que la red
  podría aprender como "estilo de persona".

### D2.4 – Caché en disco
- **Decisión:** `cache/imagenes_<alto>x<ancho>.npz` (uint8, ~110 MB para 144×256), ignorada por git. Se regenera sola
  si cambia la lista de fotos (p. ej., cuando lleguen las de Juan).
- **Por qué:** redimensionar las 1001 fotos tarda ~18 s; leer la caché, ~0,2 s.

### D2.5 – Data augmentation (solo entrenamiento, capas de Keras en el pipeline tf.data)
| Capa | Rango | Por qué |
|---|---|---|
| RandomFlip | horizontal | Equivale a hacer el gesto con la otra mano (ya hay fotos de ambas). Sin volteo vertical: una mano boca abajo no es un gesto real. |
| RandomRotation | ±10° | La mano nunca está perfectamente vertical. Con giros mayores, la I (meñique) o la U (dos dedos) podrían dejar de parecerse al gesto real. |
| RandomZoom | ±10 % | Distancia distinta a la cámara. Moderado porque la mano suele estar cerca del borde y un zoom fuerte la cortaría. |
| RandomTranslation | ±10 % | La mano aparece en cualquier zona del encuadre. |
| RandomBrightness | ±0,2 (rango [0,1]) | Luz natural frente a bombilla; webcams distintas. |
| RandomContrast | factor 0,8-1,2 | Fotos con bombilla muy apagadas (David). |
- **Relleno `fill_mode='constant'` (negro):** el modo por defecto de Keras (`reflect`) crea una segunda mano reflejada
  cuando la mano está pegada al borde.
- **`value_range=(0, 1)`** en brillo y contraste: por defecto Keras supone [0, 255] y los cambios saldrían desproporcionados.
- **En tf.data y no dentro del modelo:** así los modelos de la Fase 3 son solo la CNN, y el aumento se aplica a
  entrenamiento y nunca a validación ni a test (se comprueba que la validación es determinista).

### D2.6 – Split aleatorio solo como comparación
- **Decisión:** 70/15/15 estratificado por letra, foto a foto (`datos.split_aleatorio`). Se usará en la Fase 5 para
  demostrar la fuga de información.

---

## Fase 3 – Arquitectura CNN

### D3.1 – Cuatro modelos, de la base de los apuntes a la propuesta
- **M0** (baseline de clase): Conv(32)-Pool-Conv(64)-Pool-Flatten-Dense(128)-softmax. 18,9 M parámetros, el 99,9 % en
  la densa que va tras Flatten.
- **M1**: 4 bloques [Conv3×3-BN-ReLU]×2 + MaxPool + Dropout(0,2) con 32-64-128-256 filtros; Flatten; Dense(128); Dropout(0,5);
  softmax. 5,9 M parámetros (80 % en la densa).
- **M2** (propuesto como principal): igual que M1 con **GlobalAveragePooling2D** en lugar de Flatten. 1,21 M parámetros
  (2,8 % en la densa).
- **M3**: M2 + L2 (1e-4) en convoluciones y densa.
- **Por qué esta progresión:** cada modelo cambia una sola cosa respecto al anterior, así que las diferencias se pueden
  atribuir a ese cambio (profundidad+BN → cabeza → regularización).

### D3.2 – Dos convoluciones 3×3 por bloque (estilo VGG)
- **Alternativas:** una conv por bloque; kernels 5×5.
- **Por qué:** dos 3×3 ven lo mismo que una 5×5 con menos parámetros (18 frente a 25 por canal) y una ReLU más.
  Sobre todo, el **campo receptivo** de la última conv pasa de **38 px** (1 conv/bloque) a **68 px** (2 conv/bloque),
  y la mano mide unos 30-65 px de ancho en la imagen de 256: con 2 convs, cada neurona del último bloque puede ver la
  mano entera.

### D3.3 – Orden Conv → BatchNormalization → ReLU, Conv sin bias
- **Alternativa:** Conv(activation='relu') → BN (también se usa en la práctica).
- **Por qué:** es el orden del artículo original de BN (Ioffe y Szegedy, 2015), que normaliza la entrada de la no
  linealidad. El bias de la conv sobra porque BN ya suma su propio desplazamiento (β).

### D3.4 – BatchNormalization con momentum = 0,9 (y no 0,99, el valor por defecto de Keras)
- **Problema encontrado:** con 0,99, los modelos con BN memorizaban 20 fotos en modo entrenamiento pero fallaban en
  inferencia (test de sobreajuste: 20 % de accuracy en inferencia) y la val_loss saltaba hasta 4,98.
- **Causa:** BN usa en inferencia medias y varianzas **móviles**, que empiezan en 0 y 1. Con ~20 batches por época y
  momentum 0,99, el valor inicial sigue pesando durante decenas de épocas, y las varianzas reales son mucho menores que 1.
- **Solución:** momentum 0,9 (el valor inicial pesa < 1 % en ~44 pasos, unas 2 épocas). Con ese cambio, el test de
  sobreajuste en inferencia pasa a 75 % y la val_loss máxima a 1,91 (`results/fase3_ablacion.csv`).

### D3.5 – BatchNormalization es imprescindible a esta profundidad
- **Ablación:** M2 sin BN ni Dropout se queda en 21 % de accuracy de entrenamiento tras 12 épocas (predice una sola
  clase). Con BN y sin Dropout llega al 49 %.
- **Por qué:** BN mantiene las activaciones en una escala razonable en cada capa y permite que el gradiente llegue
  bien a los primeros bloques.

### D3.6 – Dropout 0,2 tras cada MaxPooling y 0,5 en la cabeza
- **Por qué:** los apuntes recomiendan Dropout tras el pooling. En capas convolucionales los píxeles vecinos están muy
  correlacionados, así que se usa una tasa baja (0,2); en la densa, la clásica 0,5 (Srivastava et al., 2014).
- **Contrapartida medida:** el Dropout frena el entrenamiento (sin Dropout: 49 % en entrenamiento a las 12 épocas;
  con él: 31 %) y puede descuadrar las estadísticas de BN entre entrenamiento e inferencia ("variance shift",
  Li et al., 2019). Pero en la comprobación larga (60 épocas con augmentation), M2 con Dropout en los bloques generaliza
  mejor que M2 con Dropout solo en la cabeza (val_accuracy media de las 10 últimas épocas: 0,295 frente a 0,215). Se
  mantiene 0,2 y se ajustará en la Fase 4 (0 / 0,1 / 0,2).

### D3.7 – GlobalAveragePooling2D en lugar de Flatten (modelo principal M2)
- **Alternativas:** Flatten (M0, M1); GlobalMaxPooling2D (probada en la ablación).
- **Por qué:**
  1. **Parámetros:** Flatten tras el bloque 4 da 9·16·256 = 36 864 entradas a la densa (4,7 M pesos); GAP da 256
     (33 k pesos). Con ~650 fotos de entrenamiento, 4,7 M pesos en la densa invitan a memorizar.
  2. **Invariancia a la posición:** con Flatten, cada peso de la densa está atado a una zona concreta de la imagen, y
     la red tendría que aprender cada gesto en cada posición. Con GAP, la red responde a "hay un rasgo de la U en
     algún sitio", que es lo que se necesita porque la mano aparece en cualquier zona del encuadre.
  3. **Medido:** M1 (Flatten) empieza con una pérdida de entrenamiento de 13 y se queda sin aprender (22-23 % de
     accuracy de entrenamiento). M0 (Flatten) sube rápido en entrenamiento pero memoriza: su validación queda por
     debajo del azar.
- **GlobalMaxPooling:** se probó por si la media "diluía" la mano (ocupa pocas de las 144 posiciones del mapa 9×16).
  En 12 épocas sin augmentation no aprende más rápido (train 0,27 frente a 0,31), así que se queda GAP, que es lo
  estándar. GMP queda disponible (`cabeza="gmp"`) por si se quiere probar en la Fase 4.

### D3.8 – Salida softmax + entropía cruzada categórica + Adam (lr 1e-3)
- **Por qué:** clasificación multiclase excluyente (una vocal por foto). Softmax da una distribución de probabilidad
  sobre las 5 clases, y la entropía cruzada es la log-verosimilitud negativa de la clase correcta (lo visto en clase).
  Adam es el optimizador de los apuntes. El learning rate se ajusta en la Fase 4.

### D3.9 – Diagrama dibujado con matplotlib
- `plot_model` necesita pydot y graphviz (no instalados). El diagrama con matplotlib
  (`results/figuras/arquitectura_M2.png`) muestra además las formas de los mapas, que es lo que interesa en la memoria.

---

## Diagnóstico del bajo rendimiento (tras la Fase 3) – PROPUESTAS PENDIENTES DE OK

Hechos (ver `docs/notas_informe.md`, sección "Diagnóstico del bajo rendimiento inicial"): no hay fallos de código
(etiquetas y preprocesamiento comprobados); con fotos completas la red mira el fondo (Grad-CAM); la mano mide
~24-33 px de mediana sobre 256; con recorte de la mano, M2 pasa de val ~0,13-0,30 a ~0,59 (máx. 0,70).

- **PD1 – Usar el recorte de la mano como entrada principal.** Cambia la decisión D1.6 (P1.3), que dejaba el recorte
  automático como mejora futura. Propuesta: entrada principal = recorte MediaPipe 128×128; foto completa 144×256 como
  comparación (muestra por qué hace falta localizar la mano). Encaja con el enunciado: el recorte es preprocesamiento
  y la clasificación la sigue haciendo la CNN propia. **Pendiente de OK del grupo.**
- **PD2 – Qué hacer con las 43 fotos sin detección (4,3 %).** Opciones: (a) excluirlas del entrenamiento y, en la
  evaluación, contarlas aparte (y también como error para dar una cifra "de extremo a extremo"); (b) recortarlas a mano
  (son 43, unos 15 min) para tener el dataset completo; (c) recorte central: **descartado**, porque la mano casi nunca
  está en el centro. Propuesta: (b) si hay tiempo y si no (a); en ambos casos, documentar los fallos por persona y letra.
- **PD3 – M1 (Flatten) no aprende ni con fotos completas ni con recortes** (pérdida clavada en ln 5). Hipótesis: ReLU
  muertas en la densa tras la enorme pérdida inicial. Propuesta: confirmarlo midiendo las neuronas activas (prueba corta)
  y mantener M1 en la memoria como ejemplo negativo justificado (por qué no Flatten + densa grande), sin seguir ajustándolo.
- **PD4 – Recalcular lo que depende del tamaño de entrada** si se acepta PD1: campo receptivo y parámetros con 128×128
  (M2 no cambia de parámetros con GAP; M1 sí) y la figura de aumento con un recorte.
- **PD5 – Comparación con transfer learning (MobileNetV2)** sobre los recortes, solo como referencia (permitido por el CLAUDE.md).
- **Aviso operativo:** `scripts/play.py` (otro programa del usuario) ocupa hasta ~4,9 GB de GPU cada ~35 s. Antes de
  entrenar en GPU, pararlo o entrenar en la CPU (~6 veces más lento: ~18 min frente a ~3 min por modelo de 60 épocas con recortes).
