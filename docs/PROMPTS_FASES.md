# Prompts para Claude Code – Proyecto LSE

Cómo usarlo: abre una terminal en `~/AprendizajeAutomatico`, ejecuta `claude` y pega el prompt de la fase.
Claude Code lee el `CLAUDE.md` automáticamente. Haz una fase cada vez y revisa el resultado antes de pasar a la siguiente.
Si abres una sesión nueva entre fases, no pasa nada: el estado queda apuntado en `CLAUDE.md`.

---

## Fase 0 – Entorno y estructura

```
Vamos con la Fase 0 del CLAUDE.md.
1. Comprueba si tengo GPU NVIDIA (nvidia-smi) y dime qué hay.
2. Crea un entorno conda llamado "lse" con Python 3.11 e instala tensorflow (con [and-cuda] si hay GPU NVIDIA),
   opencv-python, scikit-learn, matplotlib, pandas, seaborn y pillow. No toques el entorno base.
3. Comprueba que TensorFlow se importa y si detecta la GPU.
4. Crea la estructura de carpetas del CLAUDE.md (src/, results/figuras/, models/, docs/, notebooks/) con un
   src/config.py inicial y un .gitignore que excluya models/ y las carpetas de fotos.
Al final, actualiza la sección Estado del CLAUDE.md.
```

## Fase 1 – Auditoría y carga del dataset

```
Vamos con la Fase 1 del CLAUDE.md. Recuerda: las fotos originales son de solo lectura.
1. En src/datos.py, crea una función que recorra las carpetas de cada persona y devuelva un DataFrame con
   ruta, persona, letra (carpetas de letra sin distinguir mayúsculas) y resolución.
2. Haz un script o notebook de auditoría que:
   - imprima el recuento por persona y letra,
   - guarde en results/figuras/ una cuadrícula con ejemplos aleatorios de cada letra para cada persona
     (para revisar a ojo que las etiquetas son correctas),
   - detecte fotos casi idénticas dentro de cada persona (por ejemplo, con un hash perceptual) y me diga cuántas hay.
3. Comenta el desequilibrio entre personas (David tiene casi la mitad) y propón si conviene hacer algo.
Apunta las decisiones en docs/decisiones.md y actualiza el Estado del CLAUDE.md.
```

## Fase 2 – Preprocesamiento y splits

```
Vamos con la Fase 2 del CLAUDE.md.
1. En src/datos.py, implementa la carga de imágenes a 128x128 RGB sin recortar (resize_with_pad), normalizadas a [0,1],
   y la conversión de etiquetas con to_categorical (orden de clases A, E, I, O, U).
2. Implementa los splits por persona:
   - leave-one-person-out: para cada persona, test = esa persona; de las demás, sacar validación estratificada (~15-20 %),
   - y, solo como comparación, un split aleatorio estratificado para demostrar la fuga de información.
3. Implementa el data augmentation solo para entrenamiento (rotación pequeña, zoom, desplazamiento, brillo/contraste,
   volteo horizontal) con capas de Keras o equivalente.
4. Guarda en results/figuras/ un ejemplo de una imagen original y varias versiones aumentadas.
Explícame cada decisión en docs/decisiones.md y actualiza el Estado del CLAUDE.md.
```

## Fase 3 – Arquitectura CNN

```
Vamos con la Fase 3 del CLAUDE.md. Usa el estilo de los apuntes (Sequential + model.add).
En src/modelos.py, crea:
1. Un modelo base sencillo: 2-3 bloques Conv2D(relu, padding same) + MaxPooling2D, Flatten, Dense(128, relu), Dense(5, softmax).
2. Variantes para comparar: con BatchNormalization, con Dropout tras el pooling, con más bloques/filtros,
   y con GlobalAveragePooling en lugar de Flatten.
Para cada modelo, muestra el model.summary() y explica el número de parámetros y por qué tiene esa forma.
Haz una prueba rápida de pocas épocas con un fold para comprobar que entrena.
Documenta la justificación de la arquitectura en docs/decisiones.md (esto vale el 30 % de la nota) y actualiza el Estado.
```

## Fase 4 – Entrenamiento y ajuste

```
Vamos con la Fase 4 del CLAUDE.md.
1. En src/entrenar.py, crea un script que entrene un experimento a partir de una configuración
   (modelo, learning rate, batch size, dropout, tamaño de imagen, augmentation sí/no) con EarlyStopping y ModelCheckpoint,
   guarde las curvas de loss/accuracy (train vs validación) y añada una fila a results/experimentos.csv.
2. Usando un fold fijo de leave-one-person-out para ajustar, prueba de forma ordenada:
   arquitecturas de la Fase 3, learning rate, batch size, dropout, con y sin augmentation, 128 vs 64 px.
   Primero dime el plan de experimentos y cuánto tardará aproximadamente; espera a que te dé el OK antes de lanzarlo.
3. Al final, haz una tabla resumen, elige la mejor configuración y justifícala (comentando el sobreajuste en las curvas).
Actualiza docs/decisiones.md y el Estado del CLAUDE.md.
```

## Fase 5 – Evaluación final

```
Vamos con la Fase 5 del CLAUDE.md, con la mejor configuración de la Fase 4.
1. Haz el leave-one-person-out completo: un modelo por persona dejada fuera.
   Para cada fold y en media ± desviación, calcula accuracy, precision, recall y F1 por clase.
2. Guarda la matriz de confusión de cada fold y la agregada.
3. Guarda una figura con ejemplos mal clasificados (imagen, clase real, predicha y confianza) y analiza
   qué vocales se confunden y por qué.
4. Compara con el split aleatorio para mostrar la diferencia por fuga de información.
5. Comenta la influencia del desequilibrio entre personas y de la resolución distinta de Rodrigo.
6. Entrena el modelo final con todas las personas y guárdalo en models/ para la demo.
Escribe un resumen del análisis en docs/decisiones.md y actualiza el Estado del CLAUDE.md.
```

## Fase 6 – Demo webcam (opcional)

```
Vamos con la Fase 6 del CLAUDE.md. Crea src/demo_webcam.py con OpenCV: abre la webcam, dibuja un recuadro donde poner
la mano, preprocesa ese recorte igual que en el entrenamiento y muestra en pantalla la vocal predicha y la confianza.
Salir con la tecla q.
```

## Fase 7 – Memoria

```
Vamos con la Fase 7 del CLAUDE.md. Redacta en docs/memoria/ un borrador de la memoria en Markdown con:
introducción y objetivo; fundamentación teórica (CNN, convolución, pooling, ReLU, softmax, entropía cruzada, Adam,
backpropagation, BatchNorm, Dropout, data augmentation, división de datos y fuga de información, métricas);
captura del dataset (protocolo de fotos y recuento); preprocesamiento; arquitectura y justificación;
entrenamiento y ajuste (tabla de experimentos); evaluación y análisis de errores; conclusiones y mejoras;
y referencias bibliográficas (apuntes de la asignatura, Goodfellow et al. "Deep Learning", Chollet "Deep Learning with Python",
documentación de TensorFlow/Keras, artículos originales de BatchNorm, Dropout y Adam).
Usa solo números que estén en results/ y referencia las figuras de results/figuras/.
```
