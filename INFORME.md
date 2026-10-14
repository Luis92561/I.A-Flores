# Clasificación automática de especies de flores mediante *transfer learning* sobre Oxford Flowers-102

---

## 1. Portada y ficha del proyecto

| Campo | Valor |
|---|---|
| **Título** | Clasificación automática de especies de flores mediante *transfer learning* sobre Oxford Flowers-102 |
| **Asignatura** | Inteligencia Artificial |
| **Autor** | Luis Alfredo Jaramillo Restrepo |
| **Profesor** | Andrés Atehortúa Gutiérrez |
| **Fecha** | _Fecha de entrega_ |
| **Institución** | _Nombre de la universidad / facultad_ |
| **Tipo de trabajo** | Proyecto final de visión por computadora (clasificación de imágenes de grano fino) |
| **Estado** | Pendiente de la corrida final en Google Colab: las métricas de la sección 8 son marcadores de posición hasta que se ejecute el notebook con GPU. |

**Resumen del artefacto entregado.** Un sistema de IA que, dada una fotografía, identifica la
especie de flor entre las 102 categorías del conjunto de datos *Oxford Flowers-102* y devuelve
las cinco especies más probables con su probabilidad. El sistema consta de un paquete de código
en `src/` (datos, modelo, entrenamiento, evaluación, inferencia y demo), un notebook único
ejecutable de principio a fin en Google Colab con GPU T4 gratuita, y una interfaz interactiva
construida con Gradio.

**Documentos de referencia del proyecto.** El plan de trabajo y las decisiones de diseño están
en [PLAN_TRABAJO.md](PLAN_TRABAJO.md); las interfaces congeladas y la propiedad de cada archivo,
en el contrato interno; las instrucciones de uso y ejecución, en [README.md](README.md).
Este informe no duplica esos documentos: documenta el problema, el método, el protocolo
experimental y los resultados.

---

## 2. Resumen

Este informe documenta el diseño, la implementación y el protocolo de evaluación de un sistema
de visión por computadora capaz de identificar la especie de una flor a partir de una fotografía.
El problema se aborda como una clasificación de imágenes de grano fino sobre el conjunto de datos
Oxford Flowers-102, que contiene 8189 imágenes distribuidas en 102 especies, con solo 1020
imágenes de entrenamiento (diez por clase) y una elevada similitud visual entre varias categorías.
El enfoque emplea aprendizaje por transferencia sobre un *backbone* EfficientNet-B0 preentrenado
en ImageNet, con el *backbone* congelado durante las primeras épocas y una cabeza nueva
`Dropout(0,3) + Linear(1280, 102)`, seguida de un ajuste fino con tasa de aprendizaje diferencial.
El entrenamiento usa AdamW, suavizado de etiquetas, RandAugment, precisión mixta y parada
temprana. La evaluación se realiza sobre el *split* oficial de *test* y reporta exactitud top-1
y top-5, macro-F1, matriz de confusión y análisis de las confusiones más frecuentes. Los
resultados numéricos se incorporan tras la ejecución final en Google Colab; este documento los
deja marcados como pendientes y no adelanta cifras.

---

## 3. Introducción y planteamiento del problema

### 3.1 Contexto

La identificación de especies vegetales a partir de fotografías es una tarea con interés práctico
(agricultura, monitoreo de biodiversidad, educación ambiental) y con interés académico, porque es
un caso prototípico de **clasificación de imágenes de grano fino** (*fine-grained visual
classification*): las clases no se distinguen por su forma global, como un coche frente a un
perro, sino por diferencias sutiles de forma, textura y color de pétalos, hojas y estambres.

### 3.2 Formulación del problema

Dada una imagen $x \in \mathbb{R}^{H \times W \times 3}$ que contiene una flor, se busca estimar

$$P(y = c \mid x), \qquad c \in \{1, \dots, 102\},$$

donde cada $c$ corresponde a una de las 102 especies del catálogo. El sistema se implementa como
una red neuronal convolucional que produce 102 *logits*; tras aplicar *softmax* se obtiene una
distribución de probabilidad sobre las especies y se devuelve el *top-5* ordenado.

### 3.3 Por qué es un problema difícil

1. **Clases visualmente similares.** Varias de las 102 especies son casi indistinguibles a simple
   vista para una persona no experta, e incluso para un experto sin la flor en la mano. El propio
   conjunto de datos fue construido, en palabras de sus autores, con «gran similitud entre clases
   y pequeña similitud dentro de la clase» [1].
2. **Muy pocas muestras de entrenamiento.** El *split* oficial de entrenamiento contiene 1020
   imágenes para 102 clases, es decir, **10 imágenes por clase**. Con ese volumen, una red
   entrenada desde cero se sobreajusta de inmediato; memoriza las diez fotos en lugar de aprender
   el concepto de la especie.
3. **Variabilidad intraclase alta.** Las imágenes del conjunto de datos presentan «grandes
   variaciones de escala, pose e iluminación» [9]: la misma especie aparece fotografiada de frente,
   de perfil, con fondo de cielo, de suelo o de follaje, con distintas distancias focales.
4. **Desbalance entre clases.** En el conjunto completo, cada categoría tiene entre 40 y 258
   imágenes [9], de modo que la dificultad efectiva no es uniforme entre especies.
5. **Espacio de salida grande.** Con 102 clases, la probabilidad de acertar al azar es de
   aproximadamente 0,98 % en top-1 y de 4,9 % en top-5. Cualquier resultado útil debe alejarse
   muchísimo de esa línea base.

### 3.4 Objetivo

**Objetivo general.** Construir, entrenar y evaluar un clasificador de especies de flores que
alcance un desempeño sustancialmente superior a la línea base aleatoria sobre el *split* oficial
de *test* de Oxford Flowers-102, utilizando únicamente recursos gratuitos y reproducibles
(Google Colab con GPU T4).

**Objetivos específicos.**

1. Implementar un *pipeline* completo y modular: datos → aumento de datos → entrenamiento →
   evaluación → inferencia → demo interactivo.
2. Emplear *transfer learning* con un *backbone* preentrenado en ImageNet para compensar la
   escasez de datos de entrenamiento.
3. Reportar métricas estandarizadas y comparables con la literatura: exactitud top-1, exactitud
   top-5, macro-F1, matriz de confusión 102 × 102 y análisis de errores.
4. Ofrecer una interfaz de uso realista, con un umbral de confianza que permita al sistema
   declarar «no reconocida» cuando la imagen no pertenece al catálogo conocido.
5. Documentar honestamente las limitaciones del sistema y las líneas de trabajo futuro.

### 3.5 Alcance

**Lo que el sistema SÍ hace.**

- Clasifica **una imagen completa** en una de 102 especies conocidas, entrenando sobre los
  *splits* oficiales del conjunto de datos (sin reparticiones propias).
- Devuelve el *top-5* de especies con su probabilidad, lo que permite un uso asistido (una
  persona decide) en lugar de un uso automático ciego.
- Rechaza explícitamente las imágenes en las que no confía, mediante el umbral `min_conf`.
- Corre tanto en CPU como en GPU, con un modo *debug* que valida todo el *pipeline* en
  aproximadamente un minuto antes de gastar GPU.

**Lo que el sistema NO hace.**

- No reconoce especies fuera de las 102 del catálogo: ante una flor desconocida puede decir
  «no reconocida», pero eso es un rechazo, no una identificación.
- No detecta ni segmenta varias flores en una misma imagen: si la foto contiene un ramo, la
  predicción corresponde a la flor dominante según el encuadre.
- No entrena una CNN desde cero (se descartó por costo computacional y peor desempeño esperado).
- No incluye aplicación publicada en web o móvil, ni base de datos de usuarios, ni modelo de
  producción con monitoreo.
- No se validó con fotografías propias del autor; esa validación queda como trabajo futuro.

---

## 4. Marco teórico breve

### 4.1 Redes neuronales convolucionales

Una red neuronal convolucional (CNN) apila capas de convolución, normalización y no linealidad
que extraen representaciones jerárquicas de la imagen: las capas iniciales detectan bordes y
texturas, las intermedias patrones locales (manchas, pétalos, centros florales) y las finales
combinaciones de alto nivel cercanas a la categoría semántica. El sesgo inductivo de la
convolución —localidad y equivarianza a la traslación— hace que estos modelos sean adecuados para
imágenes y mucho más eficientes en parámetros que una red totalmente conectada equivalente.

La familia **EfficientNet** [2] propone escalar de forma compuesta y balanceada la profundidad,
la anchura y la resolución de entrada mediante un coeficiente único, partiendo de una arquitectura
base obtenida por búsqueda automática. **EfficientNet-B0** es el miembro más pequeño de la familia
y ofrece una relación precisión/costo muy favorable, lo que lo hace adecuado para GPU T4 y para
entrenamiento en tiempos de minutos. Como alternativa clásica, **ResNet** [3] introdujo las
conexiones residuales, que permiten entrenar redes muy profundas al facilitar el flujo del
gradiente; se usa aquí como referencia de comparación en la discusión de trabajo futuro.

### 4.2 Aprendizaje por transferencia y por qué funciona con pocos datos

El *transfer learning* consiste en reutilizar los pesos de un modelo entrenado en una tarea con
muchos datos (aquí, clasificación de 1000 clases de ImageNet) y adaptarlos a una tarea nueva con
pocos datos. La justificación empírica es que las representaciones aprendidas en las capas
iniciales e intermedias son **genéricas** y transferibles: Yosinski et al. [11] mostraron que las
características de las primeras capas son prácticamente independientes de la tarea y que
transferirlas mejora el desempeño cuando el conjunto de destino es pequeño.

En la práctica se aplica en dos etapas:

1. **Etapa congelada:** se conserva el *backbone* preentrenado como extractor de características
   fijo y se entrena únicamente la cabeza nueva `Dropout(0,3) → Linear(1280, 102)`. Esto evita
   que los gradientes, al principio ruidosos, destruyan los pesos preentrenados.
2. **Etapa de ajuste fino:** se descongela el *backbone* y se ajusta con una tasa de aprendizaje
   mucho menor que la de la cabeza, para refinar las representaciones sin deshacerlas.

Con 10 imágenes por clase, esta estrategia es prácticamente obligatoria: la alternativa
(entrenar desde cero) requeriría órdenes de magnitud más datos.

### 4.3 Aumento de datos

El aumento de datos (*data augmentation*) genera versiones transformadas de cada imagen de
entrenamiento (recortes, volteos, cambios de color y de geometría) que preservan la etiqueta. Su
efecto es doble: aumenta el tamaño efectivo del conjunto y actúa como regularizador, forzando al
modelo a ser invariante a transformaciones que no cambian la especie. **RandAugment** [10]
automatiza esta idea: en lugar de definir a mano una secuencia de operaciones, muestrea
aleatoriamente $N$ transformaciones de un espacio predefinido con una magnitud $M$ común, lo que
reduce el costo de ajustar hiperparámetros de aumento de datos.

### 4.4 Suavizado de etiquetas

El suavizado de etiquetas (*label smoothing*) [4] reemplaza la etiqueta *one-hot* por una
distribución que asigna $1-\varepsilon$ a la clase correcta y $\varepsilon/(K-1)$ al resto. Esto
evita que el modelo se vuelva excesivamente confiado, reduce el sobreajuste y mejora la
calibración de las probabilidades, lo cual es relevante aquí porque el umbral de confianza de la
sección 6.4 depende de que las probabilidades sean mínimamente interpretables.

### 4.5 Precisión mixta

La **precisión mixta** [6] combina aritmética de 16 bits (FP16) y de 32 bits (FP32) durante el
entrenamiento: las operaciones costosas se ejecutan en FP16 y el escalado de la pérdida
(*loss scaling*) evita el desbordamiento de gradientes en magnitudes pequeñas. En las GPU con
*tensor cores* el resultado es un entrenamiento notablemente más rápido y con menor uso de
memoria, a cambio de un ajuste de implementación (`torch.amp.autocast` + `GradScaler`). En este
proyecto la precisión mixta está activada por defecto (`amp=True`) y **se ignora automáticamente
cuando el dispositivo es CPU**, de modo que el mismo código corre en ambos entornos.

### 4.6 Tasa de aprendizaje diferencial y planificador

Una red de *transfer learning* tiene dos poblaciones de parámetros con necesidades distintas: la
cabeza, inicializada aleatoriamente, necesita una tasa de aprendizaje alta; el *backbone*
preentrenado necesita una tasa baja para no perder lo aprendido. El **aprendizaje con tasa
diferencial** asigna un grupo de parámetros a cada población dentro del mismo optimizador. Sobre
esa base se aplica un planificador de tipo coseno con calentamiento (*warmup*) inicial [13]: las
primeras épocas suben la tasa de aprendizaje de forma gradual para estabilizar el entrenamiento
de la cabeza nueva, y después decae suavemente siguiendo una curva coseno hasta casi cero, lo que
favorece la convergencia fina.

### 4.7 Parada temprana

La parada temprana (*early stopping*) monitorea la exactitud de validación y detiene el
entrenamiento cuando deja de mejorar durante un número fijo de épocas (`early_stop_patience`). Es
la defensa más directa contra el sobreajuste en un régimen de pocos datos: se conserva el
checkpoint del mejor época y se descartan las posteriores, que solo añaden varianza. En este
proyecto el criterio de guardado del checkpoint es exactamente el mismo, de modo que el modelo
final es el de mejor validación, no el de la última época.

### 4.8 La métrica top-5 y su relevancia con 102 clases

La **exactitud top-5** (*top-5 accuracy*) considera correcta una predicción si la clase verdadera
aparece entre las cinco clases más probables. Es la métrica estándar en tareas de clasificación
con muchas clases visualmente próximas, desde la competencia ImageNet [12], y aquí es
especialmente informativa por tres razones:

1. **Dificultad intrínseca del problema.** Con 102 especies muy similares, el error top-1 no
   distingue entre «el modelo confunde dos especies casi idénticas» y «el modelo no tiene idea».
   La brecha entre top-1 y top-5 mide exactamente esa diferencia: si top-5 es mucho mayor que
   top-1, el modelo sí reconoce la flor pero no logra decidir entre un puñado de candidatas.
2. **Utilidad real.** En un uso asistido, ofrecer cinco candidatas con probabilidad permite que
   la persona usuaria (o un experto botánico) resuelva la ambigüedad en segundos.
3. **Justicia con las clases difíciles.** Un modelo con top-1 moderado pero top-5 alto sigue
   siendo útil como sistema de apoyo; juzgarlo solo por top-1 castigaría en exceso la parte
   genuinamente ambigua del problema.

> **Advertencia metodológica.** El *test* de Flowers-102 tiene 1020 imágenes de validación y 6149
> de *test*, pero solo 10 imágenes de entrenamiento por clase. Por eso es indispensable evaluar
> sobre los *splits* oficiales y no sobre una partición propia: cualquier partición inventada
> dejaría de ser comparable con la literatura y, si se hiciera mal, produciría **fuga de datos**
> (imágenes de la misma planta en entrenamiento y en prueba), inflando las métricas.

---

## 5. Dataset: Oxford Flowers-102

### 5.1 Descripción

**Oxford Flowers-102** (también *102 Category Flower Dataset*) fue publicado por Nilsback y
Zisserman [1], [9] y contiene fotografías de **102 especies de flores que ocurren comúnmente en el
Reino Unido**. El conjunto completo reúne **8189 imágenes**; cada categoría tiene entre **40 y
258 imágenes** [9]. Las imágenes presentan grandes variaciones de escala, pose e iluminación, y el
propio conjunto incluye «categorías con gran variación intraclase y varias categorías muy
similares entre sí», lo que lo convierte en un banco de pruebas exigente de clasificación de grano
fino.

### 5.2 Particiones oficiales

El conjunto de datos define sus propias particiones, distribuidas en el archivo `setid.mat` de la
página oficial [9] y respetadas por `torchvision.datasets.Flowers102`:

| *Split* | Imágenes | Clases | Imágenes por clase | Uso en el proyecto |
|---|---:|---:|---:|---|
| Entrenamiento (*train*) | 1020 | 102 | 10 | Ajuste de los pesos del modelo |
| Validación (*val*) | 1020 | 102 | 10 | Selección del mejor *checkpoint* y parada temprana |
| Prueba (*test*) | 6149 | 102 | ~60 (variable) | Medición final de métricas |
| **Total** | **8189** | **102** | — | — |

### 5.3 Por qué se usan los *splits* oficiales

Se emplean las particiones oficiales, sin repartición propia, por tres razones:

1. **Comparabilidad con la literatura.** Los números publicados sobre Flowers-102 se obtienen
   sobre estas particiones. Una partición propia haría que las métricas de este informe no fueran
   comparables con ninguna referencia.
2. **Trazabilidad y reproducibilidad.** Cualquier persona puede reproducir la evaluación con el
   mismo comando y obtener la misma partición, sin depender de una semilla de reparto.
3. **Separación de las particiones, con una salvedad medida.** Las particiones oficiales son
   disjuntas a nivel de imagen (ninguna ruta se repite entre *train*, *val* y *test*) y no se
   aplica aumentación de datos en validación ni en prueba. Ahora bien, una comprobación por
   contenido (MD5) encontró **cuatro pares de imágenes byte a byte idénticas**, de los cuales
   **dos cruzan particiones** (*train*↔*test* y *val*↔*test*). Es una fuga residual de dos
   imágenes sobre 8189 (≈0,02 %), irrelevante para las métricas agregadas, pero se documenta
   porque invalida cualquier afirmación de separación estricta "por planta fotografiada".

La consecuencia práctica es que el conjunto de *test* (6149 imágenes) es considerablemente mayor
que el de entrenamiento (1020). Esto es intencional en el diseño del *benchmark*: la dificultad
no está en tener muchos datos, sino en aprender con muy pocos.

### 5.4 Limitación estructural: 10 imágenes de entrenamiento por clase

El punto crítico del conjunto de datos es que el *split* de entrenamiento ofrece **10 imágenes por
clase**. Diez fotografías no alcanzan para cubrir la variabilidad natural de una especie
(diferentes etapas de floración, ángulos, fondos, iluminación y subespecies). Esta restricción es
también la razón por la que los resultados de este tipo de tarea dependen fuertemente del
preentrenamiento en ImageNet y de una regularización agresiva, y explica por qué la varianza entre
ejecuciones no es despreciable.

### 5.5 Preparación de los datos en el proyecto

La carga de los tres *splits* y la construcción de las transformaciones están implementadas en
`src/data.py` (`build_dataloaders`, `build_transforms`). Los nombres legibles de las 102 especies
se obtienen del propio conjunto de datos o, si no están disponibles, del archivo
`cat_to_name.json` de la página de Oxford, que se cachea localmente; si tampoco hubiera conexión,
el sistema degrada a nombres genéricos (`especie_001` … `especie_102`) sin fallar. Los
*DataLoaders* de validación y prueba no se barajan y no descartan el último lote parcial, de modo
que la evaluación recorre siempre el mismo conjunto completo.

---

## 6. Metodología

### 6.1 Arquitectura del modelo

Se emplea **EfficientNet-B0** [2] con pesos preentrenados en ImageNet como extractor de
características, más una cabeza de clasificación nueva:

```
Entrada (B, 3, 224, 224)
   │
   ├─ Backbone EfficientNet-B0  (bloques convolucionales preentrenados en ImageNet)
   │        └─ salida: mapa de características de 1280 canales
   │
   ├─ AdaptiveAvgPool2d(1)      → vector de 1280 características
   │
   └─ Cabeza:  Dropout(0.3) → Linear(1280 → 102)   → 102 logits
```

El *backbone* se usa como extractor de características; su clasificador original de 1000 clases
se descarta por completo. La capa lineal nueva tiene 1280 × 102 + 102 = 130 662 parámetros, y el
resto de los parámetros del modelo conserva los pesos preentrenados.

**Esquema de entrenamiento en dos etapas.**

| Etapa | Épocas | Estado del *backbone* | Tasa de aprendizaje | Objetivo |
|---|---|---|---|---|
| 1. Cabeza | 1 a `freeze_epochs` (3) | Congelado (`requires_grad = False`) | Cabeza: `lr` = 3 × 10⁻⁴ | Aprender un clasificador válido sin perturbar los pesos preentrenados |
| 2. Ajuste fino | `freeze_epochs` + 1 en adelante | Entrenable | Cabeza: `lr`; *backbone*: `backbone_lr` = 3 × 10⁻⁵ | Refinar las representaciones de alto nivel con un paso mucho más pequeño |

Durante la primera etapa el clasificador aleatorio produce gradientes grandes; mantener el
*backbone* congelado evita que esos gradientes degraden características valiosas. En la segunda
etapa el ajuste se hace con una tasa dos órdenes de magnitud menor para el *backbone*, que es la
idea central del aprendizaje con tasa diferencial. El optimizador se crea una sola vez con los dos
grupos de parámetros, de modo que la transición entre etapas no requiere reconstruirlo.

### 6.2 Preprocesado y aumento de datos

| Fase | Transformaciones |
|---|---|
| Entrenamiento | `RandomResizedCrop(224)`, volteo horizontal, **RandAugment** [10], normalización con las estadísticas de ImageNet |
| Validación y prueba | `Resize(256)` → `CenterCrop(224)` → normalización con las estadísticas de ImageNet |

Las estadísticas de normalización son las de ImageNet, coherentes con el preentrenamiento del
*backbone*: media `(0.485, 0.456, 0.406)` y desviación estándar `(0.229, 0.224, 0.225)`,
definidas como constantes `MEAN` y `STD` en `src/config.py`.

El aumento de datos es agresivo de forma deliberada, porque es una de las pocas defensas
disponibles contra el sobreajuste con 10 imágenes por clase. La secuencia exacta y sus
parámetros viven en `src/data.py` (`build_transforms`), que es la fuente autoritativa; este
informe describe el diseño, no duplica la implementación. Junto con el aumento de datos, se
aplican las siguientes medidas de regularización: `Dropout(0.3)` en la cabeza, `label_smoothing`
= 0.1 [4], `weight_decay` = 1 × 10⁻⁴ y parada temprana.

En evaluación **no** se aplica aumento de datos: la imagen se redimensiona y se recorta al centro,
de manera que la métrica mida el reconocimiento y no la robustez a transformaciones aleatorias.

### 6.3 Hiperparámetros reales

Todos los valores de la tabla siguiente provienen literalmente de `src/config.py` (clase
`Config`), que es la fuente única de verdad del proyecto.

| Hiperparámetro | Campo en `src/config.py` | Valor |
|---|---|---|
| *Backbone* | `backbone` | `"efficientnet_b0"` |
| Pesos preentrenados | `pretrained` | `True` (ImageNet) |
| *Dropout* de la cabeza | `dropout` | 0,3 |
| Número de clases | `num_classes` | 102 |
| Tamaño de imagen | `img_size` / `IMAGE_SIZE` | 224 × 224 |
| Épocas (máximo) | `epochs` | 25 |
| Tamaño de lote | `batch_size` | 32 |
| Tasa de aprendizaje de la cabeza | `lr` | 3 × 10⁻⁴ |
| Tasa de aprendizaje del *backbone* | `backbone_lr` | 3 × 10⁻⁵ |
| Optimizador | `build_optimizer` en `src/model.py` | AdamW [5] |
| Decaimiento de pesos | `weight_decay` | 1 × 10⁻⁴ |
| Suavizado de etiquetas | `label_smoothing` | 0,1 |
| Épocas de calentamiento (*warmup*) | `warmup_epochs` | 2 |
| Épocas con *backbone* congelado | `freeze_epochs` | 3 |
| Paciencia de parada temprana | `early_stop_patience` | 6 |
| Precisión mixta | `amp` | `True` (ignorada en CPU) |
| *Top-k* de inferencia | `topk` | 5 |
| Umbral de confianza | `min_conf` | 0,25 |
| Semilla | `seed` | 42 |
| Procesos de carga de datos | `num_workers` | 2 (0 en modo *debug*) |
| Dispositivo | `device` | `"auto"` |
| Modo *debug* | `debug` / `debug_per_class` | `False` / 4 |

**Nota sobre la selección de hiperparámetros.** Estos valores son decisiones de diseño del
proyecto, documentadas como tales en [PLAN_TRABAJO.md](PLAN_TRABAJO.md) y
el contrato interno; no provienen de una búsqueda sistemática de hiperparámetros ni de
una optimización automática. Se eligieron siguiendo las prácticas habituales para *transfer
learning* con pocos datos (tasa baja para el *backbone*, AdamW con decaimiento de pesos,
calentamiento y decaimiento coseno, suavizado de etiquetas) y con la restricción de que la corrida
completa quepa en el presupuesto de tiempo de una GPU T4 gratuita. **La ausencia de una búsqueda
de hiperparámetros es una limitación reconocida** (sección 10).

### 6.4 Semillas y reproducibilidad

- `src/utils.py` define `set_seed(seed)`, que fija la semilla de `random`, de `numpy` y de `torch`
  (CPU y CUDA), además de la variable de entorno `PYTHONHASHSEED`, con el valor `seed = 42` por
  defecto.
- El *pipeline* llama a `set_seed(cfg.seed)` al inicio del entrenamiento.
- La partición de datos es fija por construcción (particiones oficiales), de modo que no hay
  aleatoriedad en el reparto.
- El checkpoint guarda la configuración completa con la que se entrenó
  (`config_usada.json` y la clave `config` dentro de `best.pt`), junto con la época y la
  exactitud de validación alcanzadas.
- **Límite honesto de la reproducibilidad.** Fijar las semillas hace que una misma corrida sea
  repetible, pero no garantiza resultados idénticos entre hardware o versiones de biblioteca
  distintas: la selección de *kernels* de cuDNN y las operaciones de precisión mixta introducen
  no determinismo. Además, con 10 imágenes por clase la varianza entre semillas distintas es
  apreciable, por lo que **las cifras de la sección 8 corresponden a una sola corrida** y no
  incluyen intervalos de confianza.

### 6.5 Umbral de confianza en la inferencia

En la fase de inferencia, si la probabilidad máxima del *top-1* es menor que `min_conf` = 0,25,
el sistema no devuelve ninguna de las 102 especies, sino el mensaje
**«no reconocida / fuera del catálogo»**.

**Justificación.** El catálogo es cerrado: el modelo solo conoce 102 especies. Sin umbral, el
sistema se vería obligado a elegir siempre una clase, incluso ante una flor que no está en el
catálogo, una imagen borrosa o una fotografía que no contiene ninguna flor. En esos casos la
distribución *softmax* se reparte entre muchas clases y la probabilidad máxima es baja, lo que
constituye una señal de duda aprovechable. El valor 0,25 no asegura una detección perfecta de
imágenes desconocidas —ningún umbral lo hace— y su calibración óptima depende del uso previsto:
un umbral más alto rechaza más imágenes dudosas a costa de rechazar también flores conocidas
difíciles. Además, como el modelo se entrenó con suavizado de etiquetas [4], sus probabilidades
tienden a ser menos extremas que las de un modelo entrenado con etiquetas *one-hot*, lo que
refuerza la utilidad de un umbral moderado. El umbral es un parámetro configurable
(`min_conf` en `Config`, `--min-conf` en la CLI de `src/predict.py`), no una constante oculta.

---

## 7. Protocolo experimental

### 7.1 Entrenamiento

```
python -m src.train                    # corrida completa (25 épocas, GPU)
python -m src.train --debug --epochs 2 --device cpu   # corrida de humo en CPU
```

| Aspecto | Definición |
|---|---|
| Datos de entrenamiento | *Split* oficial *train* (1020 imágenes, 102 clases) |
| Datos de validación | *Split* oficial *val* (1020 imágenes) |
| Criterio de *checkpoint* | Se guarda `best.pt` **solo** cuando mejora la exactitud top-1 de validación |
| Parada temprana | Si la exactitud de validación no mejora en 6 épocas consecutivas |
| Precisión | Precisión mixta (FP16/FP32) cuando el dispositivo es CUDA y `amp=True` |
| Planificador | Decaimiento coseno con 2 épocas de calentamiento, aplicado a los dos grupos de parámetros |
| Semilla | 42 (`set_seed`), fijada al inicio |
| Duración típica | ~15 a 25 minutos en GPU T4 para 25 épocas, lote 32 a 224 px |

Se distingue explícitamente un **modo *debug*** (`debug=True`), que usa un subconjunto de 4
imágenes por clase, 2 épocas como máximo, sin calentamiento, sin congelado y sin precisión mixta.
Su propósito es verificar en aproximadamente un minuto que la descarga del conjunto de datos, los
*DataLoaders*, el modelo y las rutas funcionan en Colab, **antes** de gastar GPU en la corrida
real. Las salidas del modo *debug* se escriben en `resultados/debug/`, de modo que nunca
sobrescriben las de la corrida completa.

### 7.2 Evaluación

```
python -m src.evaluate --checkpoint resultados/best.pt --split test
```

La evaluación se ejecuta **una sola vez**, al final, sobre el *split* oficial de *test*, con el
*checkpoint* seleccionado por validación. El *test* no se usa en ninguna decisión de
entrenamiento: ni para elegir la época, ni para ajustar hiperparámetros, ni para comparar
variantes. Los *transforms* se reconstruyen a partir del `img_size` guardado en el propio
*checkpoint*, de modo que un modelo entrenado a otra resolución se evalúa coherentemente.

### 7.3 Qué se reporta

| Artefacto | Contenido |
|---|---|
| Exactitud top-1 | Proporción de imágenes cuya clase verdadera es la más probable |
| Exactitud top-5 | Proporción de imágenes cuya clase verdadera está entre las cinco más probables |
| Macro-F1 | Media no ponderada del F1 por clase; penaliza el mal desempeño en clases raras |
| `classification_report` | Precisión, exhaustividad y F1 por cada una de las 102 especies |
| Matriz de confusión | Matriz 102 × 102, normalizada, en `confusion_matrix.png` |
| Confusiones más frecuentes | Ranking de pares (clase real, clase predicha) ordenado por número de casos, en `top_confusions.png` |
| Predicciones por imagen | CSV con columnas `ruta, etiqueta_real, prediccion, prob, correcto` |
| Curvas de entrenamiento | Pérdida y exactitud por época (entrenamiento y validación) en `curvas.png` |
| Historial completo | `history.json`: pérdida, exactitud, top-5, tasa de aprendizaje, mejor época y segundos por corrida |
| Métricas agregadas | `metrics.json`: `top1`, `top5`, `macro_f1`, `per_class`, `top_confusions`, `segundos` |

Además de las cifras agregadas, el protocolo incluye un **análisis de errores** (sección 9) que
examina qué especies se confunden entre sí y por qué. Un informe que solo presentara exactitud
top-1 sería insuficiente para un problema de grano fino, donde el patrón de errores es la
información más útil.

### 7.4 Hardware y entorno de ejecución

| Elemento | Valor |
|---|---|
| Entorno de ejecución | Google Colab (cuaderno `notebooks/flores_colab.ipynb`) |
| GPU | NVIDIA T4 (16 GB), acelerador gratuito de Colab |
| Tiempo estimado de la corrida completa | ~15 a 25 minutos (25 épocas, lote 32, 224 px) |
| Tiempo del modo *debug* | ~1 minuto (verificación del *pipeline*) |
| Alternativa en CPU | Funciona, pero es órdenes de magnitud más lento; se usa solo para pruebas de humo locales |
| Persistencia | *Checkpoint*, métricas y figuras se pueden guardar en Google Drive para sobrevivir a una desconexión |

El proyecto **no requiere GPU local**: el entrenamiento real se ejecuta en la GPU T4 de Colab.
El entorno local se usa únicamente para pruebas de humo y verificación del código en CPU.

---

## 8. Resultados

> **Estado: pendiente de la corrida final en Google Colab.** Las celdas de esta sección se
> completan con la salida real de `python -m src.evaluate`. Ninguna cifra de esta sección ha sido
> estimada, simulada ni inventada.

### 8.1 Métricas principales (split oficial de *test*, 6149 imágenes)

| Métrica | Valor | Fuente del dato |
|---|---|---|
| Exactitud top-1 | _pendiente de la corrida en Colab_ | `resultados/metrics.json` → `top1` |
| Exactitud top-5 | _pendiente de la corrida en Colab_ | `resultados/metrics.json` → `top5` |
| Macro-F1 | _pendiente de la corrida en Colab_ | `resultados/metrics.json` → `macro_f1` |
| Tiempo de entrenamiento | _pendiente de la corrida en Colab_ | `resultados/history.json` → `segundos` |
| Mejor época | _pendiente de la corrida en Colab_ | `resultados/history.json` → `best_epoch` |
| Épocas ejecutadas | _pendiente de la corrida en Colab_ | `resultados/history.json` → `epochs_run` |
| Mejor exactitud de validación | _pendiente de la corrida en Colab_ | `resultados/history.json` → `best_val_acc` |
| Dispositivo utilizado | _pendiente de la corrida en Colab_ | `resultados/history.json` → `device` |
| Número de parámetros entrenables | _pendiente de la corrida en Colab_ | `resultados/history.json` → `num_params` |

**Cómo obtener estos valores.** Tras ejecutar el notebook completo, ejecutar
`python -m src.evaluate --checkpoint resultados/best.pt --split test` y copiar los valores de
`resultados/metrics.json` y `resultados/history.json` en la tabla anterior, sustituyendo los
marcadores `_pendiente de la corrida en Colab_`.

### 8.2 Referencia de la literatura

A modo de contexto, la literatura sobre *transfer learning* en Oxford Flowers-102 sitúa a los
*backbones* de la familia EfficientNet en un rango aproximado de **~92 % a ~97 % de exactitud
top-1** sobre el *split* oficial de *test*. Se cita como referencia para juzgar si la corrida
propia resultó razonable.

> **Esta referencia NO es una medición de este proyecto.** Es un rango reportado en la literatura
> y depende del *backbone*, del esquema de ajuste fino, de las épocas y de la resolución de
> entrada. No debe presentarse como resultado propio, ni usarse para rellenar la tabla 8.1. Si la
> corrida final queda por debajo de ese rango, la conclusión correcta es analizar por qué (pocas
> épocas, sobreajuste, umbral de decisión), no ajustar el número.

Como referencia adicional, el trabajo original de Nilsback y Zisserman [1] —anterior al auge de
las CNN, con descriptores de forma, textura y color combinados mediante un SVM de múltiples
kernels— reporta **72,8 %** de exactitud media combinando los cuatro descriptores, frente a
**55,1 %** del mejor descriptor individual. Esta comparación ilustra cuánto ha aportado el
aprendizaje profundo por transferencia a este problema.

### 8.3 Figuras

<!-- AQUI VAN LAS FIGURAS: curvas.png, confusion_matrix.png -->

**Instrucciones para insertar las figuras.** Después de la corrida en Colab, copiar las figuras
generadas a la carpeta `resultados/figuras/` del repositorio y sustituir el comentario HTML
anterior por las líneas siguientes (ajustando los pies de figura con los valores reales):

```markdown
### Curvas de entrenamiento

![Curvas de pérdida y exactitud por época](resultados/figuras/curvas.png)

**Figura 1.** Evolución de la pérdida y de la exactitud (top-1 y top-5) en entrenamiento y
validación a lo largo de las _N_ épocas ejecutadas. Se marca la mejor época
(_pendiente de la corrida en Colab_). Una brecha creciente entre las curvas de entrenamiento y de
validación indica sobreajuste, esperable con 10 imágenes de entrenamiento por clase.

### Matriz de confusión

![Matriz de confusión 102 x 102 sobre el split de test](resultados/figuras/confusion_matrix.png)

**Figura 2.** Matriz de confusión 102 × 102 sobre el *split* oficial de *test*. Las filas
corresponden a la clase real y las columnas a la clase predicha. La diagonal concentra los
aciertos; los valores fuera de la diagonal identifican las especies que el modelo confunde entre
sí.

### Confusiones más frecuentes

![Ranking de confusiones más frecuentes](resultados/figuras/top_confusions.png)

**Figura 3.** Pares (especie real, especie predicha) con mayor número de confusiones. Es la
figura que se comenta en la sección 9.
```

**Notas útiles para esta inserción.**

- Las tres figuras (`curvas.png`, `confusion_matrix.png`, `top_confusions.png`) se generan
  automáticamente: `curvas.png` al entrenar y las dos restantes al evaluar.
- Si el informe se entrega como PDF o documento independiente, conviene copiar las imágenes a una
  carpeta `figuras/` junto al informe y ajustar las rutas relativas en las líneas anteriores.
- La tabla de la sección 9.2 se rellena con las 20 primeras entradas del campo `top_confusions`
  de `resultados/metrics.json`.

---

## 9. Análisis de errores (plantilla)

### 9.1 Cómo leer las confusiones

El análisis de errores se realiza sobre la matriz de confusión 102 × 102 y sobre el ranking de
`top_confusions`. El procedimiento es el siguiente:

1. **Identificar la diagonal.** La diagonal de la matriz de confusión contiene los aciertos de
   cada especie. Las especies con valores bajos en su diagonal son las que el modelo reconoce
   peor, independientemente de lo bien que le vaya en promedio.
2. **Localizar los máximos fuera de la diagonal.** Cada valor alto fuera de la diagonal es un par
   (especie real → especie predicha) que el modelo confunde de manera sistemática. Un par con
   muchos casos no es ruido: es una ambigüedad real entre dos clases.
3. **Comprobar la simetría.** Si la confusión va en un solo sentido (A se predice como B, pero B
   casi nunca como A), suele indicar un desbalance en los datos o diferencias de frecuencia:
   el modelo «tira» hacia la clase más representada o más fácil.
4. **Cruzar con el top-5.** Si un par confundido aparece casi siempre dentro del *top-5* del
   modelo, el error es de decisión final, no de representación: el modelo «ve» correctamente las
   dos candidatas pero no logra separarlas.
5. **Inspeccionar imágenes concretas.** El CSV de predicciones permite localizar las imágenes mal
   clasificadas de una especie concreta y revisarlas visualmente (fondo, iluminación, encuadre,
   oclusión), para distinguir un error del modelo de una etiqueta dudosa o de una imagen atípica.

### 9.2 Registro de las confusiones más frecuentes

| # | Especie real | Especie predicha | Casos | Observación | Fuente |
|---:|---|---|---:|---|---|
| 1 | _pendiente de la corrida en Colab_ | _pendiente de la corrida en Colab_ | _pendiente_ | | `metrics.json` → `top_confusions` |
| 2 | _pendiente_ | _pendiente_ | _pendiente_ | | `metrics.json` → `top_confusions` |
| … | | | | | |
| 20 | _pendiente_ | _pendiente_ | _pendiente_ | | `metrics.json` → `top_confusions` |

### 9.3 Qué se espera encontrar

Esta subsección describe **hipótesis previas**, no resultados. No debe leerse como hallazgo.

1. **Concentración del error en pares de especies visualmente similares.** Es el patrón esperado
   en clasificación de grano fino y es el motivo por el cual el conjunto de datos fue diseñado con
   «varias categorías muy similares» [9]. Se espera que los pares más confundidos correspondan a
   especies que comparten forma y color de pétalos, y que la mayoría de esos pares caigan dentro
   del *top-5* del modelo.
2. **Desempeño desigual entre clases.** Por el desbalance del conjunto completo (entre 40 y 258
   imágenes por clase [9]) y por la diferencia intrínseca de dificultad, se espera una macro-F1
   inferior a la exactitud top-1 agregada: el promedio por clase castiga a las especies difíciles
   que el promedio global diluye.
3. **Peor desempeño en clases con mucha variabilidad intraclase.** Algunas categorías incluyen
   subespecies, híbridos o etapas de floración muy distintas; con 10 imágenes de entrenamiento es
   poco probable cubrir esa variabilidad.
4. **Posible sesgo hacia especies con imágenes más homogéneas.** Las clases con fotografías de
   fondo e iluminación muy consistentes tienden a aprenderse mejor, y pueden capturar errores de
   clases visualmente parecidas pero más variables.
5. **Errores asociados a la imagen, no a la especie.** Fotos con la flor pequeña en el encuadre,
   con fondo dominante, con poca luz o con oclusiones parciales. El *pipeline* no incluye
   detección de objetos, así que no puede aislar la flor del fondo.

**Advertencia.** Las cinco hipótesis anteriores son expectativas derivadas de las características
del conjunto de datos y del problema, y sirven como guía de lectura de las figuras. Solo el
análisis de la matriz de confusión real de la corrida final permite confirmarlas o refutarlas.

---

## 10. Limitaciones

1. **Catálogo cerrado de 102 especies.** El modelo solo puede predecir una de las 102 especies con
   las que se entrenó. Ante una flor distinta, la única respuesta honesta es el rechazo por umbral
   de confianza («no reconocida / fuera del catálogo»), que **no identifica** la especie
   desconocida: solo evita afirmar algo falso. El parámetro `min_conf = 0.25` reduce los falsos
   positivos, pero no garantiza detectar todos los casos fuera de catálogo, y su valor óptimo
   depende del uso previsto.

2. **Sesgo del conjunto de datos.** Flowers-102 se recolectó principalmente en el **Reino Unido** y
   con fotografías «de catálogo»: fondo, iluminación y encuadre relativamente homogéneos [9]. Un
   buen resultado en el *test* oficial no garantiza un buen desempeño con fotografías tomadas en
   otras geografías, con fondos cargados, con poca luz, o de subespecies e híbridos distintos de
   los del catálogo. La distribución de *test* es la misma que la de entrenamiento, así que la
   métrica mide generalización **dentro** del conjunto de datos, no fuera de él.

3. **Una sola flor por imagen, sin detección.** El sistema clasifica la imagen completa. No
   localiza, no detecta y no segmenta flores: en una foto con varias flores (un ramo, un jardín, un
   prado) la predicción corresponde a la flor dominante según el encuadre, o a ninguna si la
   confianza cae por debajo del umbral. Tampoco hay garantía de que la imagen contenga
   efectivamente una flor.

4. **Solo 10 imágenes de entrenamiento por clase.** Es la limitación más determinante. Diez
   fotografías no cubren la variabilidad real de una especie, y el sobreajuste es el riesgo
   principal del proyecto. Se mitiga con aumento de datos fuerte, suavizado de etiquetas,
   `weight_decay`, `dropout`, parada temprana y selección del mejor *checkpoint* por validación,
   pero no se elimina.

5. **Sin validación con fotografías propias.** Todas las métricas de este informe provienen del
   *test* oficial del conjunto de datos. No se realizó ninguna prueba con imágenes tomadas por el
   autor en condiciones reales (teléfono móvil, interiores, fondos domésticos). Por lo tanto, el
   desempeño en uso real **no está medido** y podría ser sustancialmente inferior al del *test*.

6. **Hiperparámetros no optimizados.** Los valores de la sección 6.3 son decisiones de diseño
   razonadas, alineadas con las prácticas habituales, pero no son el resultado de una búsqueda
   sistemática. No se puede afirmar que sean los mejores para esta tarea.

7. **Una sola corrida, sin intervalos de confianza.** Con 10 imágenes por clase, la varianza entre
   semillas es apreciable. Al reportar una única ejecución (semilla 42) no se cuantifica esa
   incertidumbre; las diferencias pequeñas entre variantes del modelo no serían concluyentes sin
   repetir la corrida con varias semillas.

8. **Dependencia del preentrenamiento en ImageNet.** El desempeño del sistema depende en gran
   medida de los pesos preentrenados. Si su descarga falla (red bloqueada), el código degrada de
   forma controlada a pesos aleatorios, pero en ese caso el desempeño esperado cae drásticamente y
   los resultados no son comparables con los de una corrida normal.

9. **Licencias y uso responsable.** Las imágenes del conjunto de datos se distribuyen para uso
   académico y de investigación y pertenecen a sus autores originales; no se redistribuyen en el
   repositorio y no deben usarse con fines comerciales.

---

## 11. Conclusiones y trabajo futuro

### 11.1 Conclusiones

El proyecto implementa un sistema completo y reproducible de clasificación de especies de flores
sobre Oxford Flowers-102, desde la carga de los *splits* oficiales hasta una demostración
interactiva, y establece un protocolo de evaluación honesto y comparable con la literatura.

Las conclusiones que pueden sostenerse **antes** de la corrida final son las siguientes:

1. **El enfoque es metodológicamente adecuado al problema.** Con 10 imágenes de entrenamiento por
   clase, el aprendizaje por transferencia sobre un *backbone* preentrenado en ImageNet es la
   opción razonable frente a entrenar desde cero [11].
2. **La dificultad del problema es real y está documentada.** El propio conjunto de datos fue
   construido con clases muy similares entre sí y gran variabilidad intraclase [1], [9]; el error
   residual no es un defecto de implementación, sino la parte genuinamente ambigua de la tarea.
3. **La separación estricta entre validación y prueba es una garantía de honestidad.** El *test*
   no participa en ninguna decisión de entrenamiento, y el *checkpoint* se selecciona por
   exactitud de validación, no por la métrica final.
4. **Las cifras de desempeño aún no existen.** Este informe no reporta resultados propios porque
   la corrida en Google Colab está pendiente; cualquier número de la sección 8 es, hasta entonces,
   un marcador de posición. Esta decisión es deliberada: es preferible un informe incompleto y
   veraz a uno completo y especulativo.

### 11.2 Trabajo futuro

Las líneas siguientes se ordenan de menor a mayor esfuerzo estimado.

1. **Ajuste fino con fotografías propias.** El paso de mayor valor práctico. Consiste en anotar un
   conjunto pequeño de fotografías tomadas por el autor (idealmente decenas de imágenes por
   especie de interés) y ajustar el modelo ya entrenado con una tasa de aprendizaje muy baja.
   Permite adaptar el sistema al dominio real de uso (teléfono móvil, fondos domésticos) y medir
   la brecha respecto al *test* oficial, que hoy no está cuantificada.
2. **Ampliación del catálogo.** Añadir especies locales al conjunto de clases, ya sea repitiendo
   el ajuste fino con las clases nuevas o incorporando conjuntos de datos complementarios. Requiere
   resolver el problema del desbalance entre clases antiguas y nuevas, y volver a evaluar con un
   conjunto de prueba propio, porque las particiones oficiales solo cubren las 102 especies
   originales.
3. **Detección multi-flor.** Incorporar un detector (por ejemplo una familia tipo YOLO o un
   *pipeline* de detección + clasificación) para localizar cada flor en la imagen y clasificarla
   por separado. Es el cambio que permitiría usar el sistema con ramos, jardines o praderas, donde
   el enfoque actual solo captura la flor dominante.
4. **Exportación a ONNX o TFLite para uso móvil.** Serializar el modelo a un formato portable y
   cuantizarlo (por ejemplo a enteros de 8 bits) para ejecutarlo en un teléfono sin conexión.
   Interesa medir la pérdida de exactitud que introduce la cuantización, ya que es un cambio de
   compromiso entre tamaño, latencia y precisión.
5. **Comparación sistemática de *backbones*.** El contrato de `src/model.py` ya soporta
   `efficientnet_b0` (por defecto), `resnet50` [3] y `mobilenet_v3_large`. Entrenar los tres con
   idéntico protocolo y semilla permitiría comparar exactitud, macro-F1, número de parámetros y
   tiempo de entrenamiento. Ampliar la comparación a `efficientnet_b3` permitiría evaluar si un
   modelo mayor compensa su costo en una GPU T4.
6. **Búsqueda de hiperparámetros y repetición con varias semillas.** Explorar de forma acotada la
   tasa de aprendizaje del *backbone*, el número de épocas y la resolución de entrada, y repetir
   la configuración elegida con al menos tres semillas para reportar media y desviación estándar.
   Es el complemento necesario para que las diferencias entre variantes sean concluyentes.
7. **Reanudación de entrenamiento y entrenamiento distribuido.** El proyecto documenta como
   mejora futura una función de reanudación desde *checkpoint* dentro del propio bucle de
   entrenamiento, útil frente a las desconexiones de Colab; hoy la reanudación se resuelve
   recuperando `resultados/` desde Drive y saltando directamente a la evaluación o a la
   demostración.
8. **Explicabilidad.** Generar mapas de activación (por ejemplo Grad-CAM) para visualizar en qué
   región de la imagen se apoya la predicción. Sería un complemento valioso del análisis de
   errores: permitiría distinguir un error por confusión entre especies similares de un error por
   atender a un fondo o a un objeto irrelevante.

---

## 12. Referencias

Formato IEEE, ordenadas por aparición en el texto.

[1] M-E. Nilsback and A. Zisserman, «Automated flower classification over a large number of
classes», en *Proc. 6th Indian Conf. Computer Vision, Graphics and Image Processing (ICVGIP)*,
Bhubaneswar, India, 2008, pp. 722–729, doi: 10.1109/ICVGIP.2008.47.

[2] M. Tan and Q. V. Le, «EfficientNet: Rethinking model scaling for convolutional neural
networks», en *Proc. 36th Int. Conf. Machine Learning (ICML)*, Long Beach, CA, EE. UU., 2019,
pp. 6105–6114.

[3] K. He, X. Zhang, S. Ren and J. Sun, «Deep residual learning for image recognition», en *Proc.
IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, EE. UU., 2016,
pp. 770–778, doi: 10.1109/CVPR.2016.90.

[4] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna, «Rethinking the inception
architecture for computer vision», en *Proc. IEEE Conf. Computer Vision and Pattern Recognition
(CVPR)*, Las Vegas, NV, EE. UU., 2016, pp. 2818–2826, doi: 10.1109/CVPR.2016.308.

[5] I. Loshchilov and F. Hutter, «Decoupled weight decay regularization», en *Proc. Int. Conf.
Learning Representations (ICLR)*, New Orleans, LA, EE. UU., 2019.

[6] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston,
O. Kuchaiev, G. Venkatesh and H. Wu, «Mixed precision training», en *Proc. Int. Conf. Learning
Representations (ICLR)*, Vancouver, BC, Canadá, 2018.

[7] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin,
N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani,
S. Chilamkurthy, B. Steiner, L. Fang, J. Bai and S. Chintala, «PyTorch: An imperative style,
high-performance deep learning library», en *Advances in Neural Information Processing Systems
(NeurIPS) 32*, 2019, pp. 8024–8035.

[8] *TorchVision: Datasets, transforms and models specific to computer vision*, PyTorch.
[En línea]. Disponible en: <https://github.com/pytorch/vision>. [Consultado: _fecha_].

[9] M-E. Nilsback and A. Zisserman, «102 Category Flower Dataset», Visual Geometry Group,
University of Oxford. [En línea]. Disponible en:
<https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>. [Consultado: _fecha_].

[10] E. D. Cubuk, B. Zoph, J. Shlens and Q. V. Le, «RandAugment: Practical automated data
augmentation with a reduced search space», en *Proc. IEEE/CVF Conf. Computer Vision and Pattern
Recognition Workshops (CVPRW)*, 2020, arXiv:1909.13719.

[11] J. Yosinski, J. Clune, Y. Bengio and H. Lipson, «How transferable are features in deep neural
networks?», en *Advances in Neural Information Processing Systems (NeurIPS) 27*, 2014,
pp. 3320–3328.

[12] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy,
A. Khosla, M. Bernstein, A. C. Berg and L. Fei-Fei, «ImageNet large scale visual recognition
challenge», *International Journal of Computer Vision*, vol. 115, n.º 3, pp. 211–252, 2015,
doi: 10.1007/s11263-015-0816-y.

[13] I. Loshchilov and F. Hutter, «SGDR: Stochastic gradient descent with warm restarts», en
*Proc. Int. Conf. Learning Representations (ICLR)*, Toulon, Francia, 2017.

---

## Apéndice A. Trazabilidad de las cifras de este informe

Para dejar explícito qué es dato y qué no lo es:

| Tipo de cifra | Ejemplos en este informe | Origen |
|---|---|---|
| Hiperparámetros | 25 épocas, lote 32, `lr` 3 × 10⁻⁴, `backbone_lr` 3 × 10⁻⁵, `dropout` 0,3, `min_conf` 0,25, `seed` 42 | Lectura directa de `src/config.py` |
| Dimensiones del conjunto de datos | 102 clases, 1020 / 1020 / 6149 imágenes, 8189 en total, 10 imágenes de entrenamiento por clase, 40–258 imágenes por clase | Particiones oficiales del conjunto de datos [1], [9] y contrato del proyecto |
| Dimensiones del modelo | Entrada 224 × 224, 1280 características, 102 logits, matriz de confusión 102 × 102 | `src/config.py` y `src/model.py` |
| Expectativas del problema | ~92–97 % top-1 de referencia; 72,8 % y 55,1 % del trabajo original; línea base aleatoria ≈ 0,98 % top-1 | Literatura [1]; cálculo aritmético para la línea base |
| **Resultados propios (top-1, top-5, macro-F1, tiempos, mejor época)** | Sección 8 | **No existen todavía**: marcadores `_pendiente de la corrida en Colab_` |

**Declaración final.** Este informe no contiene ningún resultado experimental inventado,
estimado ni extrapolado. Todas las celdas de resultados están marcadas como pendientes y se
completan únicamente con la salida real de la ejecución del notebook en Google Colab.
