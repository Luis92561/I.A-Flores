# IA flores — Reconocimiento de especies de flores por imagen

Sistema de visión por computadora que, dada una foto, identifica la especie de flor
entre las **102 especies** del dataset **Oxford Flowers-102**, usando *transfer learning*
sobre un backbone preentrenado en ImageNet y devolviendo el *top-5* con probabilidades.

- Ejecutable de principio a fin en **Google Colab** con GPU T4 gratuita (no hace falta instalar nada local).
- Código modular en `src/` + notebook único + demo interactivo con Gradio.
- Splits **oficiales** del dataset (no se inventan particiones), métricas top-1 / top-5 / macro-F1.

> Las interfaces exactas, la propiedad de cada archivo, las firmas congeladas y las reglas
> de entorno están en **[CONTRATOS.md](CONTRATOS.md)** (documento autoritativo). El plan,
> las decisiones D1–D6 y los riesgos están en **[PLAN_TRABAJO.md](PLAN_TRABAJO.md)**.
> Este README no los duplica: describe *cómo usarlo*.

---

## 1. Resultados esperados

**Esta tabla está deliberadamente vacía.** El proyecto no incluye cifras medidas: hay que
ejecutar el notebook en Colab para obtenerlas. Los valores `_pendiente de tu ejecución_`
se sustituyen con la salida real de `python -m src.evaluate` (archivo `resultados/metrics.json`).

| Métrica | Split | Valor |
|---|---|---|
| Accuracy top-1 | test | _pendiente de tu ejecución_ |
| Accuracy top-5 | test | _pendiente de tu ejecución_ |
| Macro-F1 | test | _pendiente de tu ejecución_ |

**Referencia de la literatura (no es un resultado medido aquí).** Con *transfer learning*
sobre Flowers-102 y los splits oficiales, un backbone tipo EfficientNet-B0 suele situarse en
el rango aproximado de **~92–97 % top-1**. Ese rango es una expectativa de la literatura y de
los resultados publicados sobre este dataset, **no una medición de este repositorio**: el
número real depende de la ejecución, de la GPU, de las épocas y del azar de la semilla.
Trátalo como referencia para juzgar si tu corrida salió razonable, no como promesa.

---

## 2. Inicio rápido en Google Colab (recomendado)

1. **Abre el notebook.** Sube `notebooks/flores_colab.ipynb` a [Google Colab](https://colab.research.google.com/)
   (o ábrelo desde Drive) y conéctate. **No hace falta subir el proyecto**: el notebook clona
   automáticamente este repositorio (`https://github.com/Luis92561/I.A-Flores`) porque
   `MODO_CODIGO = "auto"` se resuelve a `"clone"` cuando no encuentra el proyecto local.
   Si prefieres no clonar, puedes subir `IA_flores_colab.zip` y cambiar `MODO_CODIGO` a `"zip"`.
2. **Activa la GPU T4.** Menú *Entorno de ejecución → Cambiar tipo de entorno de ejecución →
   Acelerador de hardware: GPU T4*. La primera celda del notebook verifica que `torch.cuda`
   esté disponible; si no lo está, no sigas.
3. **(Opcional pero recomendado) Monta Google Drive.** Así el checkpoint, las figuras y las
   métricas sobreviven a una desconexión. El notebook incluye una celda de montaje y una celda
   de "reanudar desde checkpoint".
4. **Ejecuta primero el modo debug.** *Entorno de ejecución → Ejecutar todo* con `debug=True`
   (subconjunto pequeño, 1–2 épocas, sin AMP). Tarda alrededor de un minuto y confirma que la
   descarga del dataset, los DataLoaders, el modelo y las rutas funcionan en Colab.
5. **Entrena en serio.** Vuelve a poner `debug=False` y ejecuta todo de nuevo. La corrida
   completa (25 épocas, batch 32 a 224 px) está pensada para unos **15–25 minutos** de GPU T4.
6. **Lanza la demo.** La última celda abre la interfaz Gradio: arrastra o sube una foto y
   obtén el *top-5* de especies con su probabilidad.

> **Regla de Colab (importante).** En Colab `torch` y `torchvision` ya vienen con CUDA.
> El notebook instala **solo** lo que falta (`gradio`, `scikit-learn`, `matplotlib`) y nunca
> reinstala torch por pip. Ver **[CONTRATOS.md](CONTRATOS.md)** §1 para el detalle.

---

## 3. Uso local (opcional)

Pensado para probar el pipeline en tu equipo (funciona en CPU; la GPU acelera).

```bash
# 1. Crear y activar el entorno virtual (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

```bash
# 3. Entrenar (escribe resultados/best.pt, history.json, curvas.png)
python -m src.train
python -m src.train --epochs 5 --debug --device cpu   # corrida rápida de humo

# 4. Evaluar sobre el split oficial de test
python -m src.evaluate --checkpoint resultados/best.pt

# 5. Predecir una imagen suelta (top-5)
python -m src.predict --image ruta\a\mi_foto.jpg --checkpoint resultados/best.pt --topk 5
```

Notas:

- El dataset se descarga automáticamente (~330 MB) en `data/` la primera vez.
- `python -m src.train` acepta `--epochs`, `--debug` y `--device`; `src.predict` acepta
  `--image`, `--checkpoint` y `--topk`. Para el resto de opciones, consulta `--help`.
- Las rutas por defecto (`data/`, `resultados/`) están definidas en `src/config.py` y
  están excluidas de git (ver `.gitignore`). No hay rutas absolutas de Windows en el código.
- La demo local se lanza con `python -m src.app` (equivale a lo que hace Colab en el paso 6).

---

## 4. Estructura del proyecto

```
IA flores/
├─ notebooks/
│  └─ flores_colab.ipynb      # notebook principal: todo el flujo, listo para "Ejecutar todo"
├─ src/
│  ├─ config.py               # dataclass Config: rutas, hiperparámetros, seed, modo debug
│  ├─ utils.py                # semillas, selección de dispositivo, JSON, conteo de parámetros
│  ├─ data.py                 # transforms + DataLoaders de los splits oficiales
│  ├─ model.py                # backbone preentrenado + cabeza nueva, checkpoints
│  ├─ train.py                # loop de entrenamiento: AMP, scheduler, early stopping, historial
│  ├─ evaluate.py             # top-1/top-5/macro-F1, matriz de confusión, curvas
│  ├─ predict.py              # inferencia sobre una imagen nueva (CLI + API)
│  └─ app.py                  # interfaz Gradio (upload + webcam, salida top-k)
├─ tests/
│  ├─ test_data.py            # carga de splits y forma de los lotes
│  ├─ test_model.py           # forward y ida/vuelta de checkpoints
│  ├─ test_train.py           # 2 épocas en modo debug
│  ├─ test_eval_predict.py    # métricas y predicción de una imagen
│  └─ test_smoke.py           # pipeline end-to-end con dataset sintético
├─ docs/
│  └─ VERIFICACION.md         # auditoría adversarial del notebook y del código
├─ data/                      # descargado automáticamente (no se versiona)
├─ resultados/                # best.pt, metrics.json, figuras, predicciones (no se versiona)
├─ requirements.txt           # dependencias del entorno local de verificación
├─ README.md                  # este archivo
├─ INFORME.md                 # informe académico con resultados y figuras
├─ CONTRATOS.md               # interfaces y propiedad de archivos (autoritativo)
└─ PLAN_TRABAJO.md            # plan, decisiones y riesgos
```

La correspondencia archivo ↔ subagente dueño y las firmas congeladas de cada módulo están en
**[CONTRATOS.md](CONTRATOS.md)** §2 y §3.

---

## 5. Cómo funciona (resumen)

1. **Datos.** Se usa `torchvision.datasets.Flowers102` con sus splits oficiales
   (**1020 train / 1020 val / 6149 test**, 102 clases). Entrenamiento con
   `RandomResizedCrop(224)`, volteo horizontal y *augmentación* de color; validación y test
   con `Resize(256)` + `CenterCrop(224)`. Normalización con las estadísticas de ImageNet.
2. **Transfer learning.** Se parte de un backbone **EfficientNet-B0 preentrenado en ImageNet**
   y se sustituye su clasificador por una **cabeza nueva** (`Dropout(0.3) → Linear(feat, 102)`).
   Las primeras épocas entrenan con el backbone congelado; después se descongela el backbone
   **completo** (el código no selecciona "últimos bloques": `set_backbone_trainable(model, True)`
   en `src/train.py`) usando *learning rate* diferencial (más bajo para el backbone que para la
   cabeza).
3. **Entrenamiento.** AdamW con *weight decay*, scheduler coseno con *warmup*, `label_smoothing=0.1`
   y **AMP** (*mixed precision*, automáticamente ignorado en CPU). Se guarda el checkpoint
   **solo cuando mejora** la accuracy de validación, y hay **early stopping** con paciencia
   configurable para no sobreajustar.
4. **Umbral de confianza.** En inferencia, si la probabilidad máxima del *top-1* queda por
   debajo de `min_conf` (0.25 por defecto en `src/config.py`), el sistema responde
   **"no reconocida / fuera del catálogo"** en lugar de forzar una de las 102 clases. Es la
   forma de admitir que la foto puede ser una flor que el modelo no conoce.
5. **Evaluación.** Top-1, top-5 y macro-F1 sobre test, más `classification_report`, matriz de
   confusión 102×102, ranking de confusiones más frecuentes y exportación de predicciones a CSV.

Los hiperparámetros reales viven en `src/config.py` (`Config`); los de arriba son los valores
por defecto de ese archivo, no una copia independiente.

---

## 6. Limitaciones honestas

- **Solo 102 especies.** El modelo no conoce ninguna flor fuera del catálogo de Flowers-102.
  Gracias al umbral de confianza puede decir "no reconocida", pero eso es un rechazo, no un
  reconocimiento: no identifica la especie desconocida.
- **10 imágenes de entrenamiento por clase.** El split oficial de train tiene 1020 imágenes
  para 102 clases. Es muy poco; el sobreajuste es el riesgo principal y por eso se usan
  augmentación fuerte, *label smoothing*, early stopping y evaluación en el test oficial.
- **Una flor por foto.** El sistema clasifica **la imagen completa**, no detecta ni segmenta
  varias flores en la misma foto. Con un ramo, la predicción corresponde a la flor dominante
  según el encuadre (o a ninguna, si la confianza cae bajo el umbral).
- **Sesgo del dataset.** Flowers-102 fue recolectado en su mayoría en el Reino Unido y con
  fotos "de catálogo" (fondo, iluminación y encuadre relativamente homogéneos). Fotos de otro
  país, con fondo cargado, poca luz, o de una subespecie/híbrido distinto pueden degradar
  mucho el resultado. Un buen *top-1* en test no garantiza buen desempeño en tus fotos.
- **Clases visualmente parecidas.** Varias de las 102 especies son casi idénticas a simple
  vista; los errores se concentran ahí. Para eso sirve el top-5 y el ranking de confusiones.

---

## 7. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `torch.cuda.is_available()` devuelve `False` y el entrenamiento es lentísimo | La GPU no está activada en Colab | *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*, y vuelve a ejecutar desde la primera celda |
| `CUDA out of memory` | Batch demasiado grande o memoria fragmentada por otras celdas | Baja `batch_size` de 32 a 16 (o 8) en la `Config`; reinicia el entorno para liberar memoria; considera entrenar a 160 px y evaluar/inferir a 224 px |
| Colab se desconecta y se pierde todo el progreso | Límite de sesión o inactividad | Monta Drive y entrena escribiendo el checkpoint en Drive (celda de montaje), y usa la celda de "reanudar desde checkpoint" al reconectar |
| La descarga del dataset (330 MB) tarda muchísimo o se corta | Red de Colab saturada | Reintenta (la descarga es reanudable por caché), guarda `data/` en Drive para reutilizarla, y usa `debug=True` para validar el resto del pipeline mientras tanto |
| Tras `pip install` la GPU deja de funcionar | Se reinstaló `torch`/`torchvision` desde pip y quedaron ruedas CPU | **No** ejecutes `pip install -r requirements.txt` en Colab: instala solo `gradio scikit-learn matplotlib`. Reinicia el entorno y deja el torch preinstalado de Colab |
| El notebook no encuentra `resultados/best.pt` | No se ha entrenado todavía, o el checkpoint quedó en otra ruta | Ejecuta primero el entrenamiento; en `src/app.py` y `src/predict.py` pasa `--checkpoint` con la ruta correcta |
| Los resultados cambian entre corridas | Semillas de librerías no fijadas por completo, o `num_workers` distinto | Fija la semilla con `set_seed(cfg.seed)` (ya se hace en el pipeline) y usa `num_workers=0` en modo debug |

---

## 8. Créditos y licencia

**Dataset.** Este proyecto usa **Oxford Flowers-102**. Cita obligatoria si reutilizas los datos
o los resultados:

> Nilsback, M-E. and Zisserman, A. *Automated Flower Classification over a Large Number of Classes.*
> Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP), 2008.

- Página del dataset: <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>

**Imágenes.** Las fotografías pertenecen a sus autores originales y se distribuyen para
**uso académico y de investigación**. No se redistribuyen en este repositorio: se descargan
automáticamente mediante `torchvision`. No las uses con fines comerciales.

**Código de este repositorio.** Escrito como trabajo de curso (sexto semestre, Inteligencia
Artificial). Los pesos preentrenados de ImageNet que usa el backbone conservan la licencia de
su proyecto de origen (torchvision / PyTorch). Añade aquí la licencia que decidas aplicar a tu
propio código si piensas publicarlo.

---

## 9. Estado del proyecto

**Estado: código completo y verificado; falta únicamente la corrida final en Colab.** Todos los
módulos de `src/` están implementados, probados con sus tests y ejercitados de punta a punta
(entrenar → evaluar → predecir → demo). Los resultados de la sección 1 los produce tu propia
ejecución del notebook; hasta entonces esas celdas siguen con el marcador
`_pendiente de tu ejecución_`.

| Componente | Estado | Verificación |
|---|---|---|
| `src/config.py`, `src/utils.py` | Listo (Fase 0, congelado) | Importables sin `torch`; usados por todos los módulos |
| `src/data.py` | Listo | 17/17 tests; splits reales 1020 / 1020 / 6149 (8189 imágenes) |
| `src/model.py` | Listo | 22/22 tests; pesos ImageNet descargan; ida/vuelta de checkpoint |
| `src/train.py` | Listo | Entrena en CPU y GPU; escribe `best.pt`, `history.json`, `config_usada.json` y `figuras/curvas.png` |
| `src/evaluate.py` | Listo | Escribe `metrics.json`, `figuras/confusion_matrix.png`, `figuras/top_confusions.png` y `predicciones/<split>_predicciones.csv` |
| `src/predict.py` | Listo | `predict_image` / `predict_batch` con caché de modelo y umbral `min_conf` |
| `src/app.py` | Listo | Demo Gradio (`gr.Blocks`) con subida de imagen, webcam y portapapeles; no crashea si falta el checkpoint |
| `notebooks/flores_colab.ipynb` | Listo | Version final: 26 celdas, 13 de código, ejecutada de principio a fin en modo debug sin errores |
| `tests/` | Listo | `test_data.py`, `test_model.py`, `test_train.py`, `test_eval_predict.py` |
| `empaquetar_colab.py` | Listo | Genera `IA_flores_colab.zip` con la lista blanca de archivos (sin `data/` ni `resultados/`) |
| Corrida completa (25 épocas en GPU T4) | **Pendiente de ti** | Es el paso que rellena la sección 1 y la sección 8 de `INFORME.md` |

**Cómo se verifica.** El pipeline se probó de punta a punta en CPU con el modo debug del
notebook (subconjunto de 4 imágenes por clase) y con los comandos de la sección 3. Además, el
notebook incluye un **modo debug** (`MODO_DEBUG = True`, activado por defecto) que escribe en
`resultados/debug/` y **nunca pisa** la corrida definitiva de `resultados/`.

**Qué falta y quién lo hace.** Lo único que queda es ejecutar el notebook en Colab con
`MODO_DEBUG = False` y copiar el bloque de resumen que imprime el **paso 8** en `INFORME.md`
(sección 8). Si un comando de este README fallara con `ModuleNotFoundError`, sería un problema
real del repositorio y conviene reportarlo: ya no queda ningún módulo por escribir.

