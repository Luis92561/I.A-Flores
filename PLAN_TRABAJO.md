# Plan de trabajo — Reconocimiento de flores por imagen (Google Colab)

**Estado:** propuesta, pendiente de aprobación. No se ha escrito código de producción todavía.

---

## 1. Objetivo

Un sistema de IA que, dada una foto, identifique la especie de flor, con tres propiedades:

1. **Entrenable en Google Colab** (GPU T4 gratis), sin instalar nada en tu PC.
2. **Reproducible y medible**: métricas top-1/top-5, matriz de confusión, análisis de errores.
3. **Usable**: demo interactivo donde arrastras una foto y recibes el *top-5* de especies con probabilidad.

## 2. Alcance

**Dentro**

- *Transfer learning* sobre **Oxford Flowers-102** (102 especies, splits oficiales: 1020 train / 1020 val / 6149 test).
- Pipeline completo: datos → augmentación → entrenamiento → evaluación → inferencia → demo.
- Notebook único ejecutable en Colab + módulos en `src/` (código limpio y reusable).
- Modo *debug* (subconjunto pequeño, 1–2 épocas) y modo completo.

**Fuera (por ahora)**

- Entrenar una CNN desde cero (peor resultado, mucho más cómputo).
- Detección/segmentación de varias flores en una misma foto.
- App publicada en la web o móvil, base de datos de usuarios.
- Dataset propio (posible Fase 2 opcional, ver §10).

## 3. Decisiones abiertas (con recomendación por defecto)

| # | Decisión | Recomendación | Alternativa |
|---|----------|---------------|-------------|
| D1 | Dataset | Oxford Flowers-102 (descarga automática) | Fotos propias etiquetadas (más lento) |
| D2 | Framework | PyTorch + torchvision | TensorFlow/Keras + `tensorflow_datasets` |
| D3 | Modelo | EfficientNet-B0 preentrenado (ImageNet) + fine-tuning | ResNet-50 / MobileNetV3 |
| D4 | Interfaz | Gradio embebido en el notebook | Solo CLI / CSV de predicciones |
| D5 | Documentos | README + informe en español | Bilingüe |
| D6 | Persistencia | Checkpoint y resultados en Google Drive | Solo sesión (se pierde) |

Si no dices lo contrario, procedo con la columna "Recomendación".

## 4. Arquitectura técnica

**Datos.** `torchvision.datasets.Flowers102` respeta los splits oficiales (no inventamos particiones → resultados comparables con la literatura). ~330 MB, descarga automática en Colab.

- Train: `RandomResizedCrop(224)`, flip horizontal, `RandAugment`/ColorJitter, Normalize (stats ImageNet).
- Val/Test: `Resize(256)` + `CenterCrop(224)` + Normalize.
- Refuerzos contra el sobreajuste (solo hay 10 imágenes de train por clase): `label_smoothing=0.1`, dropout, early stopping, y opcionalmente `WeightedRandomSampler`.

**Modelo.** Backbone preentrenado congelado + cabeza `Dropout(0.3) → Linear(feat, 102)`.
Etapa 2: descongelar los últimos bloques con *learning rate* diferencial (LR bajo para el backbone).

**Entrenamiento.** AdamW, scheduler coseno con *warmup*, mixed precision (AMP), early stopping por `val_acc`, semilla fija. Batch 32 @ 224 px → ~15–25 min en T4 para 20–30 épocas.

**Evaluación.** Top-1/top-5 accuracy, macro-F1, `classification_report`, matriz de confusión 102×102, ranking de confusiones más frecuentes, curvas de pérdida/accuracy y exportación de las imágenes mal clasificadas.

**Inferencia y demo.** `predict_image()` devuelve `[(especie, probabilidad)]` ordenado.
**Umbral de confianza**: si la probabilidad máxima es baja, responde "no reconocida / fuera del catálogo" en vez de forzar una de las 102 clases.

**Artefactos.** `best.pt` (pesos + `class_names` + transform), `history.json`, `metrics.json`, `confusion_matrix.png`, `curvas.png`, y export opcional TorchScript/ONNX.

## 5. Estructura del repo y contratos de interfaz

Estos contratos son lo que permite que varios subagentes trabajen **en paralelo sin pisarse**: cada uno implementa su archivo contra firmas fijas.

```
IA flores/
├─ notebooks/flores_colab.ipynb   # notebook principal (todo el flujo)
├─ src/
│  ├─ config.py     # dataclass Config: rutas, hiperparámetros, seed, modo debug
│  ├─ data.py       # datos y DataLoaders
│  ├─ model.py      # backbone + cabeza, congelar/descongelar, checkpoints
│  ├─ train.py      # loop de entrenamiento, AMP, early stopping, historial
│  ├─ evaluate.py   # métricas, matriz de confusión, curvas
│  ├─ predict.py    # inferencia sobre una imagen nueva (CLI + API)
│  └─ app.py        # interfaz Gradio
├─ tests/test_smoke.py            # pipeline end-to-end con dataset sintético
├─ requirements.txt
├─ README.md
├─ INFORME.md
└─ PLAN_TRABAJO.md
```

Firmas congeladas:

```python
# config.py
@dataclass
class Config:
    data_root: str; out_dir: str; backbone: str; num_classes: int
    epochs: int; batch_size: int; lr: float; img_size: int
    seed: int; num_workers: int; debug: bool        # debug = subset pequeño

# data.py
def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]
def build_transforms(cfg, train: bool) -> Callable

# model.py
def build_model(num_classes: int, backbone: str = "efficientnet_b0",
                pretrained: bool = True) -> nn.Module
def set_backbone_trainable(model, trainable: bool) -> None
def save_checkpoint(path, model, class_names, cfg) -> None
def load_checkpoint(path, device) -> tuple[nn.Module, list[str], dict]

# train.py
def train(cfg) -> dict          # escribe best.pt, history.json, curvas.png

# evaluate.py
def evaluate(checkpoint_path, cfg) -> dict   # metrics.json, confusion_matrix.png

# predict.py
def predict_image(path, checkpoint_path, topk: int = 5,
                  min_conf: float = 0.25, device: str = "auto") -> list[tuple[str, float]]

# app.py
def build_app(checkpoint_path: str) -> "gr.Interface"
```

## 6. Fases y criterios de aceptación

| Fase | Trabajo | Criterio de "hecho" |
|------|---------|---------------------|
| F0 | Contratos, `config.py`, esqueleto de repo | Firmas y carpetas creadas; nada más puede empezar sin esto |
| F1 | `src/data.py` + `requirements.txt` + shell del notebook | Carga los 3 splits y pasa un lote; funciona en CPU y GPU |
| F2 | `src/model.py` | `build_model(102)` corre un forward (1,3,224,224) → (1,102) |
| F3 | `src/train.py` | 2 épocas en subconjunto sin errores; guarda checkpoint e historial |
| F4 | `src/evaluate.py` + `src/predict.py` | Recarga checkpoint, calcula métricas y predice una imagen de test |
| F5 | `src/app.py` (Gradio) | Sube una imagen → top-5 con barras de probabilidad |
| F6 | Notebook final integrado | Clonar → "Ejecutar todo" → resultados, sin celdas rotas |
| F7 | README + INFORME | Instrucciones de uso y plantilla de resultados/documentación |
| F8 | QA: smoke test local + revisión adversarial + checklist Colab | Informe de verificación con hallazgos resueltos |
| F9 | **(tú)** ejecución real en Colab y entrega de métricas | Notebook con salidas reales guardadas |

## 7. Reparto en subagentes (lo que orquesto yo)

Los subagentes **no comparten mi contexto**: a cada uno le paso el contrato de §5, sus rutas, sus criterios de aceptación y la lista de archivos que tiene prohibido tocar.

| ID | Rol | Entregable | Depende de |
|----|-----|-----------|-----------|
| SA-00 | Arquitecto | Contratos, `config.py`, esqueleto, `PLAN_TRABAJO.md` | — |
| **Oleada 1 (paralela)** ||||
| SA-01 | Ingeniero de datos | `src/data.py` + test con dataset sintético | SA-00 |
| SA-02 | Ingeniero de modelo | `src/model.py` + test de forward/checkpoints | SA-00 |
| SA-03 | Ingeniero de entorno | `requirements.txt` + notebook shell (celdas y orden) | SA-00 |
| SA-04 | Redactor técnico | `README.md` (uso, requisitos, ejecución en Colab) | SA-00 |
| **Oleada 2 (paralela)** ||||
| SA-05 | Entrenamiento | `src/train.py`, AMP, early stopping, historial | SA-01, SA-02 |
| SA-06 | Evaluación | `src/evaluate.py` + `src/predict.py` | SA-02 |
| SA-07 | Demo | `src/app.py` (Gradio) + umbral de confianza | SA-06 (contrato) |
| SA-08 | Informe | `INFORME.md` con plantilla de resultados y figuras | SA-04 |
| **Oleada 3** ||||
| SA-09 | Integrador | Notebook final con código embebido + smoke test end-to-end | SA-03, SA-05..07 |
| SA-10 | Verificador adversarial | Auditoría: imports, rutas absolutas, fallback CPU, fugas de datos, semillas, tiempo de celda | SA-09 |
| SA-11 | *(opcional)* Extensión | Fine-tuning con **tus fotos** y manejo de clases fuera del catálogo | SA-09 |

Regla de oro del reparto: **una oleada no empieza hasta que la anterior cierra su contrato**, porque las firmas de §5 son el único acoplamiento.

## 8. Paralelización y ruta crítica

```
F0 contratos
   ├─ SA-01 datos ─┐
   ├─ SA-02 modelo ┴→ SA-05 entrenamiento ┐
   ├─ SA-03 notebook ─────────────────────┼→ SA-09 integrador → SA-10 verificador
   └─ SA-04 README → SA-08 informe ───────┘
        SA-02 → SA-06 evaluación → SA-07 demo ─┘
```

- Trabajo secuencial equivalente: **~2 h** de agente.
- Con 4 subagentes en paralelo por oleada: **~40–50 min** de reloj (≈60 % menos).
- Después, la ejecución real en Colab (tuya): ~25–35 min de GPU T4, de los cuales solo ~2 min son de smoke test.

**Yo verifico antes de dártelo**: creo un entorno virtual con Python 3.12 (ya está instalado en tu equipo) e instalo PyTorch para correr un smoke test en tu **RTX 3060 Ti** con un dataset sintético de 4 clases. Así el notebook no llega a Colab "a ciegas".

## 9. Verificación y control de calidad

1. **Smoke test local** (aquí, en tu GPU): 4 clases sintéticas, 8 imágenes, 2 épocas → todo el pipeline de punta a punta en <2 min.
2. **Modo debug** en el notebook (`debug=True`, 1 época): valida la instalación en Colab en ~1 min antes de gastar GPU.
3. **Revisión adversarial** por un subagente independiente, con checklist explícito: imports faltantes, rutas absolutas de tu PC, ausencia de fallback a CPU, fuga de datos entre train y test, semillas no fijadas, uso de `num_workers` incompatible con Colab.
4. **Checklist de Colab**: `Entorno de ejecución → GPU T4`, montaje de Drive, orden de celdas, tiempo por celda y uso de memoria.

## 10. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| Descarga lenta o fallida del dataset en Colab | Alto | Caché en Drive + reintentos + `debug` con subconjunto |
| Solo 10 imágenes de train por clase (sobreajuste) | Alto | Augmentación fuerte, label smoothing, early stopping, evaluar en el split oficial de test |
| Desconexión de Colab (límite de sesión) | Medio | Checkpoint cada época en Drive y celda de "reanudar" |
| Tu foto es una flor que **no** está entre las 102 especies | Medio | Umbral de confianza → "no reconocida" + Fase opcional de fine-tuning con tus fotos |
| Dependencias incompatibles (Colab cambia de versión) | Medio | `requirements.txt` con versiones mínimas y verificación en la primera celda |
| Python 3.14 local no soporta PyTorch | Bajo | Usar el Python 3.12 ya instalado en un venv para las pruebas |
| Licencias de imágenes | Bajo | Flowers-102 es de uso académico; documentar la cita en el README |

## 11. Lo que necesito de ti

1. Confirmar (o cambiar) las decisiones **D1–D6** de §3.
2. Ejecutar el notebook en Colab con GPU (yo no puedo acceder a Colab desde aquí).
3. Decidir si quieres la **Fase opcional con tus propias fotos** (SA-11): si sí, dime cuántas fotos y de cuántas especies dispones.
