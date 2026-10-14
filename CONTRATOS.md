# CONTRATOS DE INTERFAZ — proyecto "IA flores"

> Documento **autoritativo** de la Fase 0 (propietario: orquestador / SA-00).
> Ningún subagente modifica este archivo ni `src/config.py` ni `src/utils.py`.

---

## 0. Reglas para subagentes

1. **Un solo dueño por archivo.** Escribe únicamente los archivos que te asignaron.
   Nunca edites archivos de otro subagente ni los de Fase 0.
2. **No instales dependencias** ni descargues el dataset completo (330 MB).
   El orquestador ya preparó el entorno.
3. **Respeta las firmas de la sección 3 al pie de la letra.** Si crees que una firma
   está mal, impleméntala igual y repórtalo en tu informe final.
4. **Todo debe correr en CPU y en GPU.** Nada de `.cuda()` duro: usa
   `src.utils.resolve_device(cfg.device)`.
5. **Sin rutas absolutas de Windows** en el código (nada de rutas que empiecen por la letra
   de unidad del disco local del autor).
   Rutas relativas o derivadas de `Config`.
6. **Docstrings y comentarios en español**, nombres de código en inglés.
7. **Verifica antes de reportar.** Ejecuta tu propio test si el entorno ya está listo:
   `.venv\Scripts\python.exe -m pytest tests\test_<tu_modulo>.py -q`
   (si `import torch` falla, espera un momento y reintenta una vez; si sigue fallando,
   entrega el código y reporta "no ejecutado por entorno").
8. Tu informe final: archivos creados, comandos ejecutados, resultado exacto
   (salida literal o mensaje de error), y cualquier desviación del contrato.

---

## 1. Entorno

| Cosa | Valor |
|---|---|
| Python local de pruebas | 3.12 (el del sistema; las pruebas del proyecto se ejecutan con el venv `.venv`) |
| Venv del proyecto | `.venv\Scripts\python.exe` — **ya provisionado y verificado** |
| Ejecutar tests | `.venv\Scripts\python.exe -m pytest tests -q` |
| GPU local | RTX 3060 Ti (solo informativo; las pruebas usan CPU) |
| Destino real | Google Colab (GPU T4) |

**Instalado y verificado en `.venv`:** torch 2.14.1+cpu, torchvision 0.29.1+cpu,
numpy 2.5.2, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2,
gradio 6.30.0, pytest 9.1.1, nbformat 5.11.1, tqdm 4.70.1.

### ⚠️ Peculiaridades de esta maquina (necesarias para que tus tests corran)

1. **CAUSA RAIZ de casi todos los `PermissionError` de esta maquina (diagnosticada y
   reproducida):** bajo el sandbox, un directorio creado con modo `0o700` queda
   inaccesible (no se puede listar ni escribir dentro → `WinError 5`).
   `tempfile.mkdtemp()`, `pytest` (`tmp_path`, `cacheprovider`) y `tarfile` al extraer
   carpetas usan `os.mkdir(..., 0o700)`, y por eso fallan todas.
   - **No es un bug de pip** (correccion del diagnostico inicial): pip usa `mkdtemp`,
     asi que arrastra el mismo problema. `uv` no lo usa, y por eso si funciona.
   - Workaround de pip: usa `uv` para instalar:
     ```powershell
     .\.piptmp\tools\uv.exe pip install --python .\.venv\Scripts\python.exe <paquetes>
     ```
   - En `tests/`, `tests/conftest.py` ya sustituye `tmp_path`/`tmp_path_factory`.
   - **En Colab/Linux nada de esto aplica.**
2. **`%TEMP%` esta bloqueado por el sandbox.** Prefija tus comandos con:
   ```powershell
   $env:TMP = (Resolve-Path .pytmp).Path; $env:TEMP = $env:TMP
   $env:MPLCONFIGDIR = (Resolve-Path .pytmp).Path   # matplotlib necesita cache escribible
   ```
3. **No crees directorios con `os.mkdir(path, 0o700)` ni uses `tempfile.mkdtemp()`
   / `tarfile` sobre carpetas nuevas** (ver punto 1). En los tests pide la fixture
   `tmp_path`: `tests/conftest.py` ya la reemplaza. Crea directorios con `Path.mkdir()`
   sin `mode`.
4. **`curl.exe` no funciona**; la red sí funciona desde Python (`urllib`), que es lo
   que usan torchvision y pip.
5. En tests pesados usa `torch.set_num_threads(2)` para no saturar la maquina
   mientras otros subagentes trabajan.

**Regla de Colab:** en Colab `torch` y `torchvision` ya vienen preinstalados con CUDA.
Instalarlos por pip puede reemplazarlos por ruedas CPU y romper la GPU. El notebook
debe instalar **solo** lo que falte (`gradio`, `scikit-learn`, `matplotlib`) y
comprobar antes si torch ya está disponible.

---

## 2. Estructura y propiedad de archivos

```
PLAN_TRABAJO.md          SA-00 (hecho)
CONTRATOS.md             SA-00 (hecho)
requirements.txt         SA-00 (hecho)
src/__init__.py          SA-00 (hecho)
src/config.py            SA-00 (hecho)  <- NO TOCAR
src/utils.py             SA-00 (hecho)  <- NO TOCAR
src/data.py              SA-01
tests/test_data.py       SA-01
src/model.py             SA-02
tests/test_model.py      SA-02
notebooks/flores_colab.ipynb   SA-03 (esqueleto) -> SA-09 (version final integrada)
README.md                SA-04
src/train.py             SA-05
tests/test_train.py      SA-05
src/evaluate.py          SA-06
src/predict.py           SA-06
tests/test_eval_predict.py SA-06
src/app.py               SA-07
INFORME.md               SA-08
tests/test_smoke.py      SA-09
docs/VERIFICACION.md     SA-10
resultados/**            generado, no se versiona
data/**                  generado, no se versiona
```

---

## 3. Firmas congeladas

### `src/config.py` (ya implementado)

```python
@dataclass
class Config:
    data_root, out_dir, num_classes, img_size,
    backbone, dropout, pretrained,
    epochs, batch_size, lr, backbone_lr, weight_decay, label_smoothing,
    warmup_epochs, freeze_epochs, early_stop_patience, amp,
    topk, min_conf, seed, num_workers, device, debug, debug_per_class
    # propiedades: checkpoint_path, history_path, metrics_path, config_path
    # metodos: figures_dir(), predictions_dir(), ensure_dirs(), to_dict()
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
FLOWERS102_CLASSES = 102
```

### `src/utils.py` (ya implementado)

```python
set_seed(seed: int = 42) -> None
resolve_device(spec: str = "auto") -> torch.device
describe_device(device) -> str
ensure_dir(path) -> Path
save_json(path, obj) -> None
load_json(path) -> Any
count_params(model) -> int
```

### `src/data.py` — SA-01

```python
build_transforms(cfg: Config, train: bool) -> Callable
build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]
get_class_names(cfg: Config) -> list[str]      # 102 nombres legibles, en orden de etiqueta
```

- `build_dataloaders` devuelve `(train_dl, val_dl, test_dl, class_names)`.
- `class_names[i]` debe corresponder a la etiqueta entera `i` (0..101) que produce el dataset.
- Usa `torchvision.datasets.Flowers102` con `split="train" | "val" | "test"`.
  Tamaños esperados: **1020 / 1020 / 6149**.
- Si `cfg.debug` es `True`, devuelve subconjuntos de `cfg.debug_per_class` imágenes por clase
  (usa `torch.utils.data.Subset`) manteniendo el orden de las etiquetas.
- Nombres de clase: torchvision **no siempre** expone el nombre de la especie.
  Estrategia: usar los nombres propios del dataset si existen y no son genéricos;
  si no, descargar `cat_to_name.json` de Oxford y **cachearlo** en
  `<data_root>/cat_to_name.json`. Si tampoco hay internet, devolver
  `["especie_001", ..., "especie_102"]` sin fallar.
  ⚠️ **Pitfall conocido:** las etiquetas originales son 1-based (1..102) y las claves de
  `cat_to_name.json` son `"1".."102"`; torchvision resta 1. Mapea `class_names[i] = cat[str(i+1)]`.
- `num_workers`: `cfg.num_workers` (ya vale 0 en modo debug).
- Los DataLoaders de val/test **no** deben barajar (`shuffle=False`) y `drop_last=False`.
- Acepta `pin_memory=True` solo cuando hay CUDA.

### `src/model.py` — SA-02

```python
class FlowerClassifier(nn.Module):      # atributos OBLIGATORIOS: .backbone y .head
    def forward(self, x) -> Tensor      # (B,3,H,W) -> (B, num_classes) logits

build_model(num_classes: int = 102, backbone: str = "efficientnet_b0",
            pretrained: bool = True, dropout: float = 0.3) -> FlowerClassifier
set_backbone_trainable(model, trainable: bool) -> None
build_optimizer(model, cfg: Config) -> torch.optim.Optimizer   # grupos con LR diferencial
save_checkpoint(path, model, class_names: list[str], cfg: Config, **extra) -> None
load_checkpoint(path, device="cpu") -> tuple[FlowerClassifier, list[str], dict]
```

- Backbones soportados como mínimo: `efficientnet_b0` (defecto), `resnet50`, `mobilenet_v3_large`.
  Cualquier otro nombre debe lanzar `ValueError` con la lista de soportados.
- `pretrained=True` debe funcionar **sin internet**: si la descarga de pesos falla,
  cae a `pretrained=False` y lo avisa por `warnings.warn` (Colab a veces bloquea descargas).
- `load_checkpoint` devuelve el modelo ya en `.eval()` y en el `device` pedido.
- **Formato de checkpoint (contrato entre SA-02, SA-05 y SA-06):**

```python
torch.save({
    "model_state": model.state_dict(),
    "class_names": [...],       # 102 str
    "backbone": "efficientnet_b0",
    "num_classes": 102,
    "img_size": 224,
    "config": cfg.to_dict(),
    "epoch": int,
    "val_acc": float,
}, path)
```

### `src/train.py` — SA-05

```python
train(cfg: Config) -> dict          # entrena y devuelve el historial
# CLI: python -m src.train --epochs 5 --debug --device cpu
```

- `freeze_epochs`: primeras épocas con backbone congelado (`set_backbone_trainable(m, False)`).
- AMP (`torch.amp.autocast` + `GradScaler`) **solo** si el device es CUDA y `cfg.amp`.
- Scheduler coseno con `warmup_epochs` de warmup, sobre los dos grupos de parámetros.
- Guarda `cfg.checkpoint_path` **solo** cuando mejora `val_acc`; además `cfg.history_path`,
  `cfg.config_path` y `curvas.png` en `cfg.figures_dir()`.
- Historial (JSON serializable, listas por época):
  `{"epochs": [...], "train_loss": [...], "train_acc": [...], "val_loss": [...],
    "val_acc": [...], "val_top5": [...], "lr": [...], "best_val_acc": float,
    "best_epoch": int, "epochs_run": int, "device": str, "num_params": int, "segundos": float}`
- Early stopping con `cfg.early_stop_patience`; imprime una línea por época con `tqdm`.

### `src/evaluate.py` — SA-06

```python
evaluate(checkpoint_path: str | Path, cfg: Config | None = None, split: str = "test") -> dict
```

- Calcula top-1, top-5, macro-F1, `classification_report` y matriz de confusión 102×102.
- Escribe `metrics.json` (usa `cfg.metrics_path`, o `resultados/metrics.json` si `cfg is None`),
  `confusion_matrix.png`, `top_confusions.png` y
  `predicciones/<split>_predicciones.csv` con columnas
  `ruta, etiqueta_real, prediccion, prob, correcto`.
- `metrics.json` debe incluir: `split, num_images, top1, top5, macro_f1, per_class,
  top_confusions ([[real, pred, n], ...]), ruta_confusion, segundos`.
- Carga el modelo con `load_checkpoint` y reconstruye los transforms con `cfg.img_size`
  del propio checkpoint (no asumas 224 fijo).

### `src/predict.py` — SA-06

```python
predict_image(image_path, checkpoint_path, topk: int = 5,
              min_conf: float = 0.25, device: str = "auto") -> list[tuple[str, float]]
predict_batch(image_paths, checkpoint_path, topk: int = 5, device: str = "auto") -> list[list[tuple[str, float]]]
# CLI: python -m src.predict --image ruta.jpg --checkpoint resultados/best.pt --topk 5
```

- Acepta ruta `str`/`Path`, `PIL.Image`, `numpy.ndarray` o la tupla de Gradio.
- Cachea el modelo cargado (por ejemplo con `functools.lru_cache`) para que la demo no
  recargue 20 MB en cada clic.
- Si `max(prob) < min_conf` devuelve `[("no reconocida / fuera del catálogo", max_prob)]`.
- Si el checkpoint no existe, lanza `FileNotFoundError` con un mensaje que explique
  que hay que entrenar primero.

### `src/app.py` — SA-07

```python
build_app(checkpoint_path: str = "resultados/best.pt", topk: int = 5) -> "gr.Interface"
main(share: bool = False, checkpoint_path: str = "resultados/best.pt") -> None
```

- Entrada: imagen (upload **y** webcam). Salida: etiquetas con probabilidad (`gr.Label`)
  + un `gr.Markdown` con el top-k en tabla.
- Si falta el checkpoint, la app **no debe crashear**: muestra un aviso claro
  ("entrena primero: `python -m src.train`").
- `share=False` por defecto (en Colab `share=True` da enlace público).

### 3.b CLI congeladas (todas ejecutables con `python -m ...` desde la raíz)

```bash
python -m src.train    [--epochs N] [--batch-size N] [--backbone B] [--data-root D]
                       [--out-dir O] [--device auto|cuda|cpu] [--debug] [--seed N]
python -m src.evaluate [--checkpoint resultados/best.pt] [--split test|val]
                       [--out-dir O] [--device auto|cuda|cpu] [--debug] [--num-workers N]
python -m src.predict  --image RUTA [--checkpoint resultados/best.pt] [--topk 5]
                       [--min-conf 0.25] [--device auto|cuda|cpu]
python -m src.app      [--checkpoint resultados/best.pt] [--topk 5] [--share]
```

- Sin argumentos, `evaluate` y `app` usan `resultados/best.pt`; `train` usa los defaults de `Config`.
- Los cuatro aceptan `--help` y **no deben crashear** si falta el checkpoint: mensaje claro
  ("entrena primero: `python -m src.train`").
- `--num-workers` es una **extension autorizada** (2026-xx): el default de `Config` sigue
  siendo 2 (correcto en Colab/Linux), pero en Windows los procesos hijo del DataLoader
  pueden fallar; el flag permite bajarlo a 0. Ademas, `evaluate` debe **degradar solo**:
  si iterar con `num_workers > 0` lanza un error de permisos/multiprocessing, reintenta
  una vez con 0 avisando, en vez de morir con `[error] PermissionError`.

---

## 4. Criterio de aceptación por módulo

| Módulo | Prueba mínima que debe pasar |
|---|---|
| `data.py` | `build_dataloaders(Config(debug=True))` → 3 loaders, 102 `class_names`, 1 lote con forma `(B,3,224,224)` y etiquetas en `[0,102)` |
| `model.py` | `build_model(102)` con `pretrained=False` → forward `(1,3,224,224)` → `(1,102)`; `save`+`load_checkpoint` reproduce las mismas salidas |
| `train.py` | `train(Config(debug=True, epochs=2, device="cpu"))` termina y escribe `best.pt`, `history.json`, `curvas.png` |
| `evaluate.py` | `evaluate(checkpoint, Config(debug=True), split="test")` escribe `metrics.json` con `top1` y `top5` numéricos |
| `predict.py` | `predict_image(imagen_sintetica, checkpoint)` → 5 pares `(str, float)` con probabilidades decrecientes |
| `app.py` | `build_app(checkpoint)` devuelve un objeto Gradio sin excepciones (no hace falta lanzar servidor) |

---

## 5. Decisiones del orquestador (resueltas)

| Duda planteada por | Decisión |
|---|---|
| SA-04: faltaba CLI de `evaluate` / `app` | Añadida en §3.b. SA-06 y SA-07 **deben** implementarla tal cual; el README ya la documenta así. |
| SA-03: `REPO_GIT` placeholder | Se queda `MODO_CODIGO = "zip"` como valor por defecto: la entrega se hace como ZIP local, sin depender de GitHub. SA-09 lo deja documentado y con la vía `clone` como opción comentada. |
| SA-03: ¿unificar `resultados/debug`? | **No.** El modo debug escribe en `resultados/debug/` y la corrida completa en `resultados/`. Así el smoke test nunca pisa el `best.pt` bueno. |
| SA-03: ¿añadir `resume` a `train()`? | **No** en esta versión. La reanudación documentada es: recuperar `resultados/` de Drive y saltar a evaluación/demo. Se anota como mejora futura. |
| SA-03: cachear `data/` en Drive | SA-09 añade una celda **opcional** que copia `data/` a Drive si existe `DRIVE_DIR` (evita re-descargar 330 MB en cada sesión de Colab). |
| SA-03: `nbformat` | Ya está en `requirements.txt` (SA-00) y se instaló en el venv de verificación. |
| SA-03: acentos | Todo el proyecto se escribe **sin acentos** en el código y en los notebooks (evita problemas de codificación en Windows/Colab). Los documentos `.md` sí pueden llevarlos. |
| SA-04: frase "25 épocas, batch 32, ~15-25 min" | Correcta: coincide con los defaults reales de `src/config.py`. SA-05 no debe cambiar esos defaults sin avisar. |
| SA-04: sección §9 "Estado del proyecto" caduca | SA-09 la actualiza al estado final (o la elimina) al integrar el notebook. |

Los subagentes de la Oleada 2 reciben estas decisiones en su tarea; este documento es la
referencia si hay conflicto.
