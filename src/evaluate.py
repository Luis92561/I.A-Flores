"""Evaluacion de un checkpoint sobre un split de Oxford Flowers-102.

CONTRATO (seccion 3 del contrato interno, propietario SA-06):

    evaluate(checkpoint_path: str | Path, cfg: Config | None = None,
             split: str = "test") -> dict
    # CLI: python -m src.evaluate [--checkpoint resultados/best.pt] [--split test|val]
    #                              [--out-dir O] [--device auto|cuda|cpu] [--debug]
    #                              [--num-workers N]   (extension autorizada por SA-00)

Que escribe (todo dentro de ``cfg.out_dir``):

  * ``metrics.json``            metricas completas (ver :func:`evaluate`);
  * ``figuras/confusion_matrix.png``  matriz de confusion NxN (102x102 con el catalogo real);
  * ``figuras/top_confusions.png``    barras con las confusiones mas frecuentes;
  * ``predicciones/<split>_predicciones.csv``  una fila por imagen evaluada.

Notas de diseno:
  - El tamano de imagen se toma SIEMPRE del checkpoint (``meta["img_size"]``), nunca
    se asume 224: si el usuario pasa un ``cfg`` con otro ``img_size`` se respeta el
    del checkpoint para construir los transforms de val/test.
  - El numero de clases tambien se toma del checkpoint (``meta["num_classes"]``); los
    nombres de clase se recortan o rellenan para que ``class_names[i]`` sea la clase
    ``i`` aunque el checkpoint venga de una corrida con otro ``num_classes``.
  - Todo funciona en CPU y en GPU: el device se resuelve con
    :func:`src.utils.resolve_device` a partir de ``cfg.device`` (nada de ``.cuda()``).
  - Matplotlib se usa con el backend ``Agg`` (sin ventanas ni display).
  - No hay rutas absolutas de Windows: todas las rutas derivan de ``Config`` o del
    argumento ``checkpoint_path``.
"""

from __future__ import annotations

import csv
import time
import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:  # import relativo (paquete `src`) con respaldo absoluto (cwd en sys.path)
    from .config import IMAGE_SIZE, Config
    from .data import build_dataloaders
    from .model import load_checkpoint
    from .utils import resolve_device, save_json
except ImportError:  # pragma: no cover - depende de como se importe el modulo
    from src.config import IMAGE_SIZE, Config
    from src.data import build_dataloaders
    from src.model import load_checkpoint
    from src.utils import resolve_device, save_json

__all__ = [
    "evaluate",
    "main",
    "SPLITS_VALIDOS",
    "MENSAJE_ENTRENA",
    "MAX_CONFUSIONES",
    "CABECERA_CSV",
]

#: Splits oficiales aceptados (los mismos de ``src.data``).
SPLITS_VALIDOS: tuple[str, str, str] = ("train", "val", "test")
#: Mensaje unico para "todavia no hay modelo entrenado" (texto del contrato interno seccion 3.b).
MENSAJE_ENTRENA = "entrena primero: `python -m src.train`"
#: Maximo de confusiones guardadas/graficadas (contrato: 20).
MAX_CONFUSIONES: int = 20
#: Encabezado exacto del CSV de predicciones (en espanol, congelado por contrato).
CABECERA_CSV: tuple[str, ...] = ("ruta", "etiqueta_real", "prediccion", "prob", "correcto")

#: Senales de que el fallo viene de los procesos hijo del DataLoader (no del modelo).
#: En el sandbox de Windows de este entorno (y en algunos Windows corporativos) los
#: workers de ``multiprocessing`` no pueden abrir *named pipes*, que es el IPC que usa
#: PyTorch en Windows, y la iteracion revienta con ``PermissionError [WinError 5]``.
_SENALES_FALLO_WORKERS: tuple[str, ...] = (
    "permissionerror",
    "winerror 5",
    "acceso denegado",
    "access is denied",
    "errno 13",
    "errno 22",
    "named pipe",
    "multiprocessing",
    "dataloader worker",
    "worker",
    "spawn",
    "sharing strategy",
    "broken pipe",
    "too many open files",
)


def _es_fallo_de_workers(exc: BaseException) -> bool:
    """True si la excepcion huele a problema de los workers del DataLoader."""
    if isinstance(exc, PermissionError):
        return True
    texto = f"{type(exc).__name__}: {exc}".lower()
    return any(senal in texto for senal in _SENALES_FALLO_WORKERS)


# --------------------------------------------------------------------------- #
# Helpers de configuracion
# --------------------------------------------------------------------------- #
def _config_desde_meta(meta: dict) -> Config:
    """Reconstruye ``Config(**meta["config"])`` ignorando claves desconocidas.

    Un checkpoint escrito por una version futura podria traer campos que este
    ``Config`` no conoce: se descartan en vez de reventar.
    """
    valores = dict(meta.get("config") or {})
    validos = {campo.name for campo in fields(Config)}
    limpios = {clave: valor for clave, valor in valores.items() if clave in validos}
    cfg = Config(**limpios)
    if not str(cfg.out_dir).strip():
        cfg = replace(cfg, out_dir="resultados")
    return cfg


def _cfg_con_img_size(cfg: Config, img_size: int) -> Config:
    """Copia de ``cfg`` con el ``img_size`` del checkpoint (no asumir 224)."""
    img_size = int(img_size)
    if int(getattr(cfg, "img_size", IMAGE_SIZE)) == img_size:
        return cfg
    return replace(cfg, img_size=img_size)


def _ajustar_nombres(class_names: Sequence[str], n_clases: int) -> list[str]:
    """Lista de exactamente ``n_clases`` nombres (recorta o rellena si hace falta)."""
    nombres = [str(nombre) for nombre in class_names]
    if len(nombres) < n_clases:
        nombres += [f"clase_{i + 1:03d}" for i in range(len(nombres), n_clases)]
    return nombres[:n_clases]


def _dataloader_de_split(cfg: Config, split: str):
    """DataLoader del split pedido reutilizando :func:`src.data.build_dataloaders`.

    ``build_dataloaders`` devuelve ``(train_dl, val_dl, test_dl, class_names)``; aqui
    solo se reutiliza el loader del split solicitado (los transforms ya se construyen
    con el ``img_size`` corregido del checkpoint).
    """
    resultado = build_dataloaders(cfg)
    if not isinstance(resultado, (tuple, list)) or len(resultado) < 3:
        raise RuntimeError(
            "build_dataloaders no devolvio (train_dl, val_dl, test_dl, class_names); "
            f"se recibio {type(resultado).__name__}."
        )
    return resultado[SPLITS_VALIDOS.index(split)]


def _inferir_robusto(
    modelo,
    cfg: Config,
    split: str,
    dataloader,
    device,
    n_clases: int,
    nombres: Sequence[str],
):
    """Itera el split reintentando UNA vez con ``num_workers=0`` si fallan los workers.

    En Windows (y en particular en el sandbox de este entorno) los procesos hijo del
    ``DataLoader`` no siempre pueden abrir *named pipes*; PyTorch los usa como IPC y la
    iteracion muere con ``PermissionError [WinError 5]`` (o con un error de
    ``multiprocessing``/``spawn``). Como el modelo y las metricas no tienen nada que ver
    con eso, se avisa y se repite la inferencia con ``num_workers=0`` (que siempre
    funciona) en lugar de abortar la evaluacion.

    Si el ``dataloader`` fue inyectado por el llamador (uso de tests) no se reconstruye:
    un fallo suyo se propaga tal cual.
    """

    def _intento(cfg_actual: Config):
        cargador = (
            dataloader if dataloader is not None else _dataloader_de_split(cfg_actual, split)
        )
        return _inferir(modelo, cargador, device, n_clases, nombres)

    try:
        return _intento(cfg)
    except Exception as exc:  # noqa: BLE001 - se decide con _es_fallo_de_workers
        workers = int(getattr(cfg, "num_workers", 0) or 0)
        if dataloader is not None or workers <= 0 or not _es_fallo_de_workers(exc):
            raise
        aviso = (
            f"El DataLoader con num_workers={workers} fallo ({type(exc).__name__}: {exc}). "
            "Se reintenta una sola vez con num_workers=0; esto es habitual en Windows "
            "(los workers de multiprocessing necesitan named pipes). Para evitar el "
            "reintento usa `--num-workers 0`."
        )
        warnings.warn(aviso, RuntimeWarning, stacklevel=2)
        print(f"[evaluate] aviso: {aviso}")
        return _intento(replace(cfg, num_workers=0))


# --------------------------------------------------------------------------- #
# Rutas de las imagenes del dataset
# --------------------------------------------------------------------------- #
def _rutas_dataset(dataset) -> list[str] | None:
    """Rutas de las imagenes de ``dataset`` en el mismo orden que ``dataset[i]``.

    Soporta ``Subset`` (uno o varios niveles anidados, como los del modo debug de
    ``src.data``) y los atributos que exponen los datasets de torchvision
    (``_image_files``, ``samples``, ``imgs``, ...). Devuelve ``None`` si no se puede
    determinar (por ejemplo en un dataset sintetico de tests): el llamador usa
    entonces nombres de relleno.
    """
    if dataset is None:
        return None

    base = dataset
    indices: list[int] | None = None
    while True:  # desanida Subset(Subset(...))
        sub_indices = getattr(base, "indices", None)
        sub_dataset = getattr(base, "dataset", None)
        if sub_indices is None or sub_dataset is None:
            break
        sub_indices = list(sub_indices)
        indices = sub_indices if indices is None else [sub_indices[i] for i in indices]
        base = sub_dataset

    for atributo in ("_image_files", "image_files", "image_paths", "imgs", "samples", "_images"):
        valor = getattr(base, atributo, None)
        if valor is None or isinstance(valor, (str, bytes)):
            continue
        try:
            elementos = list(valor)
        except TypeError:
            continue
        if not elementos:
            continue

        rutas: list[str] = []
        for elemento in elementos:
            if isinstance(elemento, (str, Path)):
                rutas.append(str(elemento))
            elif (
                isinstance(elemento, (tuple, list))
                and elemento
                and isinstance(elemento[0], (str, Path))
            ):
                rutas.append(str(elemento[0]))
            else:
                rutas = []
                break
        if not rutas:
            continue
        if indices is not None:
            if any(i < 0 or i >= len(rutas) for i in indices):
                return None
            rutas = [rutas[i] for i in indices]
        return rutas
    return None


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
def _inferir(
    modelo,
    dataloader,
    device,
    n_clases: int,
    nombres: Sequence[str],
) -> tuple[list[int], list[int], list[float], list[bool], list[str]]:
    """Recorre ``dataloader`` y devuelve listas por imagen (no tensores).

    Returns:
        ``(etiquetas_reales, predicciones, probabilidades, aciertos_top5, rutas)``
        donde ``probabilidades[i]`` es la probabilidad softmax de la clase predicha
        para la imagen ``i``.
    """
    reales: list[int] = []
    predichas: list[int] = []
    probabilidades: list[float] = []
    aciertos_top5: list[bool] = []
    rutas = _rutas_dataset(getattr(dataloader, "dataset", None))

    modelo.eval()
    with torch.no_grad():
        for lote in dataloader:
            if isinstance(lote, dict):  # por comodidad con loaders "tipo dict"
                imagenes, etiquetas = lote.get("image"), lote.get("label")
            else:
                imagenes, etiquetas = lote[0], lote[1]
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.long().to(device)

            logits = modelo(imagenes)
            probs = torch.softmax(logits.detach().float(), dim=1)
            k_top5 = max(1, min(5, int(probs.shape[1])))
            valores, indices = probs.topk(k_top5, dim=1)

            reales.extend(int(v) for v in etiquetas.cpu().tolist())
            predichas.extend(int(v) for v in indices[:, 0].cpu().tolist())
            probabilidades.extend(float(v) for v in valores[:, 0].cpu().tolist())
            aciertos = (indices == etiquetas.unsqueeze(1)).any(dim=1)
            aciertos_top5.extend(bool(v) for v in aciertos.cpu().tolist())

    total = len(reales)
    if rutas is None or len(rutas) != total:
        rutas = [f"imagen_{i:05d}" for i in range(total)]
    return reales, predichas, probabilidades, aciertos_top5, rutas


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def _metricas_vacias(n_clases: int, nombres: Sequence[str]):
    """Metricas neutras para un split sin imagenes (evita que sklearn reviente)."""
    per_class = {
        nombres[i]: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        for i in range(n_clases)
    }
    per_class_f1 = {nombres[i]: 0.0 for i in range(n_clases)}
    matriz = np.zeros((n_clases, n_clases), dtype=np.int64)
    return 0.0, 0.0, 0.0, per_class, per_class_f1, matriz


def _calcular_metricas(
    reales: Sequence[int],
    predichas: Sequence[int],
    aciertos_top5: Sequence[bool],
    n_clases: int,
    nombres: Sequence[str],
):
    """top-1, top-5, macro-F1, reporte por clase y matriz de confusion."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    if not len(reales):
        return _metricas_vacias(n_clases, nombres)

    y_true = np.asarray(reales, dtype=int)
    y_pred = np.asarray(predichas, dtype=int)
    etiquetas = list(range(int(n_clases)))

    top1 = float(np.mean(y_true == y_pred))
    top5 = float(np.mean(np.asarray(aciertos_top5, dtype=float))) if len(aciertos_top5) else top1
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=etiquetas, average="macro", zero_division=0)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, zero_division=0
    )
    per_class = {
        nombres[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(int(n_clases))
    }
    # SA-03 (notebook) consume per_class_f1 como dict nombre -> float.
    per_class_f1 = {nombres[i]: float(f1[i]) for i in range(int(n_clases))}

    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas)
    return top1, top5, macro_f1, per_class, per_class_f1, matriz


def _top_confusiones(matriz: np.ndarray, nombres: Sequence[str]) -> list[list]:
    """``[[real, pred, n], ...]`` ordenada por ``n`` descendente (maximo 20).

    Solo incluye errores (``real != pred``) con ``n >= 1``.
    """
    pares: list[list] = []
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if i == j:
                continue
            n = int(matriz[i, j])
            if n > 0:
                pares.append([str(nombres[i]), str(nombres[j]), n])
    pares.sort(key=lambda fila: (-fila[2], fila[0], fila[1]))
    return pares[:MAX_CONFUSIONES]


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _pyplot():
    """``matplotlib.pyplot`` con backend ``Agg`` (requisito del contrato)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _etiquetas_cortas(nombres: Sequence[str], maximo: int = 22) -> list[str]:
    """Nombres recortados a ``maximo`` caracteres (para los ejes de las figuras)."""
    cortas = []
    for nombre in nombres:
        texto = str(nombre)
        cortas.append(texto if len(texto) <= maximo else texto[: maximo - 1] + ".")
    return cortas


def _guardar_matriz_confusion(matriz: np.ndarray, nombres: Sequence[str], destino: Path) -> None:
    """Heatmap NxN (102x102 con el catalogo real) con escala logaritmica."""
    plt = _pyplot()
    from matplotlib.colors import LogNorm

    n_clases = int(matriz.shape[0])
    datos = np.ma.masked_where(matriz == 0, matriz.astype(float))
    maximo = int(matriz.max()) if matriz.size else 0

    figura, eje = plt.subplots(figsize=(13, 11))
    norma = LogNorm(vmin=1, vmax=max(1, maximo)) if maximo > 0 else None
    imagen = eje.imshow(datos, interpolation="nearest", cmap="Blues", norm=norma)
    figura.colorbar(imagen, ax=eje, fraction=0.046, pad=0.04, label="imagenes")

    if n_clases <= 20:
        marcas = list(range(n_clases))
    else:
        paso = max(1, n_clases // 20)
        marcas = list(range(0, n_clases, paso))
    etiquetas = _etiquetas_cortas(nombres)
    eje.set_xticks(marcas)
    eje.set_yticks(marcas)
    eje.set_xticklabels([etiquetas[i] for i in marcas], rotation=90, fontsize=6)
    eje.set_yticklabels([etiquetas[i] for i in marcas], fontsize=6)
    eje.set_xlabel("Prediccion")
    eje.set_ylabel("Etiqueta real")
    eje.set_title(f"Matriz de confusion ({n_clases} clases)")

    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _guardar_top_confusiones(confusiones: Sequence[Sequence], destino: Path) -> None:
    """Barras horizontales con las confusiones mas frecuentes."""
    plt = _pyplot()

    figura, eje = plt.subplots(figsize=(9, 6))
    if confusiones:
        etiquetas = [
            f"{fila[0]} -> {fila[1]}" if len(str(fila[0]) + str(fila[1])) <= 44
            else f"{str(fila[0])[:20]}... -> {str(fila[1])[:20]}..."
            for fila in confusiones
        ]
        valores = [int(fila[2]) for fila in confusiones]
        posiciones = np.arange(len(valores))[::-1]  # la mas frecuente arriba
        eje.barh(posiciones, valores, color="#c0392b")
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas, fontsize=7)
        eje.set_xlabel("Numero de imagenes")
        eje.set_title("Confusiones mas frecuentes (real -> predicho)")
        eje.grid(axis="x", alpha=0.3)
        for posicion, valor in zip(posiciones, valores):
            eje.text(valor, posicion, f" {valor}", va="center", fontsize=7)
    else:
        eje.text(
            0.5,
            0.5,
            "Sin confusiones: todas las imagenes\nse clasificaron correctamente",
            ha="center",
            va="center",
            fontsize=12,
        )
        eje.set_axis_off()
        eje.set_title("Confusiones mas frecuentes (real -> predicho)")

    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


# --------------------------------------------------------------------------- #
# CSV de predicciones
# --------------------------------------------------------------------------- #
def _guardar_csv(
    destino: Path,
    rutas: Sequence[str],
    reales: Sequence[int],
    predichas: Sequence[int],
    probabilidades: Sequence[float],
    nombres: Sequence[str],
) -> Path:
    """CSV con encabezado ``ruta, etiqueta_real, prediccion, prob, correcto``."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(CABECERA_CSV)
        for ruta, real, predicha, prob in zip(rutas, reales, predichas, probabilidades):
            escritor.writerow(
                [
                    str(ruta),
                    str(nombres[real]),
                    str(nombres[predicha]),
                    f"{float(prob):.6f}",
                    bool(real == predicha),
                ]
            )
    return destino


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def evaluate(
    checkpoint_path: str | Path,
    cfg: Config | None = None,
    split: str = "test",
    *,
    dataloader=None,
) -> dict:
    """Evalua un checkpoint sobre ``split`` y escribe metricas, figuras y CSV.

    Args:
        checkpoint_path: ruta al ``.pt`` guardado por ``src.train``/``src.model``.
        cfg: configuracion de salida. Si es ``None`` se reconstruye desde el propio
            checkpoint con ``Config(**meta["config"])`` (y por tanto se escribe en el
            ``out_dir`` con el que se entreno). El ``img_size`` de los transforms y el
            ``num_classes`` salen siempre del checkpoint.
        split: ``"train"``, ``"val"`` o ``"test"`` (por defecto ``"test"``). Solo se usa
            para elegir el DataLoader, para nombrar el CSV y para el campo ``split``.
        dataloader: **parametro opcional de test** (keyword-only). Si se inyecta un
            ``DataLoader`` ya construido, ``evaluate`` lo usa en lugar de llamar a
            ``src.data.build_dataloaders``; sirve para evaluar sin dataset real y sin
            red (tests). En uso normal debe dejarse en ``None``.

    Robustez de los workers: si el ``DataLoader`` construido con ``cfg.num_workers > 0``
    falla por un problema de *multiprocessing*/*permisos* (tipico en Windows, donde los
    workers necesitan *named pipes*; en el sandbox de este entorno da
    ``PermissionError [WinError 5]``), ``evaluate`` avisa por ``warnings.warn`` y por
    consola y **reintenta una sola vez con ``num_workers=0``** en vez de abortar. El
    reintento solo aplica al loader que construye ``evaluate`` (no a un ``dataloader``
    inyectado).

    Returns:
        El mismo diccionario que se escribe en ``metrics.json``, con las claves
        ``split, num_images, top1, top5, macro_f1, per_class, per_class_f1,
        top_confusions, ruta_confusion, segundos`` (y algunos metadatos extra del
        checkpoint). ``per_class[clase]`` es ``{precision, recall, f1, support}`` y
        ``per_class_f1[clase]`` es directamente el float de F1 (lo consume SA-03).
        ``top_confusions`` es ``[[real, pred, n], ...]`` ordenada por ``n`` descendente
        (maximo 20).

    Raises:
        FileNotFoundError: si el checkpoint no existe (hay que entrenar primero).
        ValueError: si ``split`` no es valido o el checkpoint trae un ``cfg`` invalido.
    """
    inicio = time.perf_counter()

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint '{ckpt}'. Para generarlo, {MENSAJE_ENTRENA}."
        )

    split = str(split).lower()
    if split not in SPLITS_VALIDOS:
        raise ValueError(
            f"split invalido: {split!r}. Validos: {', '.join(SPLITS_VALIDOS)}."
        )

    device = resolve_device(getattr(cfg, "device", "auto") or "auto")
    modelo, nombres_ckpt, meta = load_checkpoint(ckpt, device=device)

    if cfg is None:
        cfg = _config_desde_meta(meta)

    n_clases = int(meta.get("num_classes") or len(nombres_ckpt) or getattr(cfg, "num_classes", 0))
    if n_clases <= 0:
        raise ValueError(
            f"El checkpoint '{ckpt}' no define num_classes y no se pudo inferir "
            f"(class_names={len(nombres_ckpt)})."
        )
    nombres = _ajustar_nombres(nombres_ckpt, n_clases)

    # El tamano de imagen del checkpoint manda (no asumir 224).
    img_size = int(meta.get("img_size") or getattr(cfg, "img_size", IMAGE_SIZE))
    cfg = _cfg_con_img_size(cfg, img_size)
    cfg.ensure_dirs()

    reales, predichas, probabilidades, aciertos_top5, rutas = _inferir_robusto(
        modelo, cfg, split, dataloader, device, n_clases, nombres
    )
    top1, top5, macro_f1, per_class, per_class_f1, matriz = _calcular_metricas(
        reales, predichas, aciertos_top5, n_clases, nombres
    )

    confusiones = _top_confusiones(matriz, nombres)
    ruta_confusion = cfg.figures_dir() / "confusion_matrix.png"
    _guardar_matriz_confusion(matriz, nombres, ruta_confusion)
    _guardar_top_confusiones(confusiones, cfg.figures_dir() / "top_confusions.png")
    ruta_csv = _guardar_csv(
        cfg.predictions_dir() / f"{split}_predicciones.csv",
        rutas,
        reales,
        predichas,
        probabilidades,
        nombres,
    )

    metricas: dict[str, Any] = {
        "split": split,
        "num_images": int(len(reales)),
        "top1": float(top1),
        "top5": float(top5),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "per_class_f1": per_class_f1,
        "top_confusions": confusiones,
        "ruta_confusion": str(ruta_confusion),
        "segundos": float(round(time.perf_counter() - inicio, 3)),
        # --- metadatos extra (utiles para el INFORME; no rompen el contrato) ---
        "checkpoint": str(ckpt),
        "backbone": str(meta.get("backbone", "")),
        "num_classes": int(n_clases),
        "img_size": int(img_size),
        "device": str(device),
        "epoch": int(meta.get("epoch", -1) or -1),
        "val_acc": float(meta.get("val_acc", 0.0) or 0.0),
        "ruta_predicciones": str(ruta_csv),
    }

    save_json(cfg.metrics_path, metricas)

    print(
        f"[evaluate] split={split} imagenes={metricas['num_images']} "
        f"top1={metricas['top1']:.4f} top5={metricas['top5']:.4f} "
        f"macro_f1={metricas['macro_f1']:.4f} ({metricas['segundos']:.2f}s)"
    )
    print(f"[evaluate] metricas: {cfg.metrics_path}")
    print(f"[evaluate] figuras:  {ruta_confusion} | {cfg.figures_dir() / 'top_confusions.png'}")
    print(f"[evaluate] csv:      {ruta_csv}")

    return metricas


# --------------------------------------------------------------------------- #
# CLI  (python -m src.evaluate --help)
# --------------------------------------------------------------------------- #
def build_parser():
    """Parser de la CLI congelada de la seccion 3.b."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evalua un checkpoint de flores sobre el split val o test.",
    )
    parser.add_argument(
        "--checkpoint",
        default="resultados/best.pt",
        help="ruta del checkpoint (por defecto: resultados/best.pt)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="split a evaluar (por defecto: test)",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="directorio de salida (por defecto: resultados)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto|cuda|cpu (por defecto: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="usa el subconjunto debug (cfg.debug_per_class imagenes por clase)",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=None,
        help=(
            "workers del DataLoader (por defecto: los de Config, 2). Usa 0 en Windows "
            "o dentro de un sandbox donde los workers no puedan abrir named pipes"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrada de la CLI. Devuelve 0 si todo fue bien y 1 si falta el checkpoint."""
    args = build_parser().parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"[error] No existe el checkpoint '{ckpt}'. Para generarlo, {MENSAJE_ENTRENA}.")
        print("[error] Comando: python -m src.train")
        return 1

    if args.num_workers is not None and int(args.num_workers) < 0:
        print(f"[error] --num-workers debe ser >= 0, se recibio {args.num_workers}.")
        return 1

    # Solo se pasa num_workers si el usuario lo pidio: el default sigue siendo el de
    # Config (2), que es el correcto en Colab/Linux.
    extra = {} if args.num_workers is None else {"num_workers": int(args.num_workers)}
    cfg = Config(
        out_dir=args.out_dir or "resultados",
        device=args.device,
        debug=bool(args.debug),
        pretrained=False,  # nunca se descargan pesos al evaluar
        **extra,
    )
    try:
        evaluate(ckpt, cfg, split=args.split)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de la CLI
    raise SystemExit(main())
