"""Entrenamiento con flores propias (catalogo aparte del de Oxford Flowers-102).

CONTRATO (Firmas congeladas):

    RUTA_FOTOS_POR_DEFECTO = "mis_flores"
    OUT_DIR_POR_DEFECTO = "resultados/custom"

    revisar_carpeta(ruta=RUTA_FOTOS_POR_DEFECTO, min_por_clase=10) -> dict
    preparar_loaders(cfg, ruta_fotos=RUTA_FOTOS_POR_DEFECTO,
                     val_fraccion=0.2, semilla=None) -> dict
    entrenar_custom(cfg=None, ruta_fotos=RUTA_FOTOS_POR_DEFECTO,
                    val_fraccion=0.2, out_dir=None) -> dict
    evaluar_custom(checkpoint_path, cfg=None, ruta_fotos=RUTA_FOTOS_POR_DEFECTO,
                   split="val") -> dict
    main(argv=None) -> int

    # CLI: python -m src.custom [--fotos RUTA] [--epochs N] [--batch-size N]
    #                           [--backbone B] [--out-dir R] [--device auto|cuda|cpu]
    #                           [--val-fraccion 0.2] [--semilla N] [--debug]
    #                           [--revisar] [--evaluar RUTA_CHECKPOINT]

Estructura esperada de las fotos (una carpeta por especie)::

    mis_flores/
      mi_rosa/
        foto1.jpg
        foto2.jpg
      mi_orquidea/
        foto1.jpg

Que hace cada pieza:

  * :func:`revisar_carpeta` inspecciona la carpeta y **informa** (nunca lanza
    excepcion): problemas que impiden entrenar, avisos de pocas fotos o de
    desequilibrio entre especies, y una recomendacion en texto.
  * :func:`preparar_loaders` construye dos ``DataLoader`` (train/val) con
    ``ImageFolder`` y las transformaciones de :func:`src.data.build_transforms`.
    Antes de transformar cada foto se aplica ``ImageOps.exif_transpose`` (el giro
    de las fotos de movil vive en los metadatos EXIF: sin esto entran tumbadas) y
    ``convert("RGB")`` (fotos en escala de grises, paleta o RGBA). El reparto
    train/val es deterministico y **sin fuga**: cada foto cae en un solo lado.
  * :func:`entrenar_custom` copia la ``Config`` recibida, ajusta ``num_classes`` y
    ``out_dir``, inyecta los loaders propios en :func:`src.train.train` (asi nunca
    se toca el dataset de Oxford) y escribe ``catalogo.json`` con la trazabilidad
    del catalogo usado.
  * :func:`evaluar_custom` reutiliza :func:`src.evaluate.evaluate` inyectandole el
    ``DataLoader`` del split pedido: no hay logica de metricas nueva.
  * El checkpoint resultante trae ``class_names``, ``num_classes``, ``backbone``,
    ``img_size`` y la config, asi que ``src.predict.predict_image``,
    :func:`src.evaluate.evaluate` y la demo funcionan con el **sin cambiar nada**.

Notas de diseno:

  - Nada de rutas absolutas de Windows ni de ``.cuda()``: el device lo resuelve
    ``src.train`` con ``cfg.device`` y ``pin_memory`` solo se activa si hay CUDA
    (igual que ``src/data.py``).
  - ``num_workers`` es ``cfg.num_workers`` (2 en Colab/Linux, 0 con ``--debug``).
    En este sandbox de Windows los workers no pueden abrir *named pipes*; como
    ``src.train`` no puede reconstruir unos loaders inyectados, este modulo
    reintenta UNA vez la corrida completa con ``num_workers=0`` si el fallo huele
    a multiprocessing (mismo criterio que ``src/train.py`` y ``src/evaluate.py``).
  - ``entrenar_custom`` no cambia ni un default de ``src/config.py``: los valores
    propios del catalogo (15 epocas, lote 16, efficientnet_b0, pretrained=True)
    solo se aplican cuando no se recibe ninguna ``Config``.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset

try:  # import relativo (paquete `src`) con respaldo absoluto (cwd en sys.path)
    from .config import Config
    from .data import build_transforms
    from .evaluate import SPLITS_VALIDOS, evaluate
    from .train import train
    from .utils import save_json
except ImportError:  # pragma: no cover - depende de como se importe el modulo
    from src.config import Config
    from src.data import build_transforms
    from src.evaluate import SPLITS_VALIDOS, evaluate
    from src.train import train
    from src.utils import save_json

__all__ = [
    "RUTA_FOTOS_POR_DEFECTO",
    "OUT_DIR_POR_DEFECTO",
    "EXTENSIONES_FOTO",
    "CATALOGO_FILENAME",
    "revisar_carpeta",
    "preparar_loaders",
    "entrenar_custom",
    "evaluar_custom",
    "build_argparser",
    "main",
]

# ---------------------------------------------------------------- constantes
#: Carpeta donde el usuario deja sus fotos (una subcarpeta por especie).
RUTA_FOTOS_POR_DEFECTO = "mis_flores"
#: Carpeta de salida del modelo propio (nunca pisa ``resultados/`` de Oxford).
OUT_DIR_POR_DEFECTO = "resultados/custom"
#: Salida del CLI cuando se usa ``--debug`` sin ``--out-dir`` (no pisar lo definitivo).
OUT_DIR_DEBUG_POR_DEFECTO = "resultados/custom_debug"
#: Nombre del archivo de trazabilidad del catalogo dentro de ``out_dir``.
CATALOGO_FILENAME = "catalogo.json"

#: Valores propios del modulo cuando no se recibe ninguna ``Config``.
EPOCHS_POR_DEFECTO = 15
BATCH_POR_DEFECTO = 16
BACKBONE_POR_DEFECTO = "efficientnet_b0"
PRETRAINED_POR_DEFECTO = True
#: Fraccion de cada especie reservada para validacion.
VAL_FRACCION_POR_DEFECTO = 0.2
#: Aviso si una especie tiene menos fotos que esto.
MIN_POR_CLASE_POR_DEFECTO = 10
#: Aviso si la clase mas grande tiene mas de este multiplo de la mas pequena.
DESEQUILIBRIO_UMBRAL = 3.0
#: Extensiones de foto admitidas (las mismas que documenta ``mis_flores/LEEME.md``).
EXTENSIONES_FOTO: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
#: Texto unico del modo alternativo para catalogos muy pequenos (no implementado).
MODO_MATRICULACION = "modo de matriculacion (aun no implementado)"

#: Pistas de texto que identifican un fallo de los workers del ``DataLoader``.
_PISTAS_WORKERS: tuple[str, ...] = (
    "winerror 5",
    "acceso denegado",
    "access is denied",
    "permission",
    "named pipe",
    "pipe",
    "multiprocessing",
    "worker",
    "shared memory",
    "shared_memory",
    "semaphore",
    "eof",
    "queue",
)


# --------------------------------------------------------------------------- #
# Helpers internos
# --------------------------------------------------------------------------- #
def _formatos() -> str:
    """Lista legible de extensiones admitidas."""
    return ", ".join(EXTENSIONES_FOTO)


def _es_imagen_valida(ruta: str) -> bool:
    """True si ``ruta`` tiene una extension de foto admitida.

    Es una funcion de modulo (no un ``lambda``) porque ``ImageFolder`` la guarda
    para los procesos hijo del ``DataLoader``: tiene que ser *picklable*.
    """
    return Path(ruta).suffix.lower() in EXTENSIONES_FOTO


def _fotos_de(carpeta: Path) -> list[Path]:
    """Fotos admitidas dentro de ``carpeta``, ordenadas y sin bajar a subcarpetas."""
    try:
        entradas = sorted(carpeta.iterdir())
    except OSError:  # carpeta ilegible: se trata como vacia, nunca revienta
        return []
    return [p for p in entradas if p.is_file() and p.suffix.lower() in EXTENSIONES_FOTO]


def _image_folder():
    """Devuelve ``torchvision.datasets.ImageFolder`` (import perezoso).

    ``torchvision.datasets`` importa ``scipy`` de forma eager; el import perezoso
    mantiene ``import src.custom`` barato y convierte el ``ImportError`` en un
    mensaje util (mismo criterio que ``src/data.py``).
    """
    try:
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        if "scipy" in str(exc).lower():
            raise ImportError(
                "torchvision.datasets no pudo importarse porque falta scipy. "
                "Instala: pip install scipy"
            ) from exc
        raise
    return ImageFolder


class _FotoDeMovil:
    """Limpieza previa de una foto de movil seguida del transform base.

    Hace dos cosas antes de ``build_transforms``:

      1. ``ImageOps.exif_transpose``: aplica el giro que la camara guardo en los
         metadatos EXIF. Sin esto las fotos verticales entran tumbadas y el modelo
         aprende con datos peores.
      2. ``convert("RGB")``: garantiza 3 canales (fotos en escala de grises,
         paleta, CMYK o RGBA).

    Es una clase de modulo (no una funcion anidada) para que sea *picklable*
    cuando ``num_workers > 0`` (Colab/Linux).
    """

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, imagen):
        if isinstance(imagen, Image.Image):
            imagen = ImageOps.exif_transpose(imagen)
            if imagen.mode != "RGB":
                imagen = imagen.convert("RGB")
        return self.transform(imagen)

    def __repr__(self) -> str:  # pragma: no cover - solo para depuracion
        return f"_FotoDeMovil({self.transform!r})"


def _repartir_clases(
    muestras: Sequence[tuple[str, int]],
    nombres: Sequence[str],
    val_fraccion: float,
    semilla: int,
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Reparto train/val deterministico y sin fuga, estrato por estrato (especie).

    Args:
        muestras: ``ImageFolder.samples`` (``(ruta, etiqueta)`` por foto).
        nombres: nombres de clase en el orden de las etiquetas.
        val_fraccion: fraccion de cada especie que va a validacion.
        semilla: semilla del reparto (misma semilla -> mismo reparto).

    Returns:
        ``(indices_train, indices_val, reparto)`` donde ``reparto[i]`` es
        ``{"train": n, "val": m, "total": n + m}`` para la clase ``i``.

    Cada especie se baraja con la MISMA instancia de ``random.Random`` (en orden
    alfabetico de etiqueta), asi que el resultado no depende del sistema de
    archivos. Una foto nunca aparece en train y en val: los indices se reparten
    con ``corte`` y no se solapan.
    """
    por_clase: dict[int, list[int]] = {}
    for indice, (_ruta, etiqueta) in enumerate(muestras):
        por_clase.setdefault(int(etiqueta), []).append(indice)

    generador = random.Random(int(semilla))
    indices_train: list[int] = []
    indices_val: list[int] = []
    reparto: list[dict[str, int]] = [
        {"train": 0, "val": 0, "total": 0} for _ in range(len(nombres))
    ]

    for etiqueta in sorted(por_clase):
        indices = list(por_clase[etiqueta])
        generador.shuffle(indices)  # deterministico con la semilla
        total = len(indices)
        n_val = max(1, int(round(total * float(val_fraccion))))
        n_val = min(n_val, total - 1)  # train conserva siempre al menos una foto
        indices_val.extend(indices[:n_val])
        indices_train.extend(indices[n_val:])
        if 0 <= etiqueta < len(reparto):
            reparto[etiqueta] = {"train": total - n_val, "val": n_val, "total": total}

    indices_train.sort()  # orden estable (solo importa para tests/inspeccion)
    indices_val.sort()
    return indices_train, indices_val, reparto


def _parece_fallo_de_workers(exc: BaseException) -> bool:
    """True si la excepcion huele a fallo de los procesos hijo del ``DataLoader``.

    En el sandbox de Windows de este entorno los workers no pueden abrir *named
    pipes* y revientan con ``PermissionError [WinError 5]``; en Colab/Linux
    funciona bien. Mismo criterio que ``src/train.py``.
    """
    if isinstance(exc, PermissionError):
        return True
    if not isinstance(exc, (OSError, RuntimeError, EOFError, MemoryError)):
        return False
    texto = f"{type(exc).__name__}: {exc}".lower()
    return any(pista in texto for pista in _PISTAS_WORKERS)


def _cfg_base(cfg: Config | None) -> Config:
    """Copia de ``cfg`` (para no mutar la del llamador) o la Config propia del modulo."""
    if cfg is not None:
        return replace(cfg)
    return Config(
        epochs=EPOCHS_POR_DEFECTO,
        batch_size=BATCH_POR_DEFECTO,
        backbone=BACKBONE_POR_DEFECTO,
        pretrained=PRETRAINED_POR_DEFECTO,
    )


def _catalogo_json(checkpoint_path: Path) -> dict:
    """Lee ``catalogo.json`` junto al checkpoint (``{}`` si no existe o esta roto)."""
    import json

    ruta = Path(checkpoint_path).parent / CATALOGO_FILENAME
    if not ruta.is_file():
        return {}
    try:
        with open(ruta, "r", encoding="utf-8") as fh:
            datos = json.load(fh)
    except (OSError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


def _payload_checkpoint(checkpoint_path: Path) -> dict:
    """Contenido crudo del ``.pt`` (para leer ``img_size``/``num_classes``/``config``)."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch muy antiguo sin weights_only
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"El checkpoint '{checkpoint_path}' no tiene el formato esperado (dict)."
        )
    return payload


def _cfg_para_checkpoint(checkpoint_path: Path, cfg: Config | None = None) -> Config:
    """Config para construir los loaders, con el ``img_size`` del checkpoint.

    El tamano de imagen manda desde el checkpoint (no se asume 224): si el modelo
    se entreno con otro ``img_size``, los transforms tienen que coincidir. Cuando
    no se recibe ``cfg`` se reconstruye la del propio checkpoint (asi
    ``metrics.json`` cae junto al modelo, como hace ``src/evaluate.py``).
    """
    payload = _payload_checkpoint(Path(checkpoint_path))

    if cfg is None:
        valores = dict(payload.get("config") or {})
        validos = {campo.name for campo in fields(Config)}
        cfg = Config(**{clave: valor for clave, valor in valores.items() if clave in validos})

    cambios: dict[str, Any] = {}
    img_size = int(payload.get("img_size") or 0)
    num_classes = int(payload.get("num_classes") or 0)
    if img_size > 0:
        cambios["img_size"] = img_size
    if num_classes > 0:
        cambios["num_classes"] = num_classes
    return replace(cfg, **cambios) if cambios else cfg


# --------------------------------------------------------------------------- #
# Revision del catalogo
# --------------------------------------------------------------------------- #
def _recomendacion(ok: bool, avisos: Sequence[str], n_clases: int, total: int, ruta: str) -> str:
    """Texto accionable: que hacer con el catalogo tal como esta."""
    if not ok:
        return (
            "Corrige los problemas listados y vuelve a ejecutar "
            "`python -m src.custom --revisar`."
        )
    if avisos:
        return (
            f"Se puede entrenar con {n_clases} especie(s) y {total} foto(s), pero el "
            "resultado mejorara anadiendo fotos: lo ideal son 20-50 por especie. "
            f"Con menos de {MIN_POR_CLASE_POR_DEFECTO} fotos por especie conviene "
            f"esperar al {MODO_MATRICULACION}."
        )
    return (
        f"Carpeta lista: {n_clases} especie(s) y {total} foto(s). "
        f"Entrena con: `python -m src.custom --fotos {ruta}`."
    )


def revisar_carpeta(
    ruta: str | Path = RUTA_FOTOS_POR_DEFECTO,
    min_por_clase: int = MIN_POR_CLASE_POR_DEFECTO,
) -> dict:
    """Inspecciona el catalogo de fotos propias y devuelve un informe (NO lanza).

    Estructura esperada: ``<ruta>/<especie>/<foto>`` con fotos ``.jpg``, ``.jpeg``,
    ``.png`` o ``.webp``.

    Args:
        ruta: carpeta con una subcarpeta por especie.
        min_por_clase: minimo recomendado de fotos por especie; por debajo se emite
            un aviso (no un problema).

    Returns:
        ``{"ok", "ruta", "clases", "total", "min_por_clase", "avisos", "problemas",
        "recomendacion"}`` donde

        * ``ok``: ``True`` solo si no hay ningun problema que impida entrenar;
        * ``clases``: ``{nombre_de_especie: numero_de_fotos}`` (incluye las carpetas
          de especie vacias con 0 fotos);
        * ``total``: suma de fotos de todas las especies;
        * ``avisos``: cosas que no bloquean (pocas fotos, desequilibrio);
        * ``problemas``: cosas que impiden entrenar (carpeta ausente o vacia, menos
          de 2 especies, fotos sueltas sin carpeta de especie, carpetas vacias);
        * ``recomendacion``: que hacer, en una frase.

    Nunca lanza excepcion: cualquier situacion se describe en ``problemas``.
    """
    base = Path(ruta)
    minimo = int(min_por_clase)
    clases: dict[str, int] = {}
    avisos: list[str] = []
    problemas: list[str] = []

    if not base.exists():
        problemas.append(
            f"no existe la carpeta '{base}'. Crea '{base}/<especie>/' y pon dentro "
            f"las fotos ({_formatos()})."
        )
    elif not base.is_dir():
        problemas.append(f"'{base}' existe pero no es una carpeta.")
    else:
        carpetas_vacias: list[str] = []
        sueltas: list[str] = []
        try:
            entradas = sorted(base.iterdir())
        except OSError as exc:  # carpeta ilegible: se informa, no se lanza
            entradas = []
            problemas.append(f"no se pudo leer la carpeta '{base}' ({type(exc).__name__}: {exc}).")
        for entrada in entradas:
            if entrada.is_dir():
                fotos = _fotos_de(entrada)
                clases[entrada.name] = len(fotos)
                if not fotos:
                    carpetas_vacias.append(entrada.name)
            elif entrada.is_file() and entrada.suffix.lower() in EXTENSIONES_FOTO:
                sueltas.append(entrada.name)

        for nombre in carpetas_vacias:
            problemas.append(
                f"la carpeta de especie '{nombre}' no tiene ninguna foto "
                f"(formatos admitidos: {_formatos()})."
            )
        if sueltas:
            problemas.append(
                f"hay {len(sueltas)} foto(s) suelta(s) en '{base}' sin carpeta de "
                f"especie (por ejemplo '{sueltas[0]}'); muevelas a '{base}/<especie>/'."
            )

        total_temporal = int(sum(clases.values()))
        con_fotos = {nombre: n for nombre, n in clases.items() if n > 0}
        if not total_temporal:
            problemas.append(
                f"no se encontro ninguna foto en '{base}' "
                f"(formatos admitidos: {_formatos()})."
            )
        elif len(con_fotos) < 2:
            problemas.append(
                f"hacen falta al menos 2 especies con fotos y solo hay "
                f"{len(con_fotos)} ('{next(iter(con_fotos))}')."
            )

    total = int(sum(clases.values()))
    con_fotos = {nombre: n for nombre, n in clases.items() if n > 0}

    # Avisos: no bloquean el entrenamiento, pero condicionan el resultado.
    for nombre in sorted(con_fotos):
        n = con_fotos[nombre]
        if n < minimo:
            avisos.append(
                f"la especie '{nombre}' tiene {n} foto(s): menos de {minimo}. Con tan "
                f"pocas fotos el resultado sera pobre; anade mas o usa el "
                f"{MODO_MATRICULACION}."
            )
    if len(con_fotos) >= 2:
        maximo = max(con_fotos.values())
        minimo_real = min(con_fotos.values())
        if minimo_real > 0 and maximo / float(minimo_real) > DESEQUILIBRIO_UMBRAL:
            mejor = max(sorted(con_fotos), key=lambda nombre: con_fotos[nombre])
            peor = min(sorted(con_fotos), key=lambda nombre: con_fotos[nombre])
            avisos.append(
                f"las clases estan desequilibradas: '{mejor}' tiene {maximo} fotos y "
                f"'{peor}' solo {minimo_real} (x{maximo / float(minimo_real):.1f}). El "
                "modelo tendera a confundir la especie con menos fotos; iguala el "
                "numero de fotos por especie."
            )

    ok = not problemas
    return {
        "ok": bool(ok),
        "ruta": str(base),
        "clases": clases,
        "total": total,
        "min_por_clase": minimo,
        "avisos": avisos,
        "problemas": problemas,
        "recomendacion": _recomendacion(ok, avisos, len(con_fotos), total, str(base)),
    }


# --------------------------------------------------------------------------- #
# DataLoaders del catalogo propio
# --------------------------------------------------------------------------- #
def preparar_loaders(
    cfg: Config,
    ruta_fotos: str | Path = RUTA_FOTOS_POR_DEFECTO,
    val_fraccion: float = VAL_FRACCION_POR_DEFECTO,
    semilla: int | None = None,
) -> dict:
    """``DataLoader`` de train/val para las fotos propias, sin fuga de datos.

    Pasos:

      1. :func:`revisar_carpeta` valida el catalogo (se lanza ``ValueError`` con los
         problemas si no es utilizable);
      2. se crea **dos veces** el ``ImageFolder`` sobre la misma carpeta: uno con el
         transform de train y otro con el de val (``build_transforms(cfg, train)``),
         cada uno envuelto en :class:`_FotoDeMovil` (EXIF + RGB);
      3. cada especie se reparte train/val con ``val_fraccion`` de forma
         determinista (``semilla`` o ``cfg.seed``) y **sin fuga**: los indices de
         train y val son disjuntos, y como los dos ``ImageFolder`` ordenan las
         clases y las fotos igual, los indices apuntan a las mismas fotos.

    Args:
        cfg: configuracion (aporta ``img_size``, ``batch_size``, ``num_workers``,
            ``seed`` y ``debug``). No puede ser ``None``.
        ruta_fotos: carpeta con una subcarpeta por especie.
        val_fraccion: fraccion de cada especie reservada para validacion
            (``0 < val_fraccion < 1``).
        semilla: semilla del reparto; ``None`` usa ``cfg.seed``.

    Returns:
        ``{"train": DataLoader, "val": DataLoader, "class_names": [...],
        "n_por_clase": {nombre: {"train", "val", "total"}}, "rutas_val": [...],
        "informe": {...}}`` (``informe`` es el de :func:`revisar_carpeta`).

    Raises:
        ValueError: si el catalogo no es utilizable, si ``val_fraccion`` esta fuera
            de rango o si alguna especie tiene menos de 2 fotos (no se podria
            reservar una para validacion sin dejar la especie sin entrenamiento).
    """
    if cfg is None:
        raise ValueError("preparar_loaders necesita una Config (cfg=None).")

    fraccion = float(val_fraccion)
    if not 0.0 < fraccion < 1.0:
        raise ValueError(
            f"val_fraccion debe estar entre 0 y 1 (sin incluir los extremos), "
            f"se recibio {val_fraccion!r}."
        )

    informe = revisar_carpeta(ruta_fotos)
    if not informe["ok"]:
        raise ValueError(
            "El catalogo de fotos no es utilizable: " + " | ".join(informe["problemas"])
        )

    escasas = sorted(n for n, total in informe["clases"].items() if 0 < total < 2)
    if escasas:
        raise ValueError(
            "Cada especie necesita al menos 2 fotos para poder reservar una en "
            f"validacion sin dejar la especie sin entrenamiento: {escasas}."
        )

    root = Path(ruta_fotos)
    ImageFolder = _image_folder()
    transform_train = _FotoDeMovil(build_transforms(cfg, train=True))
    transform_val = _FotoDeMovil(build_transforms(cfg, train=False))

    # Dos vistas del MISMO catalogo: mismo orden de clases y de fotos, distinto
    # transform. Se necesitan dos porque un Subset no puede cambiar el transform.
    dataset_train = ImageFolder(str(root), transform=transform_train, is_valid_file=_es_imagen_valida)
    dataset_val = ImageFolder(str(root), transform=transform_val, is_valid_file=_es_imagen_valida)

    class_names = [str(nombre) for nombre in dataset_train.classes]
    if [str(nombre) for nombre in dataset_val.classes] != class_names:
        raise RuntimeError(
            "Los dos ImageFolder del catalogo propio no coinciden en las clases; "
            "esto no deberia pasar (el orden de carpetas es determinista)."
        )

    semilla_efectiva = int(cfg.seed if semilla is None else semilla)
    indices_train, indices_val, reparto = _repartir_clases(
        dataset_train.samples, class_names, fraccion, semilla_efectiva
    )
    if not indices_val:
        raise ValueError(
            "El reparto de validacion quedo vacio: cada especie necesita al menos "
            "2 fotos."
        )

    sub_train = Subset(dataset_train, indices_train)
    sub_val = Subset(dataset_val, indices_val)

    # pin_memory solo con CUDA (en CPU no aporta y consume RAM), igual que src/data.py.
    pin_memory = bool(torch.cuda.is_available())
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)
    batch_size = int(cfg.batch_size)

    loader_train = DataLoader(
        sub_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    loader_val = DataLoader(
        sub_val,
        batch_size=batch_size,
        shuffle=False,  # val SIN barajar (mismo criterio que src/data.py)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    n_por_clase = {
        nombre: dict(reparto[etiqueta]) for etiqueta, nombre in enumerate(class_names)
    }
    rutas_val = [str(dataset_val.samples[indice][0]) for indice in indices_val]

    return {
        "train": loader_train,
        "val": loader_val,
        "class_names": class_names,
        "n_por_clase": n_por_clase,
        "rutas_val": rutas_val,
        "informe": informe,
    }


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def _loaders_inyectables(paquete: dict) -> tuple:
    """``(train_dl, val_dl, test_dl, class_names)`` para ``src.train.train``.

    El catalogo propio no tiene split de test oficial: el loader del split se usa
    tambien como ``test`` (``src.train`` solo lo usa para informar el numero de
    lotes).
    """
    return (
        paquete["train"],
        paquete["val"],
        paquete["val"],
        paquete["class_names"],
    )


def entrenar_custom(
    cfg: Config | None = None,
    ruta_fotos: str | Path = RUTA_FOTOS_POR_DEFECTO,
    val_fraccion: float = VAL_FRACCION_POR_DEFECTO,
    out_dir: str | Path | None = None,
) -> dict:
    """Entrena un modelo propio con las fotos de ``ruta_fotos``.

    La ``Config`` de entrada se COPIA (nunca se muta) y solo se le sobreescriben
    ``num_classes`` (numero de especies) y ``out_dir`` (``out_dir`` o
    :data:`OUT_DIR_POR_DEFECTO`). Si no se recibe ``cfg`` se parte de los valores
    propios del modulo: ``epochs=15``, ``batch_size=16``,
    ``backbone="efficientnet_b0"``, ``pretrained=True``. Los loaders se inyectan en
    :func:`src.train.train`, asi que el dataset de Oxford no se toca nunca.

    Efectos secundarios (los de ``src.train.train``): ``best.pt``, ``history.json``,
    ``config_usada.json`` y ``figuras/curvas.png`` dentro de ``out_dir``; ademas se
    escribe ``catalogo.json`` con la trazabilidad del catalogo usado (clases, fotos
    por clase, total, backbone, fecha ISO, ruta de las fotos, ``val_fraccion`` y
    semilla).

    Args:
        cfg: configuracion base (opcional).
        ruta_fotos: carpeta con una subcarpeta por especie.
        val_fraccion: fraccion de cada especie reservada para validacion.
        out_dir: carpeta de salida (por defecto :data:`OUT_DIR_POR_DEFECTO`).

    Returns:
        ``{"checkpoint": Path, "class_names": [...], "historial": dict,
        "informe": dict, "out_dir": Path}``.

    Raises:
        ValueError: si el catalogo no es utilizable.
    """
    base = _cfg_base(cfg)
    destino = Path(out_dir or OUT_DIR_POR_DEFECTO)
    paquete = preparar_loaders(
        base, ruta_fotos=ruta_fotos, val_fraccion=val_fraccion, semilla=None
    )
    class_names = list(paquete["class_names"])

    cfg_final = replace(
        base,
        num_classes=int(len(class_names)),
        out_dir=str(destino),
    )

    historial = _entrenar_con_reintento(
        cfg_final, _loaders_inyectables(paquete), ruta_fotos, val_fraccion
    )

    catalogo = {
        "clases": class_names,
        "fotos_por_clase": dict(paquete["informe"]["clases"]),
        "reparto_por_clase": paquete["n_por_clase"],
        "total": int(paquete["informe"]["total"]),
        "backbone": str(cfg_final.backbone),
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ruta_fotos": str(Path(ruta_fotos)),
        "val_fraccion": float(val_fraccion),
        "semilla": int(cfg_final.seed),
        "num_classes": int(len(class_names)),
        "img_size": int(cfg_final.img_size),
        "epochs": int(cfg_final.epochs),
        "batch_size": int(cfg_final.batch_size),
        "pretrained": bool(cfg_final.pretrained),
        "checkpoint": str(cfg_final.checkpoint_path),
    }
    save_json(destino / CATALOGO_FILENAME, catalogo)
    print(f"[custom] catalogo: {destino / CATALOGO_FILENAME}")

    return {
        "checkpoint": Path(cfg_final.checkpoint_path),
        "class_names": class_names,
        "historial": historial,
        "informe": paquete["informe"],
        "out_dir": destino,
    }


def _entrenar_con_reintento(
    cfg: Config,
    loaders: tuple,
    ruta_fotos: str | Path,
    val_fraccion: float,
) -> dict:
    """``train(cfg, loaders=...)`` con UN reintento sin workers si hace falta.

    ``src.train.train`` no reconstruye loaders inyectados (los datos son del
    llamador), asi que el reintento con ``num_workers=0`` se hace aqui: se
    reconstruyen los loaders del catalogo propio y se repite la corrida. En
    Colab/Linux no se activa nunca.
    """
    try:
        return train(cfg, loaders=loaders)
    except Exception as exc:  # noqa: BLE001 - se decide con _parece_fallo_de_workers
        workers = int(getattr(cfg, "num_workers", 0) or 0)
        if workers <= 0 or not _parece_fallo_de_workers(exc):
            raise
        aviso = (
            f"El DataLoader con num_workers={workers} fallo "
            f"({type(exc).__name__}: {exc}). Se reintenta una sola vez con "
            "num_workers=0 (en Windows los workers de multiprocessing necesitan "
            "named pipes; en Colab/Linux los workers funcionan)."
        )
        print("[custom] aviso: " + aviso)
        warnings.warn(aviso, RuntimeWarning, stacklevel=2)

        cfg_sin_workers = replace(cfg, num_workers=0)
        paquete = preparar_loaders(
            cfg_sin_workers,
            ruta_fotos=ruta_fotos,
            val_fraccion=val_fraccion,
            semilla=None,
        )
        return train(cfg_sin_workers, loaders=_loaders_inyectables(paquete))


# --------------------------------------------------------------------------- #
# Evaluacion
# --------------------------------------------------------------------------- #
def evaluar_custom(
    checkpoint_path: str | Path,
    cfg: Config | None = None,
    ruta_fotos: str | Path = RUTA_FOTOS_POR_DEFECTO,
    split: str = "val",
) -> dict:
    """Evalua un checkpoint propio sobre ``split`` reutilizando ``src.evaluate``.

    El ``DataLoader`` del split se construye con :func:`preparar_loaders` y se
    inyecta en :func:`src.evaluate.evaluate`, asi que **no hay logica de metricas
    nueva**: ``metrics.json``, la matriz de confusion, las confusiones frecuentes y
    el CSV son exactamente los del proyecto.

    Splits: el catalogo propio solo reserva fotos de validacion, por lo que
    ``"val"`` y ``"test"`` evaluan sobre esas mismas fotos (el informe lo dice en
    ``split``) y ``"train"`` evalua sobre las fotos de entrenamiento (util para
    detectar sobreajuste).

    El ``img_size`` y la ``Config`` salen del checkpoint cuando ``cfg`` es ``None``.
    La ``val_fraccion`` y la semilla del reparto se leen de ``catalogo.json`` si
    esta junto al checkpoint (asi el split evaluado es el mismo que se entreno); si
    no existe se usan ``0.2`` y ``cfg.seed``.

    Args:
        checkpoint_path: ruta del ``.pt`` propio (por ejemplo
            ``resultados/custom/best.pt``).
        cfg: configuracion de salida (``out_dir`` decide donde cae
            ``metrics.json``). Si es ``None`` se reconstruye de la del checkpoint.
        ruta_fotos: carpeta del catalogo propio.
        split: ``"train"``, ``"val"`` o ``"test"``.

    Returns:
        El mismo diccionario que ``src.evaluate.evaluate`` (``top1``, ``top5``,
        ``macro_f1``, ``per_class``, ...).

    Raises:
        FileNotFoundError: si el checkpoint no existe.
        ValueError: si el catalogo no es utilizable o el split no es valido.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint '{ckpt}'. Entrena primero: `python -m src.custom`."
        )

    nombre_split = str(split).lower()
    if nombre_split not in SPLITS_VALIDOS:
        raise ValueError(
            f"split invalido: {split!r}. Validos: {', '.join(SPLITS_VALIDOS)}."
        )

    cfg_efectiva = _cfg_para_checkpoint(ckpt, cfg)
    catalogo = _catalogo_json(ckpt)
    fraccion = float(catalogo.get("val_fraccion", VAL_FRACCION_POR_DEFECTO))
    semilla = catalogo.get("semilla")
    semilla = None if semilla is None else int(semilla)

    def _cargador(config: Config, paquete: dict):
        """El catalogo propio no tiene test aparte: 'test' usa las fotos de val."""
        return paquete["train"] if nombre_split == "train" else paquete["val"]

    paquete = preparar_loaders(
        cfg_efectiva, ruta_fotos=ruta_fotos, val_fraccion=fraccion, semilla=semilla
    )
    cargador = _cargador(cfg_efectiva, paquete)

    try:
        return evaluate(ckpt, cfg_efectiva, split=nombre_split, dataloader=cargador)
    except Exception as exc:  # noqa: BLE001 - se decide con _parece_fallo_de_workers
        workers = int(getattr(cfg_efectiva, "num_workers", 0) or 0)
        if workers <= 0 or not _parece_fallo_de_workers(exc):
            raise
        aviso = (
            f"El DataLoader con num_workers={workers} fallo "
            f"({type(exc).__name__}: {exc}). Se reintenta una sola vez con "
            "num_workers=0 (en Windows los workers de multiprocessing necesitan "
            "named pipes; en Colab/Linux los workers funcionan)."
        )
        print("[custom] aviso: " + aviso)
        warnings.warn(aviso, RuntimeWarning, stacklevel=2)

        cfg_sin_workers = replace(cfg_efectiva, num_workers=0)
        paquete = preparar_loaders(
            cfg_sin_workers, ruta_fotos=ruta_fotos, val_fraccion=fraccion, semilla=semilla
        )
        return evaluate(
            ckpt,
            cfg_sin_workers,
            split=nombre_split,
            dataloader=_cargador(cfg_sin_workers, paquete),
        )


# --------------------------------------------------------------------------- #
# CLI  (python -m src.custom --help)
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    """Argumentos de ``python -m src.custom``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.custom",
        description=(
            "Entrena (o evalua) un clasificador con TUS flores, en un catalogo "
            "aparte del de las 102 especies de Oxford. Las fotos van en "
            f"'{RUTA_FOTOS_POR_DEFECTO}/<especie>/<foto>'."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python -m src.custom --revisar\n"
            "  python -m src.custom --fotos mis_flores --epochs 15\n"
            "  python -m src.custom --evaluar resultados/custom/best.pt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fotos",
        type=str,
        default=RUTA_FOTOS_POR_DEFECTO,
        help=f"carpeta con una subcarpeta por especie (defecto: {RUTA_FOTOS_POR_DEFECTO})",
    )
    parser.add_argument("--epochs", type=int, default=None, help="numero de epocas")
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size", help="tamano de lote"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="efficientnet_b0 | resnet50 | mobilenet_v3_large",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        dest="out_dir",
        help=f"carpeta de resultados (defecto: {OUT_DIR_POR_DEFECTO})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="dispositivo (defecto: auto)",
    )
    parser.add_argument(
        "--val-fraccion",
        type=float,
        default=VAL_FRACCION_POR_DEFECTO,
        dest="val_fraccion",
        help=f"fraccion de cada especie para validacion (defecto: {VAL_FRACCION_POR_DEFECTO})",
    )
    parser.add_argument("--semilla", type=int, default=None, help="semilla del reparto y del entrenamiento")
    parser.add_argument("--debug", action="store_true", help="corrida corta (smoke test)")
    parser.add_argument(
        "--revisar",
        action="store_true",
        help="solo revisa la carpeta de fotos e informa (no entrena)",
    )
    parser.add_argument(
        "--evaluar",
        type=str,
        default=None,
        metavar="RUTA_CHECKPOINT",
        help="evalua un checkpoint propio sobre las fotos de validacion",
    )
    return parser


def _imprimir_informe(informe: dict) -> None:
    """Imprime el informe de :func:`revisar_carpeta` en texto legible."""
    print("=" * 72)
    print(f"Revision de flores propias: {informe['ruta']}")
    print(f"  estado        : {'OK' if informe['ok'] else 'CON PROBLEMAS'}")
    print(f"  especies      : {len([n for n, c in informe['clases'].items() if c > 0])}")
    print(f"  total de fotos: {informe['total']}")
    print(f"  min_por_clase : {informe['min_por_clase']}")
    print("  fotos por especie:")
    if informe["clases"]:
        for nombre in sorted(informe["clases"]):
            print(f"    - {nombre}: {informe['clases'][nombre]}")
    else:
        print("    (ninguna)")
    if informe["avisos"]:
        print("  avisos:")
        for aviso in informe["avisos"]:
            print(f"    - {aviso}")
    if informe["problemas"]:
        print("  problemas:")
        for problema in informe["problemas"]:
            print(f"    - {problema}")
    print(f"  recomendacion : {informe['recomendacion']}")
    print("=" * 72)


def _cfg_desde_args(args: argparse.Namespace) -> Config:
    """``Config`` del CLI: los valores propios del modulo + lo que se informe.

    Se parte de los mismos valores que usa :func:`entrenar_custom` cuando no recibe
    ``Config`` (``epochs=15``, ``batch_size=16``, ``efficientnet_b0``,
    ``pretrained=True``) para que ``python -m src.custom`` sin argumentos entrene
    exactamente con los valores documentados. No cambia ningun default de
    ``src/config.py``.
    """
    overrides: dict[str, Any] = {
        "epochs": EPOCHS_POR_DEFECTO,
        "batch_size": BATCH_POR_DEFECTO,
        "backbone": BACKBONE_POR_DEFECTO,
        "pretrained": PRETRAINED_POR_DEFECTO,
        "debug": bool(args.debug),
    }
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.backbone is not None:
        overrides["backbone"] = str(args.backbone)
    if args.device is not None:
        overrides["device"] = str(args.device)
    if args.semilla is not None:
        overrides["seed"] = int(args.semilla)
    return Config(**overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del CLI. Devuelve el codigo de salida.

    * ``--revisar``: imprime el informe y sale con 0 (o 1 si ``ok`` es ``False``).
    * ``--evaluar RUTA``: evalua ese checkpoint propio y sale con 0.
    * sin argumentos: entrena con ``mis_flores/`` y deja el modelo en
      ``resultados/custom/``.
    """
    args = build_argparser().parse_args(argv)

    if args.revisar:
        informe = revisar_carpeta(args.fotos)
        _imprimir_informe(informe)
        return 0 if informe["ok"] else 1

    # Igual que `src/train.py`: un --debug sin --out-dir no debe pisar los
    # resultados definitivos del catalogo propio.
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT_DIR_DEBUG_POR_DEFECTO if args.debug else OUT_DIR_POR_DEFECTO
        if args.debug:
            print(
                f"Aviso: --debug sin --out-dir -> la salida va a '{out_dir}' para no "
                f"pisar los resultados definitivos de '{OUT_DIR_POR_DEFECTO}'."
            )

    cfg = _cfg_desde_args(args)

    try:
        if args.evaluar:
            cfg_eval = replace(cfg, out_dir=str(out_dir), pretrained=False)
            metricas = evaluar_custom(
                args.evaluar, cfg_eval, ruta_fotos=args.fotos, split="val"
            )
            print(
                f"[custom] top1={metricas['top1']:.4f} top5={metricas['top5']:.4f} "
                f"macro_f1={metricas['macro_f1']:.4f}"
            )
            return 0

        entrenar_custom(
            cfg,
            ruta_fotos=args.fotos,
            val_fraccion=float(args.val_fraccion),
            out_dir=out_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - entrada del CLI
    raise SystemExit(main())
