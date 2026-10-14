"""Datos de Oxford Flowers-102: transforms, DataLoaders y nombres de clase.

CONTRATO (Firmas congeladas, propietario SA-01 — no cambiar):
    build_transforms(cfg: Config, train: bool) -> Callable
    build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]
    get_class_names(cfg: Config) -> list[str]

Notas de diseno:
  - Las importaciones de ``torch`` / ``torchvision`` son perezosas (dentro de las
    funciones). Asi ``import src.data`` no arrastra torch ni scipy, y modulos como
    ``predict.py`` pueden reutilizar ``build_transforms`` sin cargar el dataset.
    Ademas, ``torchvision.datasets.Flowers102`` necesita ``scipy`` (lee los .mat);
    el helper :func:`_flowers102_class` convierte ese ImportError en un mensaje util.
  - Nada de rutas absolutas ni de ``.cuda()``: este modulo no necesita device,
    ``pin_memory`` se activa solo si ``torch.cuda.is_available()``.
  - Pitfall documentado en el contrato interno §3: las etiquetas originales de Flowers-102
    son 1-based (1..102) y torchvision resta 1; las claves de ``cat_to_name.json``
    son ``"1".."102"``, por eso se mapea ``class_names[i] = cat[str(i + 1)]``.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from src.config import MEAN, STD, Config

if TYPE_CHECKING:  # solo para anotaciones: mantiene el import de torch perezoso
    from torch.utils.data import DataLoader

__all__ = [
    "build_transforms",
    "build_dataloaders",
    "get_class_names",
    "CAT_TO_NAME_URL",
    "CAT_TO_NAME_FILENAME",
]

#: URL oficial (Oxford VGG) del mapa id de categoria -> nombre de especie.
CAT_TO_NAME_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/cat_to_name.json"
#: Nombre del archivo de cache dentro de ``cfg.data_root``.
CAT_TO_NAME_FILENAME = "cat_to_name.json"
#: Respaldo offline versionado en el repo (claves 0-based "0".."101").
ASSET_CAT_TO_NAME = Path(__file__).resolve().parent.parent / "assets" / "cat_to_name.json"
#: Numero de clases de Flowers-102 (por si cfg no lo trae).
DEFAULT_NUM_CLASSES = 102
#: Splits oficiales, en el orden que espera el contrato.
SPLITS: tuple[str, str, str] = ("train", "val", "test")
#: Timeout de red (segundos) para la descarga de cat_to_name.json.
URL_TIMEOUT_S = 20.0

#: Nombres genericos tipo "class_0", "n12", "label_3"... no son nombres de especie.
_GENERIC_NAME_RE = re.compile(
    r"^(?:class|clase|label|etiqueta|category|categoria|species|especie|"
    r"n|img|image|idx|index)[_\- ]?\d+$",
    re.IGNORECASE,
)

#: Raices de datos donde ya se intento (y fallo) la descarga: evita reintentos lentos.
_REMOTE_ATTEMPTED: set[str] = set()


# --------------------------------------------------------------- import perezoso
def _torch():
    """Devuelve el modulo ``torch`` (import perezoso para no cargarlo al importar)."""
    import torch

    return torch


def _transforms():
    """Devuelve ``torchvision.transforms`` con un mensaje claro si falta scipy."""
    try:
        from torchvision import transforms
    except ImportError as exc:  # torchvision.datasets importa scipy.io de forma eager
        if "scipy" in str(exc).lower():
            raise ImportError(
                "torchvision no pudo importarse porque falta scipy "
                "(Flowers102 lee imagelabels.mat con scipy.io). Instala: pip install scipy"
            ) from exc
        raise
    return transforms


def _flowers102_class():
    """Devuelve la clase ``torchvision.datasets.Flowers102`` (import perezoso)."""
    try:
        from torchvision.datasets import Flowers102
    except ImportError as exc:
        if "scipy" in str(exc).lower():
            raise ImportError(
                "Flowers102 necesita scipy para leer imagelabels.mat / setid.mat. "
                "Instala: pip install scipy"
            ) from exc
        raise
    return Flowers102


# ------------------------------------------------------------------- transforms
def build_transforms(cfg: Config, train: bool) -> Callable:
    """Pipeline de preprocesado de una imagen.

    ``train=True``:  RandomResizedCrop(img_size, scale=(0.6, 1.0)) + flip horizontal
                     + RandAugment (o ColorJitter si RandAugment no existe)
                     + ToTensor + Normalize(MEAN, STD).
    ``train=False``: Resize(int(img_size * 1.14)) + CenterCrop(img_size)
                     + ToTensor + Normalize(MEAN, STD)  (determinista).

    Es valido para cualquier ``cfg.img_size`` (no asumir 224).
    """
    transforms = _transforms()
    img_size = int(cfg.img_size)
    if img_size <= 0:
        raise ValueError(f"cfg.img_size debe ser positivo, se recibio {cfg.img_size!r}")

    ops: list[Any] = []
    if train:
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)))
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        rand_augment = getattr(transforms, "RandAugment", None)
        if rand_augment is not None:
            ops.append(rand_augment(num_ops=2, magnitude=9))
        else:  # torchvision < 0.12: no hay RandAugment
            ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    else:
        ops.append(transforms.Resize(int(img_size * 1.14)))  # 224 -> 255
        ops.append(transforms.CenterCrop(img_size))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(ops)


# ------------------------------------------------------- etiquetas del dataset
def _sample_indices(total: int, k: int) -> list[int]:
    """Hasta ``k`` indices repartidos por todo el rango [0, total)."""
    if total <= 0 or k <= 0:
        return []
    if total <= k:
        return list(range(total))
    step = max(1, total // k)
    return list(range(0, total, step))[:k]


def _targets_from_attributes(dataset) -> list[int] | None:
    """Intenta leer las etiquetas de atributos conocidos (sin decodificar imagenes)."""
    for attr in ("targets", "labels", "_labels"):
        value = getattr(dataset, attr, None)
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            seq = list(value)
        except TypeError:
            continue
        if len(seq) != len(dataset):
            continue
        try:
            return [int(t) for t in seq]
        except (TypeError, ValueError):
            continue
    return None


def _targets_by_scan(dataset, indices: Sequence[int] | None = None) -> list[int]:
    """Etiquetas leyendo ``dataset[i][1]`` (lento: decodifica imagenes)."""
    if indices is None:
        indices = range(len(dataset))
    return [int(dataset[i][1]) for i in indices]


def dataset_targets(dataset, verify: int = 5) -> list[int]:
    """Etiquetas enteras de ``dataset``, alineadas con ``dataset[i]``.

    Usa primero los atributos internos de torchvision (rapido). Como el formato de
    ``_labels`` cambio entre versiones (1-based vs 0-based), comprueba unas pocas
    muestras reales y, si no coinciden, hace un escaneo completo.
    """
    targets = _targets_from_attributes(dataset)
    if targets is None:
        return _targets_by_scan(dataset)
    for idx in _sample_indices(len(dataset), verify):
        try:
            real = int(dataset[idx][1])
        except Exception:  # no se pudo verificar (p. ej. imagen corrupta): confiamos
            return targets
        if real != targets[idx]:
            return _targets_by_scan(dataset)
    return targets


def _debug_subset(dataset, per_class: int, seed: int):
    """Subconjunto reproducible de ``per_class`` imagenes por clase.

    Mantiene la correspondencia etiqueta <-> nombre porque solo selecciona indices
    del dataset original (``Subset`` conserva los targets).
    """
    from torch.utils.data import Subset

    per_class = int(per_class)
    if per_class <= 0:
        raise ValueError(f"cfg.debug_per_class debe ser >= 1, se recibio {per_class!r}")

    targets = dataset_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(int(seed))
    selected: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.extend(sorted(indices[:per_class]))
    selected.sort()

    if not selected:  # dataset vacio: no fallar, devolver un Subset vacio
        return Subset(dataset, [])
    return Subset(dataset, selected)


# ------------------------------------------------------------ nombres de clase
def _looks_generic(name: str) -> bool:
    """True si el nombre no aporta informacion de especie ('class_7', '12', ...)."""
    text = str(name).strip()
    if not text:
        return True
    if text.isdigit():
        return True
    return bool(_GENERIC_NAME_RE.match(text))


def _names_from_dataset(dataset, num_classes: int = DEFAULT_NUM_CLASSES) -> list[str] | None:
    """Nombres propios del dataset si existen y NO son genericos; si no, ``None``."""
    if dataset is None:
        return None

    candidates: list[list[Any]] = []
    for attr in ("classes", "class_names", "_class_names", "categories"):
        value = getattr(dataset, attr, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates.append(list(value))
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and class_to_idx:
        candidates.append([name for name, _ in sorted(class_to_idx.items(), key=lambda kv: int(kv[1]))])

    for candidate in candidates:
        names = [str(x).strip() for x in candidate]
        if len(names) != int(num_classes):
            continue
        if any(_looks_generic(n) for n in names):
            continue
        return names
    return None


def _names_from_mapping(mapping: Any, num_classes: int) -> list[str] | None:
    """Mapea un JSON id->nombre a una lista ordenada por etiqueta 0-based.

    Se aceptan las dos convenciones vistas en la practica:
      - Oxford ``cat_to_name.json``: claves ``"1".."102"`` (1-based), por eso se
        prueba ``mapping[str(i + 1)]`` (pitfall documentado en el contrato interno §3).
      - ``assets/cat_to_name.json`` del repo: claves ``"0".."101"`` (0-based).
    """
    if not isinstance(mapping, dict) or not mapping:
        return None
    for offset in (1, 0):  # primero 1-based (contrato), luego 0-based (asset)
        names: list[str] = []
        for i in range(int(num_classes)):
            value = mapping.get(str(i + offset))
            if value is None:
                value = mapping.get(i + offset)
            if not isinstance(value, str) or not value.strip():
                break
            names.append(value.strip())
        else:
            return names
    return None


def _names_from_cache(cfg: Config) -> list[str] | None:
    """Lee ``<data_root>/cat_to_name.json`` si existe y cubre las 102 clases."""
    path = Path(cfg.data_root) / CAT_TO_NAME_FILENAME
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    except (OSError, ValueError):
        return None
    return _names_from_mapping(mapping, int(cfg.num_classes))


def _names_from_asset(num_classes: int) -> list[str] | None:
    """Respaldo offline versionado en el repo: ``assets/cat_to_name.json``.

    Es preferible a la descarga de red: no necesita internet y esta verificado.
    """
    path = Path(ASSET_CAT_TO_NAME)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    except (OSError, ValueError):
        return None
    return _names_from_mapping(mapping, int(num_classes))


def _download_cat_to_name(cfg: Config) -> list[str] | None:
    """Descarga (una sola vez por data_root) y cachea cat_to_name.json.

    Nunca lanza excepcion: sin internet devuelve ``None`` para que el llamador
    use los nombres de relleno.
    """
    root = str(Path(cfg.data_root).resolve())
    if root in _REMOTE_ATTEMPTED:
        return None
    _REMOTE_ATTEMPTED.add(root)

    try:
        with urllib.request.urlopen(CAT_TO_NAME_URL, timeout=URL_TIMEOUT_S) as response:
            payload = response.read()
        mapping = json.loads(payload.decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return None

    names = _names_from_mapping(mapping, int(cfg.num_classes))
    if names is None:
        return None

    try:  # el cache es best-effort: si no se puede escribir, seguimos igual
        cache = Path(cfg.data_root) / CAT_TO_NAME_FILENAME
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return names


def _fallback_names(num_classes: int) -> list[str]:
    """Nombres de relleno: ['especie_001', ..., 'especie_102']."""
    return [f"especie_{i + 1:03d}" for i in range(int(num_classes))]


def _resolve_class_names(cfg: Config, dataset=None) -> list[str]:
    """Nombres de clase ordenados por etiqueta 0-based, con degradacion elegante."""
    num_classes = int(cfg.num_classes)

    if dataset is not None:
        names = _names_from_dataset(dataset, num_classes)
        if names:
            return names
    else:
        try:  # 1a) `Flowers102.classes`: atributo de CLASE, sin instanciar ni red
            Flowers102 = _flowers102_class()
        except Exception:
            Flowers102 = None
        names = _names_from_dataset(Flowers102, num_classes)
        if names:
            return names

        try:  # 1b) instancia real, por si esa version expone los nombres de otra forma
            opened = (
                Flowers102(root=str(cfg.data_root), split="train", download=False) if Flowers102 is not None else None
            )
        except Exception:
            opened = None
        names = _names_from_dataset(opened, num_classes)
        if names:
            return names

    names = _names_from_cache(cfg)  # 2) cache local (offline-friendly)
    if names:
        return names

    names = _names_from_asset(num_classes)  # 3) respaldo offline versionado en el repo
    if names:
        return names

    names = _download_cat_to_name(cfg)  # 4) descarga de Oxford + cache
    if names:
        return names

    return _fallback_names(num_classes)  # 5) nunca fallar


def get_class_names(cfg: Config) -> list[str]:
    """102 nombres legibles en orden de etiqueta (``class_names[i]`` <-> etiqueta ``i``).

    Orden de intentos:
      1. nombres propios del dataset (``torchvision.datasets.Flowers102.classes``,
         es un atributo de clase: funciona sin datos y sin internet);
      2. cache ``<data_root>/cat_to_name.json``;
      3. respaldo offline ``assets/cat_to_name.json`` del repo (claves 0-based);
      4. descarga desde Oxford (se cachea en ``<data_root>/cat_to_name.json``);
      5. ``['especie_001', ..., 'especie_102']``.

    Nunca lanza excepcion por falta de datos o de internet.
    """
    return _resolve_class_names(cfg, dataset=None)


# ---------------------------------------------------------------- data loaders
def build_dataloaders(cfg: Config) -> tuple["DataLoader", "DataLoader", "DataLoader", list[str]]:
    """DataLoaders oficiales de Flowers-102 + nombres de clase.

    Devuelve ``(train_dl, val_dl, test_dl, class_names)``. Splits oficiales
    (1020 / 1020 / 6149); en ``cfg.debug`` se usan subconjuntos de
    ``cfg.debug_per_class`` imagenes por clase.
    """
    from torch.utils.data import DataLoader

    Flowers102 = _flowers102_class()
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)

    # pin_memory solo cuando hay CUDA (en CPU no aporta y consume RAM).
    pin_memory = bool(_torch().cuda.is_available())
    num_workers = int(cfg.num_workers)

    datasets: dict[str, Any] = {}
    loaders: dict[str, Any] = {}
    for split in SPLITS:
        dataset = Flowers102(
            root=str(root),
            split=split,
            transform=build_transforms(cfg, train=(split == "train")),
            download=True,
        )
        datasets[split] = dataset
        if cfg.debug:
            dataset = _debug_subset(dataset, per_class=cfg.debug_per_class, seed=int(cfg.seed) + 17 * SPLITS.index(split))
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=(split == "train"),  # val/test SIN barajar (contrato §3)
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    class_names = _resolve_class_names(cfg, dataset=datasets["train"])
    return loaders["train"], loaders["val"], loaders["test"], class_names
