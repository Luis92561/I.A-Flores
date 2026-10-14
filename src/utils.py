"""Utilidades compartidas (semillas, dispositivo, JSON, conteo de parametros).

CONTRATO (Fase 0, propietario SA-00):
  - Firmas congeladas: todos los modulos pueden asumirlas.
  - Los subagentes NO deben modificar este archivo.
  - Las importaciones de torch son perezosas (dentro de la funcion) para que
    `load_json` / `save_json` funcionen sin torch instalado.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

__all__ = [
    "set_seed",
    "resolve_device",
    "save_json",
    "load_json",
    "ensure_dir",
    "count_params",
    "describe_device",
]


def set_seed(seed: int = 42) -> None:
    """Fija la semilla en random, numpy y torch (CPU y CUDA) para reproducibilidad."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - entorno sin torch
        pass


def resolve_device(spec: str = "auto"):
    """Devuelve un torch.device a partir de 'auto' | 'cuda' | 'cpu'.

    'auto' usa CUDA si esta disponible y si no cae a CPU (Colab sin GPU incluido).
    Si se pide 'cuda' y no hay GPU, lanza RuntimeError con un mensaje claro.
    """
    import torch

    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Se pidio device='cuda' pero torch.cuda.is_available() es False. "
            "En Colab: Entorno de ejecucion > Cambiar tipo de entorno > GPU T4."
        )
    return torch.device(spec)


def describe_device(device) -> str:
    """Texto legible para logs: 'cuda (NVIDIA T4)' o 'cpu'."""
    import torch

    if getattr(device, "type", str(device)) == "cuda":
        idx = getattr(device, "index", None) or 0
        return f"cuda ({torch.cuda.get_device_name(idx)})"
    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    """Guarda JSON serializable (convierte Path y tipos numpy/torch basicos)."""
    p = Path(path)
    ensure_dir(p.parent)

    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "item"):  # escalares numpy / torch
            return o.item()
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(f"Objeto no serializable: {type(o)!r}")

    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)


def load_json(path: str | Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_params(model) -> int:
    """Numero de parametros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
