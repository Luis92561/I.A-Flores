"""Inferencia (demo y CLI) sobre un checkpoint entrenado de Flowers-102.

CONTRATO (seccion 3 del contrato interno, propietario SA-06):

    predict_image(image_path, checkpoint_path, topk=5,
                  min_conf=0.25, device="auto") -> list[tuple[str, float]]
    predict_batch(image_paths, checkpoint_path, topk=5,
                  device="auto") -> list[list[tuple[str, float]]]
    # CLI: python -m src.predict --image ruta.jpg --checkpoint resultados/best.pt
    #                           --topk 5 [--min-conf 0.25] [--device auto|cuda|cpu]

Notas de diseno:
  - ``image_path`` acepta ``str``/``Path``, ``PIL.Image.Image``, ``numpy.ndarray``,
    dicts con ``path``/``name``/``data`` y la tupla/valor que devuelve Gradio.
  - El modelo se cachea con :func:`functools.lru_cache` por
    ``(checkpoint_path, device)``: la demo de Gradio no puede releer 20 MB en cada
    clic. :func:`limpiar_cache` vacia esa cache (tests y reentrenamientos).
  - Si ``max(prob) < min_conf`` se devuelve
    ``[("no reconocida / fuera del catálogo", max_prob)]``.
  - Si el checkpoint no existe se lanza ``FileNotFoundError`` con el recordatorio de
    entrenar primero (``python -m src.train``).
  - El ``img_size`` de los transforms viene del propio checkpoint: no se asume 224.
  - CPU y GPU por igual: el device se resuelve con :func:`src.utils.resolve_device`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:  # import relativo (paquete `src`) con respaldo absoluto (cwd en sys.path)
    from .config import Config
    from .data import build_transforms
    from .model import load_checkpoint
    from .utils import resolve_device
except ImportError:  # pragma: no cover - depende de como se importe el modulo
    from src.config import Config
    from src.data import build_transforms
    from src.model import load_checkpoint
    from src.utils import resolve_device

__all__ = ["predict_image", "predict_batch", "limpiar_cache", "main", "NO_RECONOCIDA"]

#: Etiqueta que se devuelve cuando la confianza no llega a ``min_conf``.
#: NOTA: es el unico texto acentuado del proyecto (el contrato interno seccion 5 pide el
#: codigo sin acentos, pero la tarea de SA-06 fija esta cadena literal). Si se
#: cambia aqui hay que cambiarla tambien en ``tests/test_eval_predict.py`` y en
#: ``src/app.py`` (SA-07), que debe importar esta constante en vez de copiarla.
NO_RECONOCIDA = "no reconocida / fuera del catálogo"
#: Confianza minima por defecto de ``predict_batch`` (la firma no expone ``min_conf``).
DEFAULT_MIN_CONF: float = 0.25
#: Tamano de la cache de modelos (checkpoint, device).
_CACHE_SIZE: int = 8


def _mensaje_entrena(checkpoint_path: Path) -> str:
    return (
        f"No existe el checkpoint '{checkpoint_path}'. Para generarlo, entrena primero: "
        "`python -m src.train` (comando: python -m src.train)."
    )


# --------------------------------------------------------------------------- #
# Cache del modelo
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=_CACHE_SIZE)
def _modelo_cacheado(checkpoint_path: str, device: str):
    """Carga (una sola vez por ``checkpoint_path`` + ``device``) el modelo y sus metadatos.

    Returns:
        ``(modelo, nombres_de_clase, img_size, torch.device)``.
    """
    torch_device = resolve_device(device)
    modelo, class_names, meta = load_checkpoint(checkpoint_path, device=torch_device)
    img_size = int(meta.get("img_size") or getattr(modelo, "img_size", 224) or 224)
    return modelo, tuple(str(n) for n in class_names), img_size, torch_device


def limpiar_cache() -> None:
    """Vacia la cache de modelos cargados (util en tests y al reentrenar)."""
    _modelo_cacheado.cache_clear()


def _nombres_ajustados(class_names: Sequence[str], n_clases: int) -> list[str]:
    """``n_clases`` nombres (recorta o rellena si el checkpoint no cuadra)."""
    nombres = [str(nombre) for nombre in class_names]
    if len(nombres) < n_clases:
        nombres += [f"clase_{i + 1:03d}" for i in range(len(nombres), n_clases)]
    return nombres[:n_clases]


# --------------------------------------------------------------------------- #
# Normalizacion de la entrada
# --------------------------------------------------------------------------- #
def _a_uint8(arreglo: np.ndarray) -> np.ndarray:
    """Convierte cualquier arreglo de imagen a ``uint8`` en rango 0..255."""
    if arreglo.dtype == np.uint8:
        return arreglo
    datos = arreglo.astype(np.float32)
    if np.issubdtype(arreglo.dtype, np.floating):
        maximo = float(np.nanmax(datos)) if datos.size else 0.0
        if maximo <= 1.0 + 1e-6:  # imagenes float normalizadas a [0, 1]
            datos = datos * 255.0
    return np.clip(datos, 0.0, 255.0).round().astype(np.uint8)


def _ndarray_a_pil(arreglo: np.ndarray):
    """``numpy.ndarray`` (H,W), (H,W,C) o (C,H,W) -> ``PIL.Image`` RGB."""
    from PIL import Image

    datos = np.asarray(arreglo)
    if datos.ndim == 2:  # escala de grises
        return Image.fromarray(_a_uint8(datos)).convert("RGB")
    if datos.ndim == 3 and datos.shape[0] in (1, 3, 4) and datos.shape[0] < datos.shape[-1]:
        datos = np.transpose(datos, (1, 2, 0))  # channel-first -> channel-last
    if datos.ndim != 3 or datos.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"no se pudo interpretar un arreglo de forma {datos.shape!r} como imagen "
            "(se esperaba (H,W), (H,W,3) o (3,H,W))."
        )
    return Image.fromarray(_a_uint8(datos)).convert("RGB")


def _a_pil(imagen):
    """Normaliza cualquiera de las entradas aceptadas a ``PIL.Image`` RGB.

    Acepta ``str``/``Path`` (ruta de archivo), ``PIL.Image.Image``, ``numpy.ndarray``,
    ``dict`` con ``path``/``name``/``data``/``image`` y tuplas/listas (por ejemplo
    ``(imagen, ruta)`` de Gradio).
    """
    from PIL import Image

    if imagen is None:
        raise ValueError("no se recibio ninguna imagen (imagen=None).")

    if isinstance(imagen, Image.Image):
        return imagen.convert("RGB")

    if isinstance(imagen, np.ndarray):
        return _ndarray_a_pil(imagen)

    if isinstance(imagen, (str, Path)):
        ruta = Path(imagen)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe la imagen '{ruta}'.")
        with Image.open(ruta) as abierta:
            return abierta.convert("RGB")

    if isinstance(imagen, dict):
        for clave in ("path", "name", "data", "image", "orig_name"):
            if clave in imagen and imagen[clave] is not None:
                return _a_pil(imagen[clave])
        raise ValueError(
            f"el diccionario recibido no tiene ninguna clave de imagen util "
            f"(claves: {sorted(imagen)})."
        )

    if isinstance(imagen, (tuple, list)):
        error: Exception | None = None
        for elemento in imagen:
            try:
                return _a_pil(elemento)
            except (TypeError, ValueError, FileNotFoundError) as exc:
                error = exc
        raise ValueError(
            f"no se pudo interpretar la tupla/lista recibida como imagen: {error}"
        )

    # Objetos tipo Gradio (FileData, NamedString, ...): intenta atributos de ruta.
    for atributo in ("path", "name", "orig_name", "value"):
        valor = getattr(imagen, atributo, None)
        if isinstance(valor, (str, Path, np.ndarray)):
            return _a_pil(valor)
    if hasattr(imagen, "convert"):  # cualquier cosa compatible con PIL
        return imagen.convert("RGB")

    raise TypeError(
        f"tipo de imagen no soportado: {type(imagen)!r}. Usa str, Path, PIL.Image, "
        "numpy.ndarray o el valor de Gradio."
    )


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def predict_image(
    image_path,
    checkpoint_path,
    topk: int = 5,
    min_conf: float = 0.25,
    device: str = "auto",
) -> list[tuple[str, float]]:
    """Clasifica una imagen y devuelve ``[(clase, probabilidad), ...]``.

    Args:
        image_path: ``str``/``Path``, ``PIL.Image.Image``, ``numpy.ndarray``, dict con
            ``path``/``name``/``data`` o la tupla/valor que devuelve Gradio.
        checkpoint_path: ruta del ``.pt`` entrenado.
        topk: numero de clases a devolver (se recorta a ``num_classes`` si es mayor).
        min_conf: confianza minima. Si ``max(prob) < min_conf`` se devuelve
            ``[("no reconocida / fuera del catálogo", max_prob)]``.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` (se resuelve con ``resolve_device``).

    Returns:
        Lista de ``(nombre_de_clase, probabilidad)`` con probabilidades **decrecientes**
        en ``[0, 1]`` (softmax). El modelo y los nombres de clase se leen de una cache
        ``lru_cache`` por ``(checkpoint_path, device)``.

    Raises:
        FileNotFoundError: si no existe el checkpoint (entrena primero) o la imagen.
        ValueError: si la imagen no se puede interpretar.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(_mensaje_entrena(ckpt))

    imagen = _a_pil(image_path)

    modelo, nombres_ckpt, img_size, device_real = _modelo_cacheado(str(ckpt), str(device))

    # Config auxiliar: solo aporta img_size (y debug para no crear workers) a los
    # transforms; el modelo NO se construye aqui (viene del checkpoint).
    cfg_inferencia = Config(
        num_classes=max(1, len(nombres_ckpt)),
        img_size=int(img_size),
        pretrained=False,
    )
    transform = build_transforms(cfg_inferencia, train=False)
    tensor = transform(imagen).unsqueeze(0).to(device_real)

    with torch.no_grad():
        logits = modelo(tensor)
    probabilidades = torch.softmax(logits.detach().float(), dim=1).flatten().cpu().numpy()

    n_clases = int(probabilidades.shape[0])
    nombres = _nombres_ajustados(nombres_ckpt, n_clases)
    mejor = float(probabilidades.max()) if n_clases else 0.0

    if mejor < float(min_conf):
        return [(NO_RECONOCIDA, mejor)]

    if n_clases == 0:
        return []

    k = max(1, min(int(topk), n_clases))
    indices = np.argsort(-probabilidades, kind="stable")[:k]
    return [(nombres[int(i)], float(probabilidades[int(i)])) for i in indices]


def predict_batch(
    image_paths,
    checkpoint_path,
    topk: int = 5,
    device: str = "auto",
) -> list[list[tuple[str, float]]]:
    """``predict_image`` sobre varias imagenes (el modelo se carga una sola vez).

    Args:
        image_paths: iterable de imagenes aceptadas por :func:`predict_image`.
        checkpoint_path: ruta del ``.pt`` entrenado.
        topk: numero de clases por imagen.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.

    Returns:
        Una lista de resultados (mismo orden que ``image_paths``); usa el ``min_conf``
        por defecto de 0.25 y por tanto algunas entradas pueden ser
        ``[("no reconocida / fuera del catálogo", prob)]``.
    """
    return [
        predict_image(imagen, checkpoint_path, topk=topk, min_conf=DEFAULT_MIN_CONF, device=device)
        for imagen in image_paths
    ]


# --------------------------------------------------------------------------- #
# CLI  (python -m src.predict --help)
# --------------------------------------------------------------------------- #
def build_parser():
    """Parser de la CLI congelada de la seccion 3.b."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Clasifica una imagen de flor con un checkpoint entrenado.",
    )
    parser.add_argument("--image", required=True, help="ruta de la imagen a clasificar")
    parser.add_argument(
        "--checkpoint",
        default="resultados/best.pt",
        help="ruta del checkpoint (por defecto: resultados/best.pt)",
    )
    parser.add_argument("--topk", type=int, default=5, help="numero de clases a mostrar (defecto: 5)")
    parser.add_argument(
        "--min-conf",
        dest="min_conf",
        type=float,
        default=0.25,
        help="confianza minima; por debajo se responde 'no reconocida' (defecto: 0.25)",
    )
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu (por defecto: auto)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entrada de la CLI. Devuelve 0 si todo fue bien, 1 si falta el checkpoint."""
    args = build_parser().parse_args(argv)

    try:
        resultados = predict_image(
            args.image,
            args.checkpoint,
            topk=args.topk,
            min_conf=args.min_conf,
            device=args.device,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        if not Path(args.checkpoint).is_file():
            print("[error] Entrena primero: python -m src.train")
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1

    print(f"imagen:     {args.image}")
    print(f"checkpoint: {args.checkpoint}")
    for posicion, (nombre, probabilidad) in enumerate(resultados, start=1):
        print(f"{posicion:2d}. {nombre:<42s} {probabilidad * 100:6.2f}%")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada de la CLI
    raise SystemExit(main())
