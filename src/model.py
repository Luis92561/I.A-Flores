"""Modelo de clasificacion de flores (Flowers-102) con backbones de torchvision.

CONTRATO (seccion 3 de CONTRATOS.md, propietario SA-02):

    class FlowerClassifier(nn.Module):      # atributos OBLIGATORIOS: .backbone y .head
        def forward(self, x) -> Tensor      # (B,3,H,W) -> (B, num_classes) logits

    build_model(num_classes=102, backbone="efficientnet_b0",
                pretrained=True, dropout=0.3) -> FlowerClassifier
    set_backbone_trainable(model, trainable: bool) -> None
    build_optimizer(model, cfg: Config) -> torch.optim.Optimizer
    save_checkpoint(path, model, class_names: list[str], cfg: Config, **extra) -> None
    load_checkpoint(path, device="cpu") -> tuple[FlowerClassifier, list[str], dict]

Notas de diseno:
  - Todo el modulo funciona en CPU y en GPU: no hay ningun ``.cuda()`` duro, el
    device se pasa como argumento (``load_checkpoint(..., device=...)``) o se
    resuelve con ``src.utils.resolve_device`` cuando hace falta.
  - ``pretrained=True`` degrada con elegancia a ``pretrained=False`` si la
    descarga de pesos falla (Colab a veces bloquea ``download.pytorch.org``).
  - No hay rutas absolutas de Windows: las rutas se reciben como argumento.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
from torchvision import models as tv_models

try:  # import relativo (paquete `src`) con respaldo absoluto (cwd en sys.path)
    from .config import FLOWERS102_CLASSES, IMAGE_SIZE, Config
    from .utils import resolve_device
except ImportError:  # pragma: no cover - depende de como se importe el modulo
    from src.config import FLOWERS102_CLASSES, IMAGE_SIZE, Config
    from src.utils import resolve_device

__all__ = [
    "SUPPORTED_BACKBONES",
    "CHECKPOINT_KEYS",
    "FlowerClassifier",
    "build_model",
    "set_backbone_trainable",
    "build_optimizer",
    "save_checkpoint",
    "load_checkpoint",
]

#: Backbones soportados. El primero es el defecto del proyecto.
SUPPORTED_BACKBONES: tuple[str, ...] = (
    "efficientnet_b0",
    "resnet50",
    "mobilenet_v3_large",
)

#: Claves obligatorias del checkpoint (contrato compartido con SA-05 y SA-06).
CHECKPOINT_KEYS: tuple[str, ...] = (
    "model_state",
    "class_names",
    "backbone",
    "num_classes",
    "img_size",
    "config",
    "epoch",
    "val_acc",
)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class FlowerClassifier(nn.Module):
    """Clasificador de flores = extractor de caracteristicas (``backbone``) + ``head``.

    Los atributos ``.backbone`` y ``.head`` son obligatorios por contrato: SA-05
    los usa para congelar/descongelar y para armar los grupos de parametros, y
    SA-06 / SA-07 pueden inspeccionarlos (Grad-CAM, conteo de parametros, etc.).

    Args:
        backbone: red convolucional que devuelve un mapa (B, C, H, W).
        in_features: numero de canales de salida del backbone.
        num_classes: numero de clases de salida (102 para Flowers-102).
        dropout: probabilidad de dropout antes de la capa lineal final.
        backbone_name: nombre canonico del backbone (para el checkpoint).
        pretrained: si los pesos del backbone son preentrenados (informativo).
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: int = FLOWERS102_CLASSES,
        dropout: float = 0.3,
        backbone_name: str | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(in_features), int(num_classes)),
        )
        # Metadatos utiles (no son parametros, no entran en el state_dict).
        self.backbone_name = backbone_name
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.pretrained = bool(pretrained)

    # ------------------------------------------------------------------ utils
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Vector de caracteristicas (B, in_features) antes de la cabeza lineal."""
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = self.avgpool(feats)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, num_classes) logits (sin softmax)."""
        return self.head(self.forward_features(x))


# --------------------------------------------------------------------------- #
# Construccion de backbones
# --------------------------------------------------------------------------- #
def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Module, int]:
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = tv_models.efficientnet_b0(weights=weights)
    in_features = int(net.classifier[-1].in_features)  # 1280
    return net.features, in_features


def _build_resnet50(pretrained: bool) -> tuple[nn.Module, int]:
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    net = tv_models.resnet50(weights=weights)
    in_features = int(net.fc.in_features)  # 2048
    # Se conserva todo salvo la capa fc; incluye el avgpool global.
    backbone = nn.Sequential(*list(net.children())[:-1])
    return backbone, in_features


def _build_mobilenet_v3_large(pretrained: bool) -> tuple[nn.Module, int]:
    weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    net = tv_models.mobilenet_v3_large(weights=weights)
    in_features = int(net.classifier[0].in_features)  # 960
    return net.features, in_features


_BACKBONE_BUILDERS: dict[str, Callable[[bool], tuple[nn.Module, int]]] = {
    "efficientnet_b0": _build_efficientnet_b0,
    "resnet50": _build_resnet50,
    "mobilenet_v3_large": _build_mobilenet_v3_large,
}


def build_model(
    num_classes: int = FLOWERS102_CLASSES,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> FlowerClassifier:
    """Crea un :class:`FlowerClassifier` con la cabeza adaptada a ``num_classes``.

    Args:
        num_classes: numero de clases de salida.
        backbone: uno de :data:`SUPPORTED_BACKBONES`.
        pretrained: si es ``True`` intenta descargar los pesos ImageNet; si la
            descarga falla (sin internet / Colab bloqueado) avisa por
            ``warnings.warn`` y continua con pesos aleatorios.
        dropout: dropout antes de la capa lineal final.

    Raises:
        ValueError: si ``backbone`` no esta soportado (el mensaje lista los validos).
    """
    if not isinstance(backbone, str) or backbone not in _BACKBONE_BUILDERS:
        soportados = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(
            f"Backbone no soportado: {backbone!r}. Soportados: {soportados}."
        )

    builder = _BACKBONE_BUILDERS[backbone]
    used_pretrained = bool(pretrained)

    try:
        features, in_features = builder(used_pretrained)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de descarga/carga
        if not pretrained:
            # El error es real (no de red): no lo enmascaramos.
            raise
        warnings.warn(
            f"No se pudieron cargar los pesos preentrenados de '{backbone}' "
            f"({type(exc).__name__}: {exc}). Se continua con pretrained=False "
            "(pesos aleatorios); el entrenamiento sigue siendo valido.",
            RuntimeWarning,
            stacklevel=2,
        )
        used_pretrained = False
        features, in_features = builder(False)

    return FlowerClassifier(
        backbone=features,
        in_features=in_features,
        num_classes=num_classes,
        dropout=dropout,
        backbone_name=backbone,
        pretrained=used_pretrained,
    )


# --------------------------------------------------------------------------- #
# Congelar / descongelar
# --------------------------------------------------------------------------- #
def set_backbone_trainable(model: FlowerClassifier | nn.Module, trainable: bool) -> None:
    """Activa o desactiva ``requires_grad`` en **todos** los parametros del backbone.

    Se usa en las primeras epocas (``cfg.freeze_epochs``) para entrenar solo la
    cabeza. No cambia el modo train/eval del backbone.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError(
            "set_backbone_trainable espera un modelo con atributo .backbone "
            f"(recibido {type(model).__name__})."
        )
    for param in backbone.parameters():
        param.requires_grad = bool(trainable)


# --------------------------------------------------------------------------- #
# Optimizador con LR diferencial
# --------------------------------------------------------------------------- #
def build_optimizer(model: FlowerClassifier | nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """``AdamW`` con dos grupos: cabeza con ``cfg.lr`` y backbone con ``cfg.backbone_lr``.

    Ambos grupos usan ``cfg.weight_decay``. El grupo del backbone incluye sus
    parametros aunque esten congelados en el momento de crear el optimizador:
    ``AdamW`` ignora los que tienen ``grad is None`` y asi el mismo optimizador
    sigue sirviendo cuando SA-05 descongela el backbone.
    """
    backbone = getattr(model, "backbone", None)
    head = getattr(model, "head", None)
    if backbone is None or head is None:
        raise AttributeError(
            "build_optimizer espera un modelo con atributos .backbone y .head "
            f"(recibido {type(model).__name__})."
        )

    head_params = [p for p in head.parameters()]
    backbone_params = [p for p in backbone.parameters()]

    param_groups: list[dict[str, Any]] = []
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(cfg.lr),
                "weight_decay": float(cfg.weight_decay),
                "name": "head",
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(cfg.backbone_lr),
                "weight_decay": float(cfg.weight_decay),
                "name": "backbone",
            }
        )
    if not param_groups:
        raise ValueError("El modelo no tiene parametros: no se puede crear el optimizador.")

    return torch.optim.AdamW(param_groups)


# --------------------------------------------------------------------------- #
# Checkpoints (contrato compartido con SA-05 y SA-06)
# --------------------------------------------------------------------------- #
def _model_metadata(model: FlowerClassifier | nn.Module, cfg: Config) -> tuple[str, int]:
    """(nombre del backbone, num_classes) tomados del modelo, con respaldo en cfg."""
    backbone_name = getattr(model, "backbone_name", None) or getattr(cfg, "backbone", None)
    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        head = getattr(model, "head", None)
        num_classes = getattr(head[-1], "out_features", None) if head is not None else None
    if num_classes is None:
        num_classes = getattr(cfg, "num_classes", FLOWERS102_CLASSES)
    return str(backbone_name), int(num_classes)


def save_checkpoint(
    path: str | Path,
    model: FlowerClassifier | nn.Module,
    class_names: Iterable[str],
    cfg: Config,
    **extra: Any,
) -> None:
    """Guarda el checkpoint con el formato congelado del contrato.

    Claves siempre presentes: ``model_state``, ``class_names``, ``backbone``,
    ``num_classes``, ``img_size``, ``config``, ``epoch``, ``val_acc``.
    ``**extra`` permite anadir/sobrescribir cualquier otra clave (por ejemplo
    ``epoch`` y ``val_acc`` que envia SA-05); los ``None`` no se escriben.
    """
    backbone_name, num_classes = _model_metadata(model, cfg)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "class_names": list(class_names),
        "backbone": backbone_name,
        "num_classes": num_classes,
        "img_size": int(getattr(cfg, "img_size", IMAGE_SIZE)),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "epoch": int(extra.pop("epoch", -1)),
        "val_acc": float(extra.pop("val_acc", 0.0)),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[FlowerClassifier, list[str], dict]:
    """Carga un checkpoint y reconstruye el modelo en ``.eval()`` y en ``device``.

    Returns:
        ``(model, class_names, meta)`` donde ``meta`` es el checkpoint sin
        ``model_state`` (incluye ``config``, ``img_size``, ``epoch``, ``val_acc``
        y cualquier clave extra escrita por SA-05).

    Raises:
        FileNotFoundError: si la ruta no existe (entrena primero).
        KeyError: si al checkpoint le faltan claves obligatorias.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint '{target}'. Entrena primero con "
            "`python -m src.train` para generarlo."
        )

    try:
        checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    except TypeError:  # torch muy antiguo sin el argumento weights_only
        checkpoint = torch.load(target, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise KeyError(
            f"El checkpoint '{target}' no tiene el formato esperado "
            f"(claves obligatorias: {', '.join(CHECKPOINT_KEYS)})."
        )

    faltantes = [k for k in CHECKPOINT_KEYS if k not in checkpoint]
    if faltantes:
        raise KeyError(
            f"Al checkpoint '{target}' le faltan claves obligatorias: {faltantes}."
        )

    ckpt_cfg = checkpoint.get("config") or {}
    backbone = str(checkpoint["backbone"])
    num_classes = int(checkpoint["num_classes"])
    dropout = float(ckpt_cfg.get("dropout", 0.3))

    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,  # el state_dict ya trae los pesos: nunca hay descarga
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    if isinstance(device, torch.device):
        torch_device = device
    elif str(device).lower() == "auto":  # comodidad para SA-06 / SA-07
        torch_device = resolve_device("auto")
    else:
        torch_device = torch.device(str(device))
    model.to(torch_device)
    model.eval()

    class_names = [str(name) for name in checkpoint["class_names"]]
    meta = {k: v for k, v in checkpoint.items() if k != "model_state"}
    meta["img_size"] = int(checkpoint["img_size"])
    meta["device"] = str(torch_device)
    return model, class_names, meta
