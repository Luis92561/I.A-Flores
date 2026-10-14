"""Configuracion central del proyecto de reconocimiento de flores.

CONTRATO (Fase 0, propietario SA-00):
  - Este modulo NO importa torch: debe poder importarse en cualquier entorno.
  - Los subagentes NO deben modificar este archivo. Si necesitan un campo nuevo,
    lo reportan al orquestador en su informe final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------- constantes
IMAGE_SIZE: int = 224
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet
FLOWERS102_CLASSES: int = 102


@dataclass
class Config:
    """Hiperparametros y rutas de una corrida de entrenamiento/evaluacion."""

    # --- rutas (relativas a la raiz del repo; en Colab se sobreescriben) ---
    data_root: str = "data"
    out_dir: str = "resultados"

    # --- datos ---
    num_classes: int = FLOWERS102_CLASSES
    img_size: int = IMAGE_SIZE
    # Flowers102 trae splits oficiales (train/val/test), no hace falta repartir.

    # --- modelo ---
    backbone: str = "efficientnet_b0"
    dropout: float = 0.3
    pretrained: bool = True

    # --- entrenamiento ---
    epochs: int = 25
    batch_size: int = 32
    lr: float = 3e-4  # learning rate de la cabeza
    backbone_lr: float = 3e-5  # learning rate del backbone al descongelar
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 2
    freeze_epochs: int = 3  # epocas iniciales con el backbone congelado
    early_stop_patience: int = 6
    amp: bool = True  # mixed precision (se ignora automaticamente en CPU)

    # --- inferencia / demo ---
    topk: int = 5
    min_conf: float = 0.25  # por debajo de esto -> "no reconocida"

    # --- sistema ---
    seed: int = 42
    num_workers: int = 2
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # --- modo debug (smoke test / verificar Colab en ~1 min) ---
    debug: bool = False
    debug_per_class: int = 4  # imagenes por clase en modo debug

    def __post_init__(self) -> None:
        if self.debug:
            self.epochs = min(self.epochs, 2)
            self.freeze_epochs = 0
            self.warmup_epochs = 0
            self.amp = False
            self.num_workers = 0
        self.data_root = str(self.data_root)
        self.out_dir = str(self.out_dir)

    # ------------------------------------------------------------- derivados
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.out_dir) / "best.pt"

    @property
    def history_path(self) -> Path:
        return Path(self.out_dir) / "history.json"

    @property
    def metrics_path(self) -> Path:
        return Path(self.out_dir) / "metrics.json"

    @property
    def config_path(self) -> Path:
        return Path(self.out_dir) / "config_usada.json"

    def figures_dir(self) -> Path:
        return Path(self.out_dir) / "figuras"

    def predictions_dir(self) -> Path:
        return Path(self.out_dir) / "predicciones"

    def ensure_dirs(self) -> None:
        """Crea los directorios de salida. Idempotente."""
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.figures_dir().mkdir(parents=True, exist_ok=True)
        self.predictions_dir().mkdir(parents=True, exist_ok=True)
        Path(self.data_root).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)
