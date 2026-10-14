"""Pruebas de `src/model.py` (SA-02).

Criterio de aceptacion del contrato:
  * `build_model(102)` con `pretrained=False` -> forward (1,3,224,224) -> (1,102).
  * `save_checkpoint` + `load_checkpoint` reproducen EXACTAMENTE las mismas salidas.

Todas las pruebas usan `pretrained=False` para no depender de la red.
Se ejecutan con:  .venv\\Scripts\\python.exe -m pytest tests\\test_model.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Permite `import src.model` aun si se ejecuta pytest desde otro directorio
# (el comando del contrato ya funciona porque `python -m pytest` agrega el cwd).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)  # CPU: evita que resnet50 acapare todos los nucleos

from src.config import FLOWERS102_CLASSES, Config  # noqa: E402
from src.model import (  # noqa: E402
    CHECKPOINT_KEYS,
    SUPPORTED_BACKBONES,
    FlowerClassifier,
    build_model,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)

BACKBONES = ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]
CLASS_NAMES = [f"especie_{i + 1:03d}" for i in range(FLOWERS102_CLASSES)]


def _input(batch: int = 1, seed: int = 0) -> torch.Tensor:
    """Tensor de entrada deterministico (misma entrada al comparar salidas)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, 224, 224, generator=generator)


def _trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Estructura y forward
# --------------------------------------------------------------------------- #
def test_supported_backbones_declarados() -> None:
    assert set(SUPPORTED_BACKBONES) == set(BACKBONES)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_forward_shape_y_atributos(backbone: str) -> None:
    model = build_model(102, backbone=backbone, pretrained=False)
    assert isinstance(model, FlowerClassifier)
    # Atributos obligatorios del contrato.
    assert isinstance(model.backbone, torch.nn.Module)
    assert isinstance(model.head, torch.nn.Module)
    assert model.backbone is not model.head

    model.eval()
    with torch.no_grad():
        out = model(_input())
    assert out.shape == (1, 102)
    assert torch.isfinite(out).all()


def test_pretrained_degrada_a_no_pretrained_con_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la descarga de pesos falla (Colab sin internet) avisa y sigue sin fallar."""
    import src.model as model_module

    def _builder_sin_red(pretrained: bool):
        if pretrained:
            raise RuntimeError("descarga de pesos bloqueada (simulada)")
        return model_module._build_efficientnet_b0(False)

    monkeypatch.setitem(model_module._BACKBONE_BUILDERS, "efficientnet_b0", _builder_sin_red)

    with pytest.warns(RuntimeWarning, match="pretrained"):
        model = build_model(102, backbone="efficientnet_b0", pretrained=True)

    assert model.pretrained is False
    model.eval()
    with torch.no_grad():
        assert model(_input()).shape == (1, 102)


def test_error_real_sin_pretrained_se_propaga(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con pretrained=False no se enmascara un fallo de construccion del backbone."""
    import src.model as model_module

    def _builder_roto(pretrained: bool):
        raise RuntimeError("fallo real del backbone")

    monkeypatch.setitem(model_module._BACKBONE_BUILDERS, "efficientnet_b0", _builder_roto)
    with pytest.raises(RuntimeError, match="fallo real"):
        build_model(102, backbone="efficientnet_b0", pretrained=False)


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50"])
def test_forward_batch_y_num_classes_configurable(backbone: str) -> None:
    model = build_model(10, backbone=backbone, pretrained=False, dropout=0.0).eval()
    with torch.no_grad():
        out = model(_input(batch=2, seed=1))
    assert out.shape == (2, 10)


def test_dropout_se_aplica_en_la_cabeza() -> None:
    dropout = build_model(102, backbone="efficientnet_b0", pretrained=False, dropout=0.5)
    assert isinstance(dropout.head[0], torch.nn.Dropout)
    assert dropout.head[0].p == pytest.approx(0.5)
    assert dropout.head[-1].out_features == 102


# --------------------------------------------------------------------------- #
# Congelar / descongelar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backbone", BACKBONES)
def test_set_backbone_trainable(backbone: str) -> None:
    model = build_model(102, backbone=backbone, pretrained=False)
    total_backbone = sum(p.numel() for p in model.backbone.parameters())
    assert total_backbone > 0

    head_antes = _trainable_params(model.head)
    set_backbone_trainable(model, False)
    assert _trainable_params(model.backbone) == 0
    # Solo se toca el backbone: la cabeza sigue entrenable.
    assert _trainable_params(model.head) == head_antes > 0

    set_backbone_trainable(model, True)
    assert _trainable_params(model.backbone) == total_backbone


def test_set_backbone_trainable_sin_backbone() -> None:
    with pytest.raises(AttributeError):
        set_backbone_trainable(torch.nn.Linear(3, 3), False)


# --------------------------------------------------------------------------- #
# Optimizador con LR diferencial
# --------------------------------------------------------------------------- #
def test_build_optimizer_dos_grupos() -> None:
    cfg = Config()
    model = build_model(cfg.num_classes, backbone=cfg.backbone, pretrained=False)
    optimizer = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2

    grupo_cabeza, grupo_backbone = optimizer.param_groups
    assert grupo_cabeza["lr"] == pytest.approx(cfg.lr)
    assert grupo_backbone["lr"] == pytest.approx(cfg.backbone_lr)
    assert grupo_cabeza["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert grupo_backbone["weight_decay"] == pytest.approx(cfg.weight_decay)

    ids_cabeza = {id(p) for p in grupo_cabeza["params"]}
    ids_backbone = {id(p) for p in grupo_backbone["params"]}
    assert ids_cabeza == {id(p) for p in model.head.parameters()}
    assert ids_backbone == {id(p) for p in model.backbone.parameters()}
    assert not (ids_cabeza & ids_backbone)

    # El optimizador sigue siendo valido tras descongelar el backbone.
    set_backbone_trainable(model, False)
    set_backbone_trainable(model, True)
    x = _input()
    torch.nn.functional.cross_entropy(model(x), torch.zeros(1, dtype=torch.long)).backward()
    optimizer.step()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50", "mobilenet_v3_large"])
def test_save_load_checkpoint_reproduce_salidas(tmp_path: Path, backbone: str) -> None:
    cfg = Config(img_size=224, backbone=backbone, dropout=0.2, pretrained=False)
    model = build_model(cfg.num_classes, backbone=backbone, pretrained=False, dropout=cfg.dropout)
    model.eval()

    x = _input()
    with torch.no_grad():
        salida_original = model(x).clone()

    ckpt_path = tmp_path / "best.pt"
    save_checkpoint(ckpt_path, model, CLASS_NAMES, cfg, epoch=7, val_acc=0.8123)
    assert ckpt_path.is_file()

    cargado, class_names, meta = load_checkpoint(ckpt_path, device="cpu")

    # Contrato del checkpoint.
    assert isinstance(cargado, FlowerClassifier)
    assert class_names == CLASS_NAMES
    assert len(class_names) == 102
    # Contrato del checkpoint: las 8 claves deben estar en el ARCHIVO guardado.
    crudo = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for clave in CHECKPOINT_KEYS:
        assert clave in crudo, f"falta la clave {clave!r} en el checkpoint"
    # `meta` es el checkpoint SIN los pesos (decision documentada de SA-02).
    for clave in set(CHECKPOINT_KEYS) - {"model_state"}:
        assert clave in meta, f"falta la clave {clave!r} en meta"
    assert meta["backbone"] == backbone
    assert meta["num_classes"] == 102
    assert meta["img_size"] == 224
    assert meta["epoch"] == 7
    assert meta["val_acc"] == pytest.approx(0.8123)
    assert isinstance(meta["config"], dict) and meta["config"]["backbone"] == backbone

    # El modelo cargado queda en eval() y en el device pedido.
    assert cargado.training is False
    assert next(cargado.parameters()).device.type == "cpu"

    cargado.eval()
    with torch.no_grad():
        salida_cargada = cargado(x)
    assert salida_cargada.shape == salida_original.shape
    assert torch.allclose(salida_original, salida_cargada, atol=1e-6, rtol=1e-5)
    assert (salida_original - salida_cargada).abs().max().item() == 0.0


def test_checkpoint_crea_directorios_y_defaults(tmp_path: Path) -> None:
    cfg = Config(pretrained=False)
    model = build_model(cfg.num_classes, backbone=cfg.backbone, pretrained=False)
    ckpt_path = tmp_path / "sub" / "carpeta" / "sin_extra.pt"  # no existe todavia
    save_checkpoint(ckpt_path, model, CLASS_NAMES, cfg)
    assert ckpt_path.is_file()

    _, _, meta = load_checkpoint(ckpt_path)
    assert meta["epoch"] == -1
    assert meta["val_acc"] == pytest.approx(0.0)


def test_load_checkpoint_device_auto(tmp_path: Path) -> None:
    from src.utils import resolve_device

    cfg = Config(pretrained=False)
    model = build_model(cfg.num_classes, backbone=cfg.backbone, pretrained=False)
    ckpt_path = tmp_path / "auto.pt"
    save_checkpoint(ckpt_path, model, CLASS_NAMES, cfg, epoch=0, val_acc=0.1)

    cargado, _, _ = load_checkpoint(ckpt_path, device="auto")
    esperado = resolve_device("auto")
    assert next(cargado.parameters()).device == esperado


def test_load_checkpoint_archivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "no_existe.pt")


# --------------------------------------------------------------------------- #
# Entradas invalidas
# --------------------------------------------------------------------------- #
def test_backbone_invalido_lanza_valueerror() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_model(102, backbone="vgg16", pretrained=False)

    mensaje = str(excinfo.value)
    for soportado in SUPPORTED_BACKBONES:
        assert soportado in mensaje
    assert "vgg16" in mensaje


def test_backbone_invalido_tambien_con_pretrained_true() -> None:
    with pytest.raises(ValueError):
        build_model(102, backbone="alexnet", pretrained=True)
