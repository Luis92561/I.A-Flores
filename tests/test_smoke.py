"""Smoke test end-to-end del pipeline completo (dataset sintetico, sin red).

Ejercita la cadena REAL de modulos -- `data` (transforms) -> `model` -> `train` ->
`evaluate` -> `predict` -- sobre un dataset de imagenes sinteticas en disco, con un
modelo diminuto de 4 clases. No descarga nada y tarda unos segundos en CPU.

Es el test que anuncian `README.md` (seccion 4) y `CONTRATOS.md` (seccion 2) como
"pipeline end-to-end con dataset sintetico".

Ejecutar:
    .venv\\Scripts\\python.exe -m pytest tests\\test_smoke.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

torch.set_num_threads(2)

from src import evaluate as evaluate_mod  # noqa: E402
from src import train as train_mod  # noqa: E402
from src.config import Config  # noqa: E402
from src.data import build_transforms  # noqa: E402
from src.predict import predict_image  # noqa: E402
from src.utils import load_json  # noqa: E402

CLASES = 4
POR_CLASE = 6


def _crear_dataset_sintetico(base: Path) -> Path:
    """Crea `base/clase_i/i_j.png` con colores distintos por clase."""
    for i in range(CLASES):
        carpeta = base / f"clase_{i}"
        carpeta.mkdir(parents=True, exist_ok=True)
        for j in range(POR_CLASE):
            color = ((i * 61) % 256, (j * 37) % 256, (i * 17 + j * 11) % 256)
            Image.new("RGB", (96, 96), color).save(carpeta / f"{i}_{j}.png")
    return base


def _fabrica_loaders(base: Path, cfg: Config):
    """Devuelve una funcion compatible con `build_dataloaders` sobre el dataset falso."""

    def _build(_cfg=None):
        dataset = ImageFolder(str(base), transform=build_transforms(cfg, train=False))
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        return loader, loader, loader, list(dataset.classes)

    return _build


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    config = Config(
        num_classes=CLASES,
        debug=True,  # fuerza num_workers=0, amp=False, freeze_epochs=0, epochs<=2
        epochs=1,
        backbone="mobilenet_v3_large",
        pretrained=False,  # sin red
        out_dir=str(tmp_path / "salida"),
        data_root=str(tmp_path / "datos"),
    )
    config.ensure_dirs()
    return config


def test_pipeline_completo_end_to_end(tmp_path: Path, cfg: Config, monkeypatch) -> None:
    """data -> model -> train -> evaluate -> predict, con artefactos reales en disco."""
    base = _crear_dataset_sintetico(tmp_path / "imagenes")
    fabrica = _fabrica_loaders(base, cfg)
    monkeypatch.setattr(train_mod, "build_dataloaders", fabrica)
    monkeypatch.setattr(evaluate_mod, "build_dataloaders", fabrica)

    # --- 1) ENTRENAMIENTO ---------------------------------------------------
    historial = train_mod.train(cfg)

    assert cfg.checkpoint_path.is_file(), "el entrenamiento no guardo el checkpoint"
    assert cfg.history_path.is_file()
    for clave in ("epochs", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5"):
        assert clave in historial and len(historial[clave]) == 1, f"historial incompleto: {clave}"
    assert (cfg.figures_dir() / "curvas.png").is_file()

    # --- 2) EVALUACION ------------------------------------------------------
    metricas = evaluate_mod.evaluate(cfg.checkpoint_path, cfg, split="test")

    assert metricas["num_images"] == CLASES * POR_CLASE
    assert 0.0 <= metricas["top1"] <= 1.0
    assert 0.0 <= metricas["top5"] <= 1.0
    assert metricas["top5"] >= metricas["top1"], "top-5 nunca puede ser menor que top-1"
    assert cfg.metrics_path.is_file()
    assert (cfg.figures_dir() / "confusion_matrix.png").is_file()
    csv = cfg.predictions_dir() / "test_predicciones.csv"
    assert csv.is_file()
    assert len(csv.read_text(encoding="utf-8").strip().splitlines()) == CLASES * POR_CLASE + 1

    # --- 3) INFERENCIA ------------------------------------------------------
    imagen = Image.new("RGB", (200, 150), (30, 140, 90))
    top = predict_image(imagen, cfg.checkpoint_path, topk=3, min_conf=0.0, device="cpu")

    assert len(top) == 3
    assert all(isinstance(n, str) and isinstance(p, float) for n, p in top)
    assert all(0.0 <= p <= 1.0 for _, p in top)
    assert top == sorted(top, key=lambda par: par[1], reverse=True), "no estan ordenadas"
    assert {n for n, _ in top} <= set(load_json(cfg.metrics_path)["per_class_f1"].keys())

    # --- 4) EL UMBRAL DE CONFIANZA RESPONDE "NO RECONOCIDA" -----------------
    estricto = predict_image(imagen, cfg.checkpoint_path, topk=3, min_conf=1.1, device="cpu")
    assert estricto[0][0].lower().startswith("no reconocida")
