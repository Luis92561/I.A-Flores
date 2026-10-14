"""Pruebas de `src/train.py` (SA-05).

Criterio de aceptacion del contrato (§4):
    `train(Config(debug=True, epochs=2, device="cpu"))` termina y escribe
    `best.pt`, `history.json` y `curvas.png`.

Estrategia:
  * Pruebas rapidas: se monkeypatchea `src.train.build_dataloaders` para devolver
    3 `DataLoader` con tensores aleatorios. No hay red, ni dataset, ni descarga de
    pesos (`pretrained=False`).
  * Prueba lenta (dataset real, 330 MB): `test_train_dataset_real_end_to_end` se
    salta salvo que se exporte `IA_FLORES_SLOW=1`. No usa ningun marcador de pytest
    a proposito (pytest.ini es del orquestador y no registra marcadores):
        $env:IA_FLORES_SLOW = "1"
        .\\.venv\\Scripts\\python.exe -m pytest tests\\test_train.py -q

Ejecutar:  .venv\\Scripts\\python.exe -m pytest tests\\test_train.py -q
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pytest

# Permite `import src.train` aun si se ejecuta pytest desde otro directorio.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)  # CPU: no acaparar nucleos mientras otros subagentes trabajan

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import src.train as train_module  # noqa: E402
from src.config import Config  # noqa: E402
from src.model import CHECKPOINT_KEYS, load_checkpoint  # noqa: E402
from src.train import (  # noqa: E402
    HISTORY_KEYS,
    HISTORY_LIST_KEYS,
    HISTORY_SCALAR_KEYS,
    build_argparser,
    config_from_args,
    main,
    train,
)
from src.utils import load_json  # noqa: E402

#: Claves que la celda de curvas del notebook (SA-03) indexa DIRECTO.
CLAVES_DEL_NOTEBOOK = ("epochs", "train_loss", "val_loss", "train_acc", "val_acc")

IMG_SIZE_SINTETICO = 32  # imagenes diminutas: los tests deben durar segundos


# --------------------------------------------------------------------------- #
# Infraestructura sintetica (sin red, sin dataset)
# --------------------------------------------------------------------------- #
def _split(n: int, num_classes: int, seed: int) -> TensorDataset:
    """Subconjunto sintetico reproducible: tensores aleatorios + etiquetas."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE_SINTETICO, IMG_SIZE_SINTETICO, generator=generator)
    y = torch.randint(0, int(num_classes), (n,), generator=generator)
    return TensorDataset(x, y)


def build_dataloaders_falso(n_train: int = 16, n_val: int = 8, n_test: int = 8):
    """Sustituto de `src.train.build_dataloaders` con 3 DataLoaders sinteticos.

    Respeta `cfg.batch_size` y `cfg.num_classes` para que el resto de `train()`
    (criterio, checkpoint, historial) se comporte como con datos reales.
    """

    def _fake(cfg: Config):
        batch_size = int(cfg.batch_size)
        class_names = [f"especie_{i + 1:03d}" for i in range(int(cfg.num_classes))]
        loaders = (
            DataLoader(_split(n_train, cfg.num_classes, 1), batch_size=batch_size, shuffle=True),
            DataLoader(_split(n_val, cfg.num_classes, 2), batch_size=batch_size, shuffle=False),
            DataLoader(_split(n_test, cfg.num_classes, 3), batch_size=batch_size, shuffle=False),
        )
        return loaders[0], loaders[1], loaders[2], class_names

    return _fake


def config_sintetica(tmp_path: Path, **overrides) -> Config:
    """`Config` minimo y rapido para los tests (CPU, 4 clases, pretrained=False)."""
    base = dict(
        num_classes=4,
        debug=True,
        epochs=2,
        batch_size=8,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    base.update(overrides)
    return Config(**base)


class LoaderQueFalla:
    """DataLoader falso que revienta como los workers del sandbox de Windows.

    Simula `PermissionError [WinError 5]` al iterar con `num_workers > 0` (los
    procesos hijo no pueden abrir named pipes). Expone `.dataset` y `__len__`
    porque `train()` los usa para el log.
    """

    def __init__(self, dataset, error: BaseException | None = None) -> None:
        self.dataset = dataset
        self.error = error or PermissionError(
            "[WinError 5] Acceso denegado: no se puede abrir el named pipe del worker"
        )

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        raise self.error


def build_dataloaders_con_fallo_de_workers(n_train: int = 8, n_val: int = 8, n_test: int = 8):
    """Sustituto que falla con `num_workers > 0` y funciona con `num_workers=0`.

    Devuelve `(fake, llamadas)`; `llamadas` registra el `num_workers` de cada
    invocacion para comprobar que hubo exactamente un reintento.
    """
    llamadas: list[int] = []

    def _fake(cfg: Config):
        llamadas.append(int(cfg.num_workers))
        class_names = [f"especie_{i + 1:03d}" for i in range(int(cfg.num_classes))]
        if int(cfg.num_workers) > 0:
            roto = LoaderQueFalla(_split(2, int(cfg.num_classes), 1))
            return roto, roto, roto, class_names

        real = build_dataloaders_falso(n_train, n_val, n_test)(cfg)
        return real

    return _fake, llamadas


# --------------------------------------------------------------------------- #
# End-to-end sintetico
# --------------------------------------------------------------------------- #
def test_train_sintetico_termina_y_escribe_artefactos(tmp_path: Path, monkeypatch) -> None:
    """`train()` termina, devuelve el historial completo y escribe los 4 artefactos."""
    cfg = config_sintetica(tmp_path)
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    history = train(cfg)

    # --- contrato del historial -------------------------------------------
    assert set(history) == set(HISTORY_KEYS)
    assert list(history) == list(HISTORY_KEYS), "orden de claves del contrato §3"
    for clave in CLAVES_DEL_NOTEBOOK:
        assert clave in history, f"el notebook de SA-03 indexa {clave!r} directamente"
    for clave in HISTORY_LIST_KEYS:
        assert isinstance(history[clave], list)
        assert len(history[clave]) == 2, f"{clave} debe tener una entrada por epoca"
    for clave in HISTORY_SCALAR_KEYS:
        assert not isinstance(history[clave], list)

    assert history["epochs"] == [1, 2]
    assert history["epochs_run"] == 2
    assert history["device"] == "cpu"
    assert history["num_params"] > 0
    assert history["segundos"] > 0.0
    assert 0.0 <= history["best_val_acc"] <= 1.0
    assert history["best_epoch"] in (1, 2)
    assert history["best_val_acc"] == pytest.approx(max(history["val_acc"]))
    assert history["best_val_acc"] == pytest.approx(history["val_acc"][history["best_epoch"] - 1])
    assert all(0.0 <= v <= 1.0 for v in history["val_acc"])
    assert all(0.0 <= v <= 1.0 for v in history["val_top5"])
    assert all(v > 0 for v in history["lr"])
    assert history["lr"][0] == pytest.approx(cfg.lr)  # sin warmup arranca en cfg.lr

    # --- artefactos en disco ----------------------------------------------
    assert cfg.checkpoint_path.is_file(), "falta best.pt"
    assert cfg.history_path.is_file(), "falta history.json"
    assert cfg.config_path.is_file(), "falta config_usada.json"
    curvas = cfg.figures_dir() / "curvas.png"
    assert curvas.is_file(), "falta figuras/curvas.png"
    assert curvas.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "curvas.png no es un PNG valido"

    # history.json == historial devuelto (JSON serializable de verdad)
    guardado = load_json(cfg.history_path)
    assert set(guardado) == set(HISTORY_KEYS)
    assert guardado["epochs"] == [1, 2]
    assert guardado["best_epoch"] == history["best_epoch"]
    assert guardado["num_params"] == history["num_params"]

    # config_usada.json refleja la config efectiva de la corrida
    config_guardada = load_json(cfg.config_path)
    assert config_guardada["num_classes"] == 4
    assert config_guardada["debug"] is True
    assert config_guardada["out_dir"] == str(cfg.out_dir)


def test_checkpoint_cumple_contrato_y_guarda_la_mejor_epoca(tmp_path: Path, monkeypatch) -> None:
    """`best.pt` solo se escribe al mejorar `val_acc` y trae las 8 claves del contrato."""
    cfg = config_sintetica(tmp_path)
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    history = train(cfg)

    crudo = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    for clave in CHECKPOINT_KEYS:
        assert clave in crudo, f"falta la clave {clave!r} en el checkpoint"
    assert crudo["epoch"] == history["best_epoch"]
    assert crudo["val_acc"] == pytest.approx(history["best_val_acc"])
    assert crudo["num_classes"] == 4
    assert crudo["backbone"] == cfg.backbone
    assert len(crudo["class_names"]) == 4
    assert crudo["config"]["num_classes"] == 4

    # El checkpoint es cargable por SA-02 / SA-06 y reproduce inferencia.
    modelo, class_names, meta = load_checkpoint(cfg.checkpoint_path, device="cpu")
    assert class_names == [f"especie_{i + 1:03d}" for i in range(4)]
    assert meta["val_acc"] == pytest.approx(history["best_val_acc"])
    assert modelo.training is False
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, IMG_SIZE_SINTETICO, IMG_SIZE_SINTETICO))
    assert salida.shape == (2, 4)


def test_train_registra_lr_warmup_coseno_y_descongela_backbone(tmp_path: Path, monkeypatch) -> None:
    """Warmup + coseno (SequentialLR) y descongelado del backbone en `freeze_epochs`."""
    cfg = Config(
        num_classes=4,
        debug=False,  # debug forzaria warmup=freeze=0: aqui queremos probar ambos
        epochs=4,
        batch_size=8,
        warmup_epochs=2,
        freeze_epochs=2,
        early_stop_patience=50,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    llamadas: list[bool] = []
    real = train_module.set_backbone_trainable

    def _spy(model, trainable):
        llamadas.append(bool(trainable))
        return real(model, trainable)

    monkeypatch.setattr(train_module, "set_backbone_trainable", _spy)

    history = train(cfg)

    # Congelado en las 2 primeras epocas y descongelado despues, con el MISMO
    # optimizador (no se reconstruye en ningun momento).
    assert False in llamadas, "el backbone nunca se congelo"
    assert True in llamadas, "el backbone nunca se descongelo"
    # [antes del bucle, epoca 1, epoca 2, epoca 3, epoca 4, al terminar]
    assert llamadas == [False, False, False, True, True, True]

    lrs = history["lr"]
    assert len(lrs) == 4
    assert len(set(lrs)) > 1, "el scheduler no cambio el LR"
    assert lrs[0] < lrs[1] < lrs[2], "el warmup debe subir el LR epoca a epoca"
    assert lrs[-1] < max(lrs), "el coseno debe bajar el LR"
    assert max(lrs) == pytest.approx(cfg.lr)  # pico del warmup = lr de la cabeza
    assert lrs[0] == pytest.approx(cfg.lr * 0.1)  # arranque del warmup
    assert lrs[-1] >= 0.0


def test_train_warmup_cero_usa_solo_coseno(tmp_path: Path, monkeypatch) -> None:
    """Con `warmup_epochs=0` no hay SequentialLR: el LR solo decrece (coseno)."""
    cfg = Config(
        num_classes=4,
        debug=False,
        epochs=3,
        batch_size=8,
        warmup_epochs=0,
        freeze_epochs=0,
        early_stop_patience=50,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    history = train(cfg)

    lrs = history["lr"]
    assert len(lrs) == 3
    assert lrs[0] == pytest.approx(cfg.lr)
    assert lrs[0] > lrs[1] > lrs[2]


def test_early_stopping_corta_el_entrenamiento(tmp_path: Path, monkeypatch) -> None:
    """Con `early_stop_patience=1` y val_acc que nunca mejora, para tras la 2a epoca."""
    cfg = Config(
        num_classes=4,
        debug=False,
        epochs=10,
        batch_size=8,
        warmup_epochs=0,
        freeze_epochs=0,
        early_stop_patience=1,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    # Serie determinista de val_acc decreciente: nunca hay mejora despues de la 1a.
    serie = [0.50, 0.40, 0.30, 0.20, 0.10]

    def _eval_falso(model, loader, criterion, device, amp_enabled, topk=5):
        val_acc = serie.pop(0)
        return 1.0 - val_acc, val_acc, min(1.0, val_acc + 0.1)

    monkeypatch.setattr(train_module, "_run_eval_epoch", _eval_falso)

    history = train(cfg)

    assert history["epochs_run"] == 2, "debio parar en la 2a epoca (paciencia 1)"
    assert history["epochs"] == [1, 2]
    assert history["best_epoch"] == 1
    assert history["best_val_acc"] == pytest.approx(0.50)
    assert history["val_acc"] == pytest.approx([0.50, 0.40])
    assert cfg.checkpoint_path.is_file()
    assert len(serie) == 3, "el bucle siguio consumiendo epocas tras el corte"


def test_amp_desactivado_en_cpu_sin_avisos(tmp_path: Path, monkeypatch) -> None:
    """En CPU, `amp=True` no construye GradScaler ni activa autocast, y no avisa."""
    cfg = Config(
        num_classes=4,
        debug=False,
        epochs=1,
        batch_size=8,
        warmup_epochs=0,
        freeze_epochs=0,
        amp=True,  # se debe ignorar en CPU
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    escalares: list[tuple] = []
    real_scaler = torch.amp.GradScaler

    def _spy_scaler(*args, **kwargs):
        escalares.append((args, kwargs))
        return real_scaler(*args, **kwargs)

    flags_autocast: list[bool] = []
    real_autocast = torch.amp.autocast

    def _spy_autocast(*args, **kwargs):
        flags_autocast.append(bool(kwargs.get("enabled", True)))
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", _spy_scaler)
    monkeypatch.setattr(torch.amp, "autocast", _spy_autocast)

    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        history = train(cfg)

    assert escalares == [], "no se debe crear GradScaler en CPU"
    assert flags_autocast, "autocast no se llamo"
    assert not any(flags_autocast), "autocast debe ir con enabled=False en CPU"
    assert history["epochs_run"] == 1

    mensajes = [str(w.message).lower() for w in capturados]
    assert not [m for m in mensajes if "gradscaler" in m or "cuda" in m], (
        f"AMP en CPU no debe avisar: {mensajes}"
    )


def test_train_es_reproducible_con_la_misma_semilla(tmp_path: Path, monkeypatch) -> None:
    """`set_seed(cfg.seed)` hace que dos corridas identicas den el mismo historial."""
    monkeypatch.setattr(train_module, "build_dataloaders", build_dataloaders_falso())

    primera = train(config_sintetica(tmp_path / "a", seed=123))
    segunda = train(config_sintetica(tmp_path / "b", seed=123))

    assert primera["train_loss"] == pytest.approx(segunda["train_loss"])
    assert primera["val_acc"] == pytest.approx(segunda["val_acc"])


def test_train_reintenta_con_num_workers_0_si_fallan_los_workers(tmp_path: Path, monkeypatch) -> None:
    """Fallo de multiprocessing/named pipes: aviso + UN reintento con num_workers=0."""
    fake, llamadas = build_dataloaders_con_fallo_de_workers()
    monkeypatch.setattr(train_module, "build_dataloaders", fake)

    cfg = Config(
        num_classes=4,
        debug=False,  # debug forzaria num_workers=0 y no probaria el fallback
        epochs=2,
        batch_size=8,
        num_workers=4,
        warmup_epochs=0,
        freeze_epochs=0,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )

    with pytest.warns(RuntimeWarning, match="num_workers=0"):
        history = train(cfg)

    assert llamadas == [4, 0], f"debe reconstruir los loaders una sola vez: {llamadas}"
    assert history["epochs_run"] == 2
    assert cfg.checkpoint_path.is_file()
    assert history["num_params"] > 0


def test_train_no_reintenta_errores_que_no_son_de_workers(tmp_path: Path, monkeypatch) -> None:
    """Un error real del dataset NO se enmascara con el reintento."""

    def _fake(cfg: Config):
        raise ValueError("el dataset tiene un formato inesperado")

    monkeypatch.setattr(train_module, "build_dataloaders", _fake)

    cfg = Config(
        num_classes=4,
        debug=False,
        epochs=1,
        num_workers=2,
        device="cpu",
        pretrained=False,
        out_dir=str(tmp_path / "salida"),
    )
    with pytest.raises(ValueError, match="formato inesperado"):
        train(cfg)


def test_es_error_de_workers_clasifica_bien() -> None:
    """El detector de fallos de workers no confunde errores reales."""
    from src.train import _es_error_de_workers

    assert _es_error_de_workers(PermissionError("[WinError 5] Acceso denegado"))
    assert _es_error_de_workers(RuntimeError("DataLoader worker (pid 123) is killed by signal"))
    assert _es_error_de_workers(EOFError())
    assert not _es_error_de_workers(ValueError("backbone no soportado"))
    assert not _es_error_de_workers(FileNotFoundError("imagen.jpg no existe"))


# --------------------------------------------------------------------------- #
# CLI (§3.b)
# --------------------------------------------------------------------------- #
def test_cli_help_no_crashea() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_sin_argumentos_usa_los_defaults_de_config() -> None:
    cfg = config_from_args(build_argparser().parse_args([]))
    assert cfg.to_dict() == Config().to_dict()


def test_cli_aplica_los_overrides_informados() -> None:
    args = build_argparser().parse_args(
        [
            "--epochs", "7",
            "--batch-size", "16",
            "--backbone", "resnet50",
            "--data-root", "datos-x",
            "--out-dir", "salida-x",
            "--device", "cpu",
            "--seed", "7",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.epochs == 7
    assert cfg.batch_size == 16
    assert cfg.backbone == "resnet50"
    assert cfg.data_root == "datos-x"
    assert cfg.out_dir == "salida-x"
    assert cfg.device == "cpu"
    assert cfg.seed == 7
    # Lo no informado conserva los defaults (dataclasses.replace sobre Config()).
    base = Config()
    assert cfg.lr == base.lr
    assert cfg.dropout == base.dropout
    assert cfg.early_stop_patience == base.early_stop_patience
    assert cfg.debug is False


def test_cli_num_workers_default_de_config_y_override() -> None:
    """`--num-workers` es una extension acordada; el default de Config sigue siendo 2."""
    base = Config()
    assert base.num_workers == 2, "el default de Config no debe cambiar (Colab)"

    cfg = config_from_args(build_argparser().parse_args([]))
    assert cfg.num_workers == 2

    cfg = config_from_args(build_argparser().parse_args(["--num-workers", "0"]))
    assert cfg.num_workers == 0

    cfg = config_from_args(build_argparser().parse_args(["--num-workers", "6"]))
    assert cfg.num_workers == 6

    # Con --debug, Config fuerza num_workers=0 (subset en memoria, sin workers).
    cfg = config_from_args(build_argparser().parse_args(["--debug", "--num-workers", "6"]))
    assert cfg.num_workers == 0


def test_cli_debug_usa_resultados_debug_y_no_pisa_la_corrida_completa() -> None:
    cfg = config_from_args(build_argparser().parse_args(["--debug"]))
    assert cfg.debug is True
    assert cfg.out_dir == "resultados/debug"
    assert cfg.checkpoint_path == Path("resultados/debug") / "best.pt"

    # `--out-dir` explicito siempre gana.
    cfg = config_from_args(build_argparser().parse_args(["--debug", "--out-dir", "otra/salida"]))
    assert cfg.out_dir == "otra/salida"

    # Modo debug: Config.__post_init__ recorta epocas y desactiva freeze/warmup/amp.
    cfg = config_from_args(build_argparser().parse_args(["--debug", "--epochs", "7"]))
    assert cfg.epochs == 2
    assert cfg.freeze_epochs == 0
    assert cfg.warmup_epochs == 0
    assert cfg.amp is False
    assert cfg.num_workers == 0


# --------------------------------------------------------------------------- #
# End-to-end con el dataset REAL (lento, opt-in)
# --------------------------------------------------------------------------- #
def _parece_error_de_red(exc: BaseException) -> bool:
    """True si la excepcion huele a falta de internet (para saltar, no fallar)."""
    texto = f"{type(exc).__name__}: {exc}".lower()
    pistas = (
        "download",
        "descarga",
        "url",
        "http",
        "https",
        "connection",
        "conexion",
        "timed out",
        "timeout",
        "socket",
        "network",
        "unreachable",
        "ssl",
        "name or service",
        "getaddrinfo",
    )
    return any(pista in texto for pista in pistas)


def test_train_dataset_real_end_to_end(tmp_path: Path) -> None:
    """2 epocas REALES con Flowers-102 en CPU (lento: ~minutos).

    Se salta salvo que se exporte `IA_FLORES_SLOW=1`:

        $env:IA_FLORES_SLOW = "1"
        .\\.venv\\Scripts\\python.exe -m pytest tests\\test_train.py -q -s

    Si la descarga del dataset (o de los pesos) falla por red, se hace `skip`.

    NOTA (peculiaridad de esta maquina, no del proyecto): el sandbox bloquea
    `~/.cache/torch`, asi que para que esta prueba use pesos preentrenados DE
    VERDAD hay que apuntar TORCH_HOME a una carpeta escribible del workspace:

        $env:TORCH_HOME = (Resolve-Path .piptmp\\torchhome).Path

    Sin eso, `src/model.py` degrada a `pretrained=False` con un `RuntimeWarning`
    (comportamiento correcto y ya probado por SA-02) y el test sigue pasando.
    """
    if os.environ.get("IA_FLORES_SLOW") != "1":
        pytest.skip("lento (dataset real): exporta IA_FLORES_SLOW=1 para ejecutarlo")

    cfg = Config(
        debug=True,
        epochs=2,
        device="cpu",
        backbone="mobilenet_v3_large",
        out_dir=str(tmp_path / "real"),
    )

    inicio = time.perf_counter()
    try:
        history = train(cfg)
    except Exception as exc:  # noqa: BLE001 - red caida -> skip, bug real -> raise
        if _parece_error_de_red(exc):
            pytest.skip(f"sin red para el dataset/pesos reales: {exc}")
        raise
    segundos = time.perf_counter() - inicio

    print(
        f"\n[SLOW] Flowers-102 real, mobilenet_v3_large, CPU: 2 epocas en "
        f"{segundos:.1f} s ({segundos / 60.0:.2f} min)"
    )

    assert history["epochs_run"] == 2
    assert history["epochs"] == [1, 2]
    assert history["num_params"] > 300_000
    assert all(0.0 <= v <= 1.0 for v in history["val_acc"])
    assert cfg.checkpoint_path.is_file()
    assert cfg.history_path.is_file()
    assert cfg.config_path.is_file()
    assert (cfg.figures_dir() / "curvas.png").is_file()

    # El dataset real trae las 102 especies y el checkpoint lo refleja.
    crudo = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    assert crudo["num_classes"] == 102
    assert len(crudo["class_names"]) == 102
    assert crudo["backbone"] == "mobilenet_v3_large"
    for clave in CHECKPOINT_KEYS:
        assert clave in crudo
