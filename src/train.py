"""Entrenamiento de Flowers-102 (propietario: SA-05).

CONTRATO (CONTRATOS.md seccion 3, `src/train.py`):

    train(cfg: Config) -> dict          # entrena y devuelve el historial
    # CLI: python -m src.train --epochs 5 --debug --device cpu

Detalles congelados que respeta este modulo:
  * `freeze_epochs`: las primeras epocas entrenan SOLO la cabeza
    (`set_backbone_trainable(model, False)`); despues se descongela el backbone con
    el MISMO optimizador (el grupo "backbone" ya estaba en el AdamW de
    `build_optimizer`, que es [0] = cabeza con `cfg.lr` y [1] = backbone con
    `cfg.backbone_lr`).
  * Scheduler: warmup de `cfg.warmup_epochs` + coseno, aplicado a los dos grupos.
  * AMP (`torch.amp.autocast` + `torch.amp.GradScaler`) SOLO si el device es CUDA
    y `cfg.amp`; en CPU queda desactivado y sin avisos.
  * `cfg.checkpoint_path` se guarda SOLO cuando mejora `val_acc`, con `epoch=` y
    `val_acc=` como `**extra` (asi el checkpoint cumple las 8 claves del contrato).
  * Artefactos: `best.pt`, `history.json`, `config_usada.json` y `figuras/curvas.png`
    (matplotlib en backend "Agg": nunca hay `.cuda()` duro ni ventanas).

Claves del historial (listas por epoca salvo las seis ultimas):
    epochs, train_loss, train_acc, val_loss, val_acc, val_top5, lr,
    best_val_acc, best_epoch, epochs_run, device, num_params, segundos

Convenio de numeracion: `history["epochs"]` y `best_epoch` son **1-based** (la
primera epoca es 1) para que los informes y la celda de curvas del notebook de
SA-03 lean "epoca 3" de forma natural. El `epoch` del checkpoint usa el mismo
convenio. `history["lr"]` guarda el LR del grupo de la cabeza que se aplico
durante la epoca (se lee justo antes del `scheduler.step()`).

CLI congelada (seccion 3.b) mas `--num-workers N` (extension acordada con el
orquestador; el default de `Config` sigue siendo 2):
    python -m src.train [--epochs N] [--batch-size N] [--backbone B] [--data-root D]
                        [--out-dir O] [--device auto|cuda|cpu] [--debug] [--seed N]
                        [--num-workers N]

Robustez de datos: si la iteracion del DataLoader con `num_workers > 0` falla por
multiprocessing / permisos (sandbox de Windows: los workers no pueden abrir named
pipes), `train` avisa por `warnings.warn` y REINTENTA UNA VEZ esa etapa con
`num_workers=0`, en vez de abortar el entrenamiento a mitad. En Colab/Linux los
workers funcionan y esta red de seguridad no se activa.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

try:  # import relativo (paquete `src`) con respaldo absoluto (cwd en sys.path)
    from .config import Config
    from .data import build_dataloaders
    from .model import (
        build_model,
        build_optimizer,
        save_checkpoint,
        set_backbone_trainable,
    )
    from .utils import count_params, describe_device, resolve_device, save_json, set_seed
except ImportError:  # pragma: no cover - depende de como se importe el modulo
    from src.config import Config
    from src.data import build_dataloaders
    from src.model import (
        build_model,
        build_optimizer,
        save_checkpoint,
        set_backbone_trainable,
    )
    from src.utils import count_params, describe_device, resolve_device, save_json, set_seed

__all__ = [
    "HISTORY_LIST_KEYS",
    "HISTORY_SCALAR_KEYS",
    "HISTORY_KEYS",
    "train",
    "build_scheduler",
    "config_from_args",
    "build_argparser",
    "main",
]

#: Claves del historial que son listas con una entrada por epoca.
HISTORY_LIST_KEYS: tuple[str, ...] = (
    "epochs",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_top5",
    "lr",
)
#: Claves del historial que son escalares (resumen de la corrida).
HISTORY_SCALAR_KEYS: tuple[str, ...] = (
    "best_val_acc",
    "best_epoch",
    "epochs_run",
    "device",
    "num_params",
    "segundos",
)
#: Todas las claves del historial, en el orden del contrato.
HISTORY_KEYS: tuple[str, ...] = HISTORY_LIST_KEYS + HISTORY_SCALAR_KEYS

#: Directorio de salida por defecto del CLI cuando se usa `--debug` (decision §5
#: del contrato: el modo debug NO debe pisar los resultados definitivos).
DEBUG_OUT_DIR = "resultados/debug"

#: Pistas de texto que identifican un fallo de los workers del DataLoader
#: (multiprocessing / named pipes / memoria compartida).
_PISTAS_WORKERS: tuple[str, ...] = (
    "winerror 5",
    "acceso denegado",
    "access is denied",
    "permission",
    "named pipe",
    "pipe",
    "dataloader",
    "data loader",
    "worker",
    "multiprocessing",
    "shared memory",
    "shared_memory",
    "semaphore",
    "queue",
    "eof",
)


# --------------------------------------------------------------------------- #
# Utilidades internas
# --------------------------------------------------------------------------- #
def _progress(iterable: Iterable, **kwargs: Any):
    """`tqdm` si esta disponible; si no, el iterable tal cual (sin romper nada)."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm esta en requirements.txt
        return iterable
    return tqdm(iterable, **kwargs)


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Numero de aciertos top-``k`` (``k`` se recorta a `num_classes`)."""
    num_classes = int(logits.shape[1])
    k = max(1, min(int(k), num_classes))
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return int(correct[:k].reshape(-1).sum().item())


def _move(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Lleva un lote (imagenes, etiquetas) al device resolviendo listas/tuplas."""
    images, targets = batch[0], batch[1]
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True).long()
    return images, targets


def _run_train_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    """Una epoca de entrenamiento. Devuelve ``(loss_media, acc_top1)``."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    seen = 0

    bar = _progress(loader, desc=f"train {epoch}/{epochs}", leave=False)
    for batch in bar:
        images, targets = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(targets.size(0))
        seen += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_correct += _topk_correct(logits.detach(), targets, 1)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(
                loss=f"{total_loss / max(seen, 1):.4f}",
                acc=f"{total_correct / max(seen, 1):.3f}",
            )
    if hasattr(bar, "close"):
        bar.close()

    return total_loss / max(seen, 1), total_correct / max(seen, 1)


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    topk: int = 5,
) -> tuple[float, float, float]:
    """Evalua un split. Devuelve ``(loss_media, acc_top1, acc_top5)``."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    seen = 0

    for batch in loader:
        images, targets = _move(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        seen += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_top1 += _topk_correct(logits, targets, 1)
        total_top5 += _topk_correct(logits, targets, topk)

    seen = max(seen, 1)
    return total_loss / seen, total_top1 / seen, total_top5 / seen


def _es_error_de_workers(exc: BaseException) -> bool:
    """True si la excepcion huele a fallo de los procesos hijo del DataLoader.

    En el sandbox de Windows de este entorno los workers no pueden abrir *named
    pipes* y revientan con ``PermissionError [WinError 5]``; en Linux/Colab
    ``num_workers > 0`` funciona bien. En vez de abortar un entrenamiento ya
    empezado, `train` reintenta UNA vez la etapa con ``num_workers=0``.
    """
    if not isinstance(exc, (OSError, RuntimeError, EOFError, MemoryError)):
        return False
    if isinstance(exc, PermissionError):  # el caso observado: WinError 5
        return True
    texto = f"{type(exc).__name__}: {exc}".lower()
    return any(pista in texto for pista in _PISTAS_WORKERS)


def _make_grad_scaler() -> Any:
    """`torch.amp.GradScaler("cuda")` (API del contrato) con respaldo para torch < 2.3.

    Solo se llama cuando AMP esta realmente activo (device CUDA + `cfg.amp`), asi
    que en CPU no se construye nunca y no hay avisos.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - torch antiguo en Colab
        return torch.cuda.amp.GradScaler(enabled=True)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> Any:
    """Scheduler warmup + coseno sobre TODOS los grupos de parametros.

    Con ``cfg.warmup_epochs > 0`` se usa
    ``SequentialLR([LinearLR, CosineAnnealingLR], milestones=[warmup_epochs])``;
    con ``warmup_epochs == 0`` se usa solo ``CosineAnnealingLR``. En ambos casos el
    factor de escala se aplica por grupo, asi que se respeta el LR diferencial
    cabeza/backbone que armo `build_optimizer`.
    """
    warmup = max(0, int(cfg.warmup_epochs))
    epochs = max(1, int(cfg.epochs))
    cosine_epochs = max(1, epochs - warmup)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=0.0
    )
    if warmup <= 0:
        return cosine

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
    )


def plot_history(history: dict, path: str | Path) -> Path:
    """Guarda `curvas.png` (loss, accuracy y learning rate por epoca).

    Usa el backend "Agg" ANTES de importar `pyplot`, salvo que `pyplot` ya este
    importado (p. ej. dentro de un notebook, donde cambiar el backend a "Agg"
    romperia el `plt.show()` de las celdas siguientes).
    """
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:  # no pisar el backend de Jupyter
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(history.get("epochs") or [])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    if epochs:
        axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
        axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
        axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
        axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
        axes[1].plot(epochs, history["val_top5"], marker="s", alpha=0.6, label="val top-5")
        axes[2].plot(epochs, history["lr"], marker="o", color="tab:green")
        best_epoch = history.get("best_epoch")
        if best_epoch:
            axes[1].axvline(best_epoch, ls="--", lw=1, color="tab:red", alpha=0.7)
            axes[1].annotate(
                f"mejor val_acc={history.get('best_val_acc', 0):.3f}",
                xy=(best_epoch, history.get("best_val_acc", 0)),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
    else:  # corrida de 0 epocas: la figura existe igualmente, pero vacia
        for axis in axes:
            axis.text(0.5, 0.5, "sin epocas registradas", ha="center", va="center")

    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Perdida (label smoothing incluido)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy top-1 / top-5")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("epoca")
    axes[2].set_ylabel("learning rate")
    axes[2].set_title("LR de la cabeza (warmup + coseno)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Entrenamiento Flowers-102 ({history.get('device', '?')})")
    fig.tight_layout()
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def train(cfg: Config) -> dict:
    """Entrena el clasificador y devuelve el historial (JSON serializable).

    Args:
        cfg: hiperparametros y rutas (ver `src/config.py`).

    Returns:
        Diccionario con las claves de :data:`HISTORY_KEYS`. Las listas tienen una
        entrada por epoca ejecutada; `best_val_acc`, `best_epoch`, `epochs_run`,
        `device`, `num_params` y `segundos` resumen la corrida.

    Efectos secundarios: crea `cfg.out_dir` (y `figuras/`, `predicciones/`) y
    escribe `best.pt`, `history.json`, `config_usada.json` y `figuras/curvas.png`.
    """
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()

    print("=" * 72)
    print("Entrenamiento Flowers-102")
    print(f"  dispositivo : {describe_device(device)}")
    print(f"  backbone    : {cfg.backbone} (pretrained={cfg.pretrained}, dropout={cfg.dropout})")
    print(
        f"  epocas      : {cfg.epochs} | batch: {cfg.batch_size} | img: {cfg.img_size}"
        f" | clases: {cfg.num_classes}"
    )
    print(
        f"  lr cabeza/backbone: {cfg.lr}/{cfg.backbone_lr} | warmup: {cfg.warmup_epochs}"
        f" | freeze: {cfg.freeze_epochs} | amp: {cfg.amp}"
    )
    print(f"  debug       : {cfg.debug} | salida: {Path(cfg.out_dir)}")
    print("=" * 72)

    # ---------------------------------------------------------------- datos
    train_dl, val_dl, _test_dl, class_names = build_dataloaders(cfg)
    num_workers_efectivo = int(cfg.num_workers)
    print(
        f"  lotes: train={len(train_dl)} val={len(val_dl)} | "
        f"imagenes: train={len(train_dl.dataset)} val={len(val_dl.dataset)} | "
        f"nombres de clase: {len(class_names)} | num_workers: {num_workers_efectivo}"
    )

    # ---------------------------------------------------------------- modelo
    model = build_model(
        num_classes=int(cfg.num_classes),
        backbone=str(cfg.backbone),
        pretrained=bool(cfg.pretrained),
        dropout=float(cfg.dropout),
    ).to(device)

    freeze_epochs = max(0, int(cfg.freeze_epochs))
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)  # primeras epocas: solo la cabeza
    else:
        set_backbone_trainable(model, True)

    # El optimizador SIEMPRE se construye una sola vez, con los dos grupos:
    # [0] cabeza (cfg.lr) y [1] backbone (cfg.backbone_lr), incluso si el backbone
    # esta congelado ahora mismo. Al descongelar se reutiliza el mismo objeto.
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    topk = max(1, min(int(cfg.topk), int(cfg.num_classes)))

    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    scaler = _make_grad_scaler() if amp_enabled else None
    if bool(cfg.amp) and not amp_enabled:
        print("  AMP solicitado pero desactivado: el device no es CUDA (CPU).")
    if amp_enabled:
        print("  AMP activado (autocast fp16 + GradScaler).")

    # ------------------------------------------------------------- historial
    history: dict[str, Any] = {key: [] for key in HISTORY_LIST_KEYS}
    best_val_acc = -1.0
    best_epoch = -1
    start = time.perf_counter()
    epochs = max(0, int(cfg.epochs))
    epochs_sin_mejora = 0

    def _etapa(etapa: str, epoch: int) -> tuple[float, ...]:
        """Corre una etapa de la epoca leyendo los loaders ACTUALES del scope."""
        if etapa == "train":
            return _run_train_epoch(
                model, train_dl, criterion, optimizer, scaler, device, amp_enabled, epoch, epochs
            )
        return _run_eval_epoch(model, val_dl, criterion, device, amp_enabled, topk=topk)

    def _etapa_con_fallback(etapa: str, epoch: int) -> tuple[float, ...]:
        """Como `_etapa`, pero si fallan los workers del DataLoader reintenta UNA vez.

        El reintento reconstruye los loaders con ``num_workers=0`` (mismo
        `data_root`, mismos subconjuntos) y vuelve a correr SOLO esa etapa, de modo
        que no se pierde el entrenamiento ya hecho. En Colab/Linux no se activa
        nunca: alli ``num_workers=2`` funciona y acelera.
        """
        nonlocal num_workers_efectivo, train_dl, val_dl, _test_dl, class_names
        try:
            return _etapa(etapa, epoch)
        except Exception as exc:  # noqa: BLE001 - se re-lanza si no es de workers
            if num_workers_efectivo <= 0 or not _es_error_de_workers(exc):
                raise
            mensaje = (
                f"La etapa '{etapa}' fallo con num_workers={num_workers_efectivo} "
                f"({type(exc).__name__}: {exc}). Se reintenta una sola vez con "
                "num_workers=0 (en Colab/Linux los workers funcionan; en el sandbox "
                "de Windows no pueden abrir named pipes)."
            )
            print("Aviso: " + mensaje)
            warnings.warn(mensaje, RuntimeWarning, stacklevel=2)

            data_cfg = replace(cfg, num_workers=0)
            train_dl, val_dl, _test_dl, class_names = build_dataloaders(data_cfg)
            num_workers_efectivo = 0
            return _etapa(etapa, epoch)  # unico reintento: si falla, propaga

    for epoch_index in range(epochs):
        epoch = epoch_index + 1  # 1-based (ver docstring del modulo)
        congelado = epoch <= freeze_epochs
        set_backbone_trainable(model, not congelado)

        train_loss, train_acc = _etapa_con_fallback("train", epoch)
        val_loss, val_acc, val_top5 = _etapa_con_fallback("val", epoch)

        # LR del grupo de la cabeza (indice 0) USADO en esta epoca: se lee antes
        # del `scheduler.step()` para que la curva de `history["lr"]` muestre el LR
        # que realmente se aplico (warmup subiendo, coseno bajando) en vez del que
        # se aplicara en la epoca siguiente.
        head_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["epochs"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_top5"].append(float(val_top5))
        history["lr"].append(head_lr)

        mejora = val_acc > best_val_acc
        marca = ""
        if mejora:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            epochs_sin_mejora = 0
            save_checkpoint(
                cfg.checkpoint_path,
                model,
                class_names,
                cfg,
                epoch=epoch,
                val_acc=val_acc,
            )
            marca = " *mejor"
        else:
            epochs_sin_mejora += 1

        estado = "backbone congelado" if congelado else "backbone entrenando"
        print(
            f"[{epoch:>3}/{epochs}] {estado} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} top5 {val_top5:.4f} | "
            f"lr {head_lr:.3e}{marca}"
        )

        if epochs_sin_mejora >= max(1, int(cfg.early_stop_patience)):
            print(
                f"Early stopping: {epochs_sin_mejora} epocas sin mejorar val_acc "
                f"(paciencia {cfg.early_stop_patience}); mejor = {best_val_acc:.4f} "
                f"en la epoca {best_epoch}."
            )
            break

    segundos = time.perf_counter() - start

    # Al terminar, el backbone queda entrenable: `num_params` refleja el modelo
    # completo tal como se guardo en el checkpoint.
    set_backbone_trainable(model, True)
    num_params = int(count_params(model))

    history["best_val_acc"] = float(best_val_acc)
    history["best_epoch"] = int(best_epoch)
    history["epochs_run"] = int(len(history["epochs"]))
    history["device"] = describe_device(device)
    history["num_params"] = num_params
    history["segundos"] = float(segundos)

    # ------------------------------------------------------------ artefactos
    save_json(cfg.history_path, history)
    save_json(cfg.config_path, cfg.to_dict())
    curvas = plot_history(history, Path(cfg.figures_dir()) / "curvas.png")

    print("-" * 72)
    print(
        f"Listo en {segundos:.1f} s ({segundos / 60.0:.1f} min) | epocas ejecutadas: "
        f"{history['epochs_run']}/{epochs}"
    )
    print(f"Mejor val_acc: {best_val_acc:.4f} (epoca {best_epoch})")
    print(f"Checkpoint   : {cfg.checkpoint_path} (existe: {Path(cfg.checkpoint_path).is_file()})")
    print(f"Historial    : {cfg.history_path}")
    print(f"Config usada : {cfg.config_path}")
    print(f"Curvas       : {curvas}")
    print(f"Parametros   : {num_params:,}")
    print("=" * 72)

    return history


# --------------------------------------------------------------------------- #
# CLI (seccion 3.b del contrato)
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    """Argumentos de `python -m src.train` (CLI congelada)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description=(
            "Entrena el clasificador de flores (Flowers-102) segun CONTRATOS.md seccion 3.b. "
            "Sin argumentos usa los valores por defecto de src/config.py."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python -m src.train --debug --device cpu\n"
            "  python -m src.train --epochs 5 --backbone resnet50 --out-dir resultados/prueba\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=None, help="numero de epocas")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size", help="tamano de lote")
    parser.add_argument("--backbone", type=str, default=None, help="efficientnet_b0 | resnet50 | mobilenet_v3_large")
    parser.add_argument("--data-root", type=str, default=None, dest="data_root", help="carpeta del dataset")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir", help="carpeta de resultados")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="dispositivo")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        dest="num_workers",
        help=(
            "procesos del DataLoader (default de Config: 2, lo correcto en Colab). "
            "Usa 0 en entornos donde multiprocessing este bloqueado"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="subconjunto pequeno (smoke test)")
    parser.add_argument("--seed", type=int, default=None, help="semilla")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """`Config` base + solo los argumentos informados (`dataclasses.replace`).

    Comodidad documentada (no cambia ningun default de `Config`): si se pide
    `--debug` y no se indica `--out-dir`, la salida va a `resultados/debug` para no
    pisar los resultados definitivos de `resultados/` (decision §5 del contrato).
    """
    overrides: dict[str, Any] = {}
    for campo in ("epochs", "batch_size", "backbone", "data_root", "out_dir", "device", "seed", "num_workers"):
        valor = getattr(args, campo, None)
        if valor is not None:
            overrides[campo] = valor
    if getattr(args, "debug", False):
        overrides["debug"] = True
        if "out_dir" not in overrides:
            overrides["out_dir"] = DEBUG_OUT_DIR
            print(
                f"Aviso: --debug sin --out-dir -> la salida va a '{DEBUG_OUT_DIR}' "
                "para no pisar los resultados definitivos de 'resultados/'."
            )

    cfg = replace(Config(), **overrides)

    # Avisos utiles: los valores que Config.__post_init__ ajusta por el modo debug.
    if cfg.debug:
        for campo in ("epochs", "freeze_epochs", "warmup_epochs", "amp", "num_workers"):
            informado = overrides.get(campo)
            if informado is not None and informado != getattr(cfg, campo):
                print(
                    f"Aviso: modo debug ajusto {campo}: {informado} -> {getattr(cfg, campo)}"
                )
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI. Devuelve el codigo de salida."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover - entrada del CLI
    raise SystemExit(main())
