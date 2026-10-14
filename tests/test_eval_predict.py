"""Pruebas de `src/evaluate.py` y `src/predict.py` (SA-06).

Criterio de aceptacion del contrato (seccion 4):
  * `evaluate(checkpoint, Config(debug=True), split="test")` escribe `metrics.json`
    con `top1` y `top5` numericos, las dos figuras y el CSV de predicciones.
  * `predict_image(imagen_sintetica, checkpoint)` devuelve pares `(str, float)` con
    probabilidades decrecientes, recorta `topk` a `num_classes` y respeta `min_conf`.

Nada de red y nada del dataset real: el checkpoint es sintetico (4 clases,
`mobilenet_v3_large` con `pretrained=False`) y el DataLoader se inyecta con el
parametro opcional `dataloader` de `evaluate`.
Se ejecuta con:
    .venv\\Scripts\\python.exe -m pytest tests\\test_eval_predict.py -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Permite `import src...` aunque pytest se ejecute desde otro directorio.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

torch.set_num_threads(2)  # no saturar la maquina mientras otros subagentes trabajan

import src.evaluate as evaluate_mod  # noqa: E402
import src.predict as predict_mod  # noqa: E402
from src.config import Config  # noqa: E402
from src.evaluate import MENSAJE_ENTRENA, evaluate  # noqa: E402
from src.model import build_model, save_checkpoint  # noqa: E402
from src.predict import NO_RECONOCIDA, limpiar_cache, predict_batch, predict_image  # noqa: E402

N_CLASES = 4
NOMBRES = ["rosa", "tulipan", "girasol", "margarita"]
NOMBRES8 = [f"especie_{i + 1:02d}" for i in range(8)]
IMG_SIZE = 64  # pequeno a proposito: los tests tienen que ser rapidos en CPU
N_IMAGENES = 40  # 10 imagenes por clase
BACKBONE = "mobilenet_v3_large"  # el mas rapido de los tres soportados


# --------------------------------------------------------------------------- #
# Utilidades de test
# --------------------------------------------------------------------------- #
class _DatasetSintetico(Dataset):
    """`N_IMAGENES` tensores aleatorios con etiquetas ciclicas 0..n_clases-1."""

    def __init__(self, n: int = N_IMAGENES, n_clases: int = N_CLASES, img_size: int = IMG_SIZE, seed: int = 7):
        generador = torch.Generator().manual_seed(seed)
        self.imagenes = torch.rand(n, 3, img_size, img_size, generator=generador)
        self.etiquetas = [i % n_clases for i in range(n)]

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int):
        return self.imagenes[indice], self.etiquetas[indice]


def _loader_sintetico(batch_size: int = 16) -> DataLoader:
    return DataLoader(_DatasetSintetico(), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def _crear_checkpoint(
    destino: Path,
    n_clases: int = N_CLASES,
    nombres: list[str] | None = None,
    img_size: int = IMG_SIZE,
) -> tuple[Path, Config]:
    """Checkpoint sintetico real (sin red) + la `Config` que lo describe."""
    nombres = list(nombres if nombres is not None else NOMBRES[:n_clases])
    out_dir = destino / "resultados"
    cfg = Config(
        num_classes=n_clases,
        img_size=img_size,
        debug=True,  # num_workers=0, amp=False: no descarga nada ni abre workers
        out_dir=str(out_dir),
        data_root=str(destino / "data"),
        pretrained=False,
    )
    modelo = build_model(n_clases, BACKBONE, pretrained=False).eval()
    ckpt = out_dir / "best.pt"
    save_checkpoint(ckpt, modelo, nombres, cfg, epoch=1, val_acc=1.0)
    return ckpt, cfg


@pytest.fixture(autouse=True)
def _cache_limpia():
    """Cada test arranca con la cache de modelos vacia (y la deja vacia)."""
    limpiar_cache()
    yield
    limpiar_cache()


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_escribe_metricas_figuras_y_csv(tmp_path: Path) -> None:
    ckpt, cfg = _crear_checkpoint(tmp_path)

    metricas = evaluate(ckpt, cfg, split="test", dataloader=_loader_sintetico())

    assert isinstance(metricas, dict)
    for clave in (
        "split",
        "num_images",
        "top1",
        "top5",
        "macro_f1",
        "per_class",
        "per_class_f1",
        "top_confusions",
        "ruta_confusion",
        "segundos",
    ):
        assert clave in metricas, f"falta la clave {clave!r} en el dict devuelto"

    assert metricas["split"] == "test"
    assert metricas["num_images"] == N_IMAGENES
    for clave in ("top1", "top5", "macro_f1"):
        assert isinstance(metricas[clave], float), f"{clave} debe ser float"
        assert 0.0 <= metricas[clave] <= 1.0
    assert metricas["top5"] >= metricas["top1"] - 1e-9
    assert isinstance(metricas["segundos"], float) and metricas["segundos"] >= 0.0

    # per_class: dict clase -> {precision, recall, f1, support}
    assert set(metricas["per_class"]) == set(NOMBRES)
    for valores in metricas["per_class"].values():
        assert set(valores) == {"precision", "recall", "f1", "support"}
        assert all(isinstance(valores[k], float) for k in ("precision", "recall", "f1"))
        assert valores["support"] == N_IMAGENES // N_CLASES
        assert 0.0 <= valores["f1"] <= 1.0

    # per_class_f1: dict nombre -> float (lo consume el notebook de SA-03)
    assert set(metricas["per_class_f1"]) == set(NOMBRES)
    assert all(isinstance(v, float) for v in metricas["per_class_f1"].values())
    for nombre, f1 in metricas["per_class_f1"].items():
        assert f1 == pytest.approx(metricas["per_class"][nombre]["f1"])

    # top_confusions: [[real, pred, n], ...] con n descendente y maximo 20
    confusiones = metricas["top_confusions"]
    assert isinstance(confusiones, list)
    assert len(confusiones) <= 20
    assert len(confusiones) <= N_CLASES * (N_CLASES - 1)  # solo pares real != pred
    for fila in confusiones:
        assert isinstance(fila, list) and len(fila) == 3
        assert isinstance(fila[0], str) and isinstance(fila[1], str)
        assert isinstance(fila[2], int) and fila[2] >= 1
        assert fila[0] != fila[1]
    assert [fila[2] for fila in confusiones] == sorted((f[2] for f in confusiones), reverse=True)

    # metrics.json
    assert cfg.metrics_path.is_file()
    datos = json.loads(cfg.metrics_path.read_text(encoding="utf-8"))
    assert datos["split"] == "test"
    assert datos["num_images"] == N_IMAGENES
    for clave in ("top1", "top5", "macro_f1"):
        assert isinstance(datos[clave], (int, float))
    assert set(datos["per_class_f1"]) == set(NOMBRES)
    assert datos["ruta_confusion"] == str(cfg.figures_dir() / "confusion_matrix.png")

    # figuras (PNG reales, no vacias)
    matriz_png = cfg.figures_dir() / "confusion_matrix.png"
    confusiones_png = cfg.figures_dir() / "top_confusions.png"
    assert matriz_png.is_file() and matriz_png.stat().st_size > 1000
    assert confusiones_png.is_file() and confusiones_png.stat().st_size > 1000
    assert Path(metricas["ruta_confusion"]) == matriz_png

    # CSV: encabezado exacto + una fila por imagen
    ruta_csv = cfg.predictions_dir() / "test_predicciones.csv"
    assert ruta_csv.is_file()
    with open(ruta_csv, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    assert filas[0] == ["ruta", "etiqueta_real", "prediccion", "prob", "correcto"]
    assert len(filas) == N_IMAGENES + 1, "debe haber 40 filas + encabezado"
    for fila in filas[1:]:
        assert len(fila) == 5
        assert fila[1] in NOMBRES and fila[2] in NOMBRES
        assert 0.0 <= float(fila[3]) <= 1.0
        assert fila[4] in ("True", "False")


def test_evaluate_reconstruye_la_config_del_checkpoint_si_cfg_es_none(tmp_path: Path) -> None:
    ckpt, cfg = _crear_checkpoint(tmp_path)

    metricas = evaluate(ckpt, None, split="val", dataloader=_loader_sintetico())

    assert metricas["split"] == "val"
    # out_dir viene del `config` guardado en el checkpoint (no de `resultados/` del cwd)
    assert Path(metricas["checkpoint"]) == ckpt
    ruta_metricas = cfg.metrics_path
    assert ruta_metricas.is_file()
    datos = json.loads(ruta_metricas.read_text(encoding="utf-8"))
    assert datos["split"] == "val"
    assert (cfg.predictions_dir() / "val_predicciones.csv").is_file()


def test_evaluate_usa_el_img_size_del_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aunque el `cfg` del usuario diga 224, mandan los 64 px del checkpoint."""
    ckpt, cfg_ckpt = _crear_checkpoint(tmp_path, img_size=IMG_SIZE)
    capturado: dict = {}

    def _build_dataloaders_falso(cfg: Config):
        capturado["img_size"] = cfg.img_size
        cargador = _loader_sintetico()
        return cargador, cargador, cargador, list(NOMBRES)

    monkeypatch.setattr(evaluate_mod, "build_dataloaders", _build_dataloaders_falso)

    cfg_usuario = Config(
        num_classes=N_CLASES,
        img_size=224,  # distinto al del checkpoint a proposito
        debug=True,
        out_dir=str(cfg_ckpt.out_dir),
        data_root=str(tmp_path / "data"),
        pretrained=False,
    )
    metricas = evaluate(ckpt, cfg_usuario, split="test")

    assert capturado["img_size"] == IMG_SIZE
    assert metricas["img_size"] == IMG_SIZE
    assert metricas["num_images"] == N_IMAGENES


def test_evaluate_checkpoint_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate(tmp_path / "no_existe.pt", Config(debug=True))
    mensaje = str(excinfo.value)
    assert "entrena" in mensaje
    assert "python -m src.train" in mensaje
    assert MENSAJE_ENTRENA in mensaje


def test_evaluate_split_invalido(tmp_path: Path) -> None:
    ckpt, cfg = _crear_checkpoint(tmp_path)
    with pytest.raises(ValueError):
        evaluate(ckpt, cfg, split="entrenamiento")


# --------------------------------------------------------------------------- #
# Robustez de los workers del DataLoader (sandbox/Windows: named pipes bloqueados)
# --------------------------------------------------------------------------- #
class _LoaderQueFalla:
    """DataLoader falso: iterar con workers lanza PermissionError como en Windows."""

    dataset = None

    def __iter__(self):
        raise PermissionError("[WinError 5] Acceso denegado")


def _cfg_con_workers(cfg_base: Config, num_workers: int) -> Config:
    """Copia de ``cfg_base`` SIN debug (debug fuerza num_workers=0) y con N workers."""
    return Config(
        num_classes=cfg_base.num_classes,
        img_size=cfg_base.img_size,
        debug=False,
        num_workers=num_workers,
        out_dir=str(cfg_base.out_dir),
        data_root=str(cfg_base.data_root),
        pretrained=False,
    )


def test_evaluate_reintenta_con_num_workers_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con `num_workers>0` que falla por permisos, `evaluate` reintenta con 0 y termina."""
    ckpt, cfg_base = _crear_checkpoint(tmp_path)
    cfg = _cfg_con_workers(cfg_base, 2)
    intentos: list[int] = []

    def _build_dataloaders_falso(cfg_actual: Config):
        intentos.append(int(cfg_actual.num_workers))
        cargador = _LoaderQueFalla() if int(cfg_actual.num_workers) > 0 else _loader_sintetico()
        return cargador, cargador, cargador, list(NOMBRES)

    monkeypatch.setattr(evaluate_mod, "build_dataloaders", _build_dataloaders_falso)

    with pytest.warns(RuntimeWarning, match="num_workers"):
        metricas = evaluate(ckpt, cfg, split="test")

    assert intentos == [2, 0], "debe reintentar una sola vez con num_workers=0"
    assert metricas["num_images"] == N_IMAGENES
    assert cfg.metrics_path.is_file()
    assert (cfg.figures_dir() / "confusion_matrix.png").is_file()
    assert (cfg.predictions_dir() / "test_predicciones.csv").is_file()


def test_evaluate_no_enmascara_errores_ajenos_a_los_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un error real del modelo no se reintenta ni se oculta."""
    ckpt, cfg_base = _crear_checkpoint(tmp_path)
    cfg = _cfg_con_workers(cfg_base, 2)

    class _LoaderRoto:
        dataset = None

        def __iter__(self):
            raise ValueError("forma de tensor inesperada")

    def _build_dataloaders_falso(cfg_actual: Config):
        cargador = _LoaderRoto()
        return cargador, cargador, cargador, list(NOMBRES)

    monkeypatch.setattr(evaluate_mod, "build_dataloaders", _build_dataloaders_falso)

    with pytest.raises(ValueError, match="forma de tensor"):
        evaluate(ckpt, cfg, split="test")


def test_evaluate_con_num_workers_0_no_reintenta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El reintento solo aplica cuando cfg.num_workers > 0."""
    ckpt, cfg_base = _crear_checkpoint(tmp_path)
    cfg = _cfg_con_workers(cfg_base, 0)
    assert cfg.num_workers == 0

    def _build_dataloaders_falso(cfg_actual: Config):
        cargador = _LoaderQueFalla()
        return cargador, cargador, cargador, list(NOMBRES)

    monkeypatch.setattr(evaluate_mod, "build_dataloaders", _build_dataloaders_falso)

    with pytest.raises(PermissionError):
        evaluate(ckpt, cfg, split="test")


def test_cli_evaluate_acepta_num_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    capturado: dict = {}

    def _build_dataloaders_falso(cfg: Config):
        capturado["num_workers"] = cfg.num_workers
        cargador = _loader_sintetico()
        return cargador, cargador, cargador, list(NOMBRES)

    monkeypatch.setattr(evaluate_mod, "build_dataloaders", _build_dataloaders_falso)

    assert evaluate_mod.main(
        ["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "s1"), "--device", "cpu", "--num-workers", "0"]
    ) == 0
    assert capturado["num_workers"] == 0

    # Sin el flag sigue mandando el default de Config (2): el flag es opcional.
    assert evaluate_mod.main(
        ["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "s2"), "--device", "cpu"]
    ) == 0
    assert capturado["num_workers"] == Config().num_workers == 2

    # Valor invalido: mensaje claro, sin traza.
    assert evaluate_mod.main(["--checkpoint", str(ckpt), "--num-workers", "-1"]) == 1


# --------------------------------------------------------------------------- #
# predict_image
# --------------------------------------------------------------------------- #
def _imagen_roja() -> Image.Image:
    return Image.new("RGB", (300, 220), "red")


def test_predict_image_devuelve_pares_decrecientes(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)

    resultados = predict_image(_imagen_roja(), ckpt, topk=3, min_conf=0.0, device="cpu")

    assert len(resultados) == 3, "debe devolver exactamente topk pares"
    assert all(isinstance(nombre, str) for nombre, _ in resultados)
    assert all(isinstance(prob, float) for _, prob in resultados)

    probabilidades = [prob for _, prob in resultados]
    assert probabilidades == sorted(probabilidades, reverse=True), "deben ser decrecientes"
    assert all(0.0 <= p <= 1.0 for p in probabilidades)
    assert sum(probabilidades) <= 1.0001, "suma de probabilidades softmax <= 1"
    assert all(nombre in NOMBRES for nombre, _ in resultados)


def test_predict_image_min_conf_imposible(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)

    resultados = predict_image(_imagen_roja(), ckpt, topk=5, min_conf=1.1, device="cpu")

    assert len(resultados) == 1
    assert resultados[0][0] == NO_RECONOCIDA == "no reconocida / fuera del catálogo"
    assert isinstance(resultados[0][1], float)
    assert 0.0 <= resultados[0][1] <= 1.0


def test_predict_topk_mayor_que_num_classes(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path, n_clases=N_CLASES, nombres=NOMBRES)

    resultados = predict_image(_imagen_roja(), ckpt, topk=999, min_conf=0.0, device="cpu")

    assert len(resultados) == N_CLASES, "topk se recorta a num_classes sin fallar"
    assert len({nombre for nombre, _ in resultados}) == N_CLASES


def test_predict_topk_no_supera_ocho_clases(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path, n_clases=8, nombres=NOMBRES8)

    resultados = predict_image(_imagen_roja(), ckpt, topk=999, min_conf=0.0, device="cpu")

    assert len(resultados) == 8


def test_predict_acepta_todos_los_formatos(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    roja = _imagen_roja()
    ruta = tmp_path / "roja.png"
    roja.save(ruta)
    arreglo = np.asarray(roja)

    referencia = predict_image(roja, ckpt, topk=2, min_conf=0.0, device="cpu")

    entradas = [
        ruta,  # Path
        str(ruta),  # str
        roja,  # PIL.Image
        arreglo,  # numpy uint8 (H, W, 3)
        (arreglo.astype("float32") / 255.0),  # numpy float en [0, 1]
        arreglo.transpose(2, 0, 1),  # numpy channel-first uint8
        {"path": str(ruta)},  # dict tipo Gradio
        (arreglo, str(ruta)),  # tupla que devuelve Gradio
        [str(ruta)],  # lista de un elemento
    ]
    for entrada in entradas:
        resultados = predict_image(entrada, ckpt, topk=2, min_conf=0.0, device="cpu")
        assert [nombre for nombre, _ in resultados] == [nombre for nombre, _ in referencia], (
            f"resultado distinto para {type(entrada)!r}"
        )
        assert [round(p, 6) for _, p in resultados] == [round(p, 6) for _, p in referencia]


def test_predict_checkpoint_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        predict_image(_imagen_roja(), tmp_path / "no_existe.pt", device="cpu")
    assert "entrena" in str(excinfo.value)


def test_predict_imagen_inexistente(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        predict_image(tmp_path / "no_existe.jpg", ckpt, device="cpu")


def test_predict_usa_la_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)

    contador = {"llamadas": 0}
    load_real = predict_mod.load_checkpoint

    def _load_contado(*args, **kwargs):
        contador["llamadas"] += 1
        return load_real(*args, **kwargs)

    monkeypatch.setattr(predict_mod, "load_checkpoint", _load_contado)
    limpiar_cache()

    primero = predict_image(_imagen_roja(), ckpt, topk=2, min_conf=0.0, device="cpu")
    segundo = predict_image(_imagen_roja(), ckpt, topk=2, min_conf=0.0, device="cpu")

    assert contador["llamadas"] == 1, "la segunda llamada debe salir de la cache"
    assert primero == segundo

    # El modelo cacheado es literalmente el mismo objeto.
    modelo_a = predict_mod._modelo_cacheado(str(ckpt), "cpu")[0]
    modelo_b = predict_mod._modelo_cacheado(str(ckpt), "cpu")[0]
    assert modelo_a is modelo_b
    assert predict_mod._modelo_cacheado.cache_info().currsize == 1

    # limpiar_cache() vacia la cache y fuerza una recarga.
    limpiar_cache()
    assert predict_mod._modelo_cacheado.cache_info().currsize == 0
    predict_image(_imagen_roja(), ckpt, topk=2, min_conf=0.0, device="cpu")
    assert contador["llamadas"] == 2


def test_predict_batch(tmp_path: Path) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    rutas = []
    for indice, color in enumerate(("red", "green", "blue")):
        ruta = tmp_path / f"imagen_{indice}.png"
        Image.new("RGB", (120, 90), color).save(ruta)
        rutas.append(ruta)

    lotes = predict_batch(rutas, ckpt, topk=2, device="cpu")

    assert len(lotes) == 3
    for lote in lotes:
        assert isinstance(lote, list) and lote
        for nombre, probabilidad in lote:
            assert isinstance(nombre, str)
            assert isinstance(probabilidad, float)
            assert 0.0 <= probabilidad <= 1.0
    assert all(len(lote) <= 2 for lote in lotes)


# --------------------------------------------------------------------------- #
# CLI (se prueban las funciones main/build_parser: sin subprocess ni tuberias)
# --------------------------------------------------------------------------- #
def test_cli_evaluate_help_y_checkpoint_ausente(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        evaluate_mod.main(["--help"])
    assert excinfo.value.code == 0
    assert "--split" in capsys.readouterr().out

    codigo = evaluate_mod.main(["--checkpoint", str(tmp_path / "no_existe.pt")])
    salida = capsys.readouterr().out
    assert codigo == 1
    assert "entrena" in salida.lower()
    assert "python -m src.train" in salida

    assert evaluate_mod.build_parser().parse_args(["--split", "val"]).split == "val"


def test_cli_evaluate_ejecuta_sobre_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    monkeypatch.setattr(evaluate_mod, "build_dataloaders", lambda cfg: (_loader_sintetico(),) * 3 + (list(NOMBRES),))

    codigo = evaluate_mod.main(
        ["--checkpoint", str(ckpt), "--split", "test", "--out-dir", str(tmp_path / "salida"), "--device", "cpu", "--debug"]
    )

    assert codigo == 0
    salida = capsys.readouterr().out
    assert "top1=" in salida
    assert (tmp_path / "salida" / "metrics.json").is_file()


def test_cli_predict_help_y_checkpoint_ausente(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        predict_mod.main(["--help"])
    assert excinfo.value.code == 0
    assert "--image" in capsys.readouterr().out

    ruta = tmp_path / "roja.png"
    _imagen_roja().save(ruta)
    codigo = predict_mod.main(["--image", str(ruta), "--checkpoint", str(tmp_path / "no_existe.pt")])
    salida = capsys.readouterr().out
    assert codigo == 1
    assert "entrena" in salida.lower()
    assert "python -m src.train" in salida


def test_cli_predict_imprime_el_topk(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ckpt, _ = _crear_checkpoint(tmp_path)
    ruta = tmp_path / "roja.png"
    _imagen_roja().save(ruta)

    codigo = predict_mod.main(
        ["--image", str(ruta), "--checkpoint", str(ckpt), "--topk", "4", "--min-conf", "0.0", "--device", "cpu"]
    )

    salida = capsys.readouterr().out
    assert codigo == 0
    assert "imagen:" in salida and "checkpoint:" in salida
    assert sum(1 for linea in salida.splitlines() if linea.strip().startswith(("1.", "2.", "3.", "4."))) == 4
    for nombre in NOMBRES:
        assert nombre in salida
