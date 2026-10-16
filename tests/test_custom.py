"""Tests de ``src/custom.py`` (entrenamiento con flores propias).

Cubre lo pedido por el contrato del modulo:

  1. ``revisar_carpeta`` con carpeta inexistente, vacia, con una sola especie, con
     clases desequilibradas, con fotos sueltas y con un caso correcto (3 especies x
     8 fotos sinteticas).
  2. ``preparar_loaders``: nombres de clase ordenados, forma de lote
     ``(B, 3, 224, 224)`` y **verificacion explicita de que no hay fuga** entre
     train y val (se comparan los nombres de archivo reales, no indices).
  3. ``entrenar_custom`` de punta a punta con un catalogo sintetico pequeno
     (``epochs=1``, ``backbone="mobilenet_v3_large"``, ``pretrained=False``,
     ``debug=True``): ``best.pt``, ``history.json`` con las 13 claves,
     ``catalogo.json`` con las clases correctas y checkpoint recargable con
     ``src.model.load_checkpoint``.
  4. ``evaluar_custom`` sobre ese checkpoint: ``top1``/``top5``/``macro_f1`` y
     ``metrics.json`` en disco.
  5. Un checkpoint propio funciona con ``src.predict.predict_image`` sin tocar
     nada: las etiquetas devueltas pertenecen al catalogo propio.
  6. CLI ``--revisar``: codigo de salida 0 y salida con el numero de fotos.

Los tests 4 y 5 entrenan con ``img_size=96`` a proposito: lo que se comprueba es la
integracion (checkpoint -> evaluacion/prediccion), no el tamano de imagen, y asi la
suite se mantiene rapida.

Comando de verificacion (peculiaridades del entorno: ver la seccion 1 del contrato):

    $env:TMP = (Resolve-Path .pytmp).Path; $env:TEMP = $env:TMP
    $env:MPLCONFIGDIR = (Resolve-Path .pytmp).Path
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_custom.py -q

La fixture ``tmp_path`` la sustituye ``tests/conftest.py`` (crea los directorios con
permisos por defecto porque en este sandbox los directorios 0o700 quedan
inaccesibles).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)  # CPU: no acaparar nucleos mientras otros subagentes trabajan

from src import custom as custom_mod  # noqa: E402
from src.config import Config  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.predict import limpiar_cache, predict_image  # noqa: E402
from src.train import HISTORY_KEYS  # noqa: E402

#: Claves exactas que devuelve ``revisar_carpeta`` (contrato congelado).
CLAVES_INFORME = {
    "ok",
    "ruta",
    "clases",
    "total",
    "min_por_clase",
    "avisos",
    "problemas",
    "recomendacion",
}

#: Clases (especies) del catalogo sintetico, ya en orden alfabetico.
CLASES = ("margarita", "rosa", "tulipan")


# --------------------------------------------------------------------- helpers
def _catalogo(
    base: Path,
    clases: tuple[str, ...] = CLASES,
    por_clase=8,
    tamano: int = 48,
) -> Path:
    """Crea un catalogo sintetico ``<especie>/<foto>.png`` y devuelve la raiz.

    ``por_clase`` admite un entero (todas las especies iguales) o un diccionario
    ``{especie: n}`` para provocar desequilibrios.
    """
    from PIL import Image

    base.mkdir(parents=True, exist_ok=True)
    for indice, nombre in enumerate(clases):
        cantidad = int(por_clase[nombre]) if isinstance(por_clase, dict) else int(por_clase)
        carpeta = base / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        for k in range(cantidad):
            color = (
                (indice * 70 + 30) % 256,
                (k * 23 + 10) % 256,
                (indice * 40 + 90) % 256,
            )
            Image.new("RGB", (tamano, tamano), color).save(carpeta / f"{nombre}_{k:02d}.png")
    return base


def _cfg_rapida(**extra) -> Config:
    """``Config`` de smoke test: sin descargas, sin workers y con pocas epocas."""
    valores = {
        "epochs": 1,
        "batch_size": 8,
        "backbone": "mobilenet_v3_large",
        "pretrained": False,
        "debug": True,
    }
    valores.update(extra)
    return Config(**valores)


def _nombres_de_loader(loader) -> set[str]:
    """Nombres de archivo reales que sirve un ``DataLoader`` de ``Subset``."""
    subset = loader.dataset
    base = subset.dataset
    return {Path(base.samples[indice][0]).name for indice in subset.indices}


# ------------------------------------------------------------------ 1. revision
def test_revisar_carpeta_inexistente(tmp_path):
    informe = custom_mod.revisar_carpeta(tmp_path / "no_existe")
    assert set(informe) == CLAVES_INFORME
    assert informe["ok"] is False
    assert informe["total"] == 0
    assert informe["clases"] == {}
    assert any("no existe" in problema for problema in informe["problemas"])
    assert informe["recomendacion"]


def test_revisar_carpeta_vacia(tmp_path):
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    informe = custom_mod.revisar_carpeta(vacia)
    assert informe["ok"] is False
    assert informe["total"] == 0
    assert informe["problemas"]
    assert any("ninguna foto" in problema for problema in informe["problemas"])


def test_revisar_carpeta_una_sola_especie(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", clases=("rosa",), por_clase=8)
    informe = custom_mod.revisar_carpeta(catalogo)
    assert informe["ok"] is False
    assert informe["clases"] == {"rosa": 8}
    assert informe["total"] == 8
    assert any("2 especies" in problema for problema in informe["problemas"])


def test_revisar_carpeta_desequilibrada(tmp_path):
    catalogo = _catalogo(
        tmp_path / "fotos", clases=("rosa", "tulipan"), por_clase={"rosa": 30, "tulipan": 8}
    )
    # min_por_clase=5 para aislar el aviso de desequilibrio del de "pocas fotos".
    informe = custom_mod.revisar_carpeta(catalogo, min_por_clase=5)
    assert informe["ok"] is True
    assert informe["total"] == 38
    assert any("desequilibrad" in aviso for aviso in informe["avisos"])
    assert informe["problemas"] == []


def test_revisar_carpeta_fotos_sueltas(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", clases=("rosa", "tulipan"), por_clase=8)
    from PIL import Image

    Image.new("RGB", (32, 32), "green").save(catalogo / "suelta.png")
    informe = custom_mod.revisar_carpeta(catalogo)
    assert informe["ok"] is False
    assert any("suelta" in problema for problema in informe["problemas"])
    # La foto suelta NO se cuenta como foto de ninguna especie.
    assert informe["total"] == 16


def test_revisar_carpeta_correcta(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", por_clase=8)
    informe = custom_mod.revisar_carpeta(catalogo)
    assert informe["ok"] is True
    assert informe["problemas"] == []
    assert informe["clases"] == {"margarita": 8, "rosa": 8, "tulipan": 8}
    assert informe["total"] == 24
    assert informe["ruta"] == str(catalogo)
    # 8 < 10 (min_por_clase por defecto) -> aviso, pero el catalogo es valido.
    assert any("menos de 10" in aviso for aviso in informe["avisos"])
    # Con un minimo mas bajo no hay nada que avisar y la recomendacion invita a entrenar.
    informe_estricto = custom_mod.revisar_carpeta(catalogo, min_por_clase=5)
    assert informe_estricto["ok"] is True
    assert informe_estricto["avisos"] == []
    assert "Entrena con" in informe_estricto["recomendacion"]


# --------------------------------------------------------- 2. preparar_loaders
def test_preparar_loaders_clases_y_forma_de_lote(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos")
    cfg = _cfg_rapida()
    paquete = custom_mod.preparar_loaders(cfg, ruta_fotos=catalogo, val_fraccion=0.25)

    assert paquete["class_names"] == list(CLASES)  # ordenadas alfabeticamente
    assert set(paquete) == {"train", "val", "class_names", "n_por_clase", "rutas_val", "informe"}
    assert paquete["informe"]["total"] == 24

    imagenes, etiquetas = next(iter(paquete["train"]))
    assert tuple(imagenes.shape) == (8, 3, 224, 224)
    assert etiquetas.shape == (8,)

    imagenes_val, etiquetas_val = next(iter(paquete["val"]))
    assert tuple(imagenes_val.shape)[1:] == (3, 224, 224)
    assert etiquetas_val.shape[0] == len(paquete["rutas_val"])

    for nombre, reparto in paquete["n_por_clase"].items():
        assert nombre in CLASES
        assert reparto["total"] == 8
        assert reparto["val"] >= 1
        assert reparto["train"] == 8 - reparto["val"]
    assert len(paquete["rutas_val"]) == sum(r["val"] for r in paquete["n_por_clase"].values())


def test_preparar_loaders_sin_fuga_entre_train_y_val(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos")
    cfg = _cfg_rapida()
    paquete = custom_mod.preparar_loaders(
        cfg, ruta_fotos=catalogo, val_fraccion=0.25, semilla=7
    )

    nombres_train = _nombres_de_loader(paquete["train"])
    nombres_val = _nombres_de_loader(paquete["val"])
    assert nombres_train and nombres_val
    # Sin fuga: ninguna foto puede estar en train y en val a la vez.
    assert nombres_train.isdisjoint(nombres_val)
    # Y el reparto cubre TODAS las fotos del catalogo, sin perder ninguna.
    todas = {foto.name for foto in catalogo.rglob("*.png")}
    assert nombres_train | nombres_val == todas
    # Las rutas de val que devuelve el paquete son las mismas fotos que sirve el loader.
    assert {Path(ruta).name for ruta in paquete["rutas_val"]} == nombres_val
    # `rutas_val` sale del ImageFolder de val: mismos indices, mismas fotos.
    assert len(paquete["rutas_val"]) == len(nombres_val)

    # El reparto es deterministico: misma semilla -> mismo val (y sin solape).
    repetido = custom_mod.preparar_loaders(
        cfg, ruta_fotos=catalogo, val_fraccion=0.25, semilla=7
    )
    assert repetido["rutas_val"] == paquete["rutas_val"]


def test_preparar_loaders_val_fraccion_invalida(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos")
    with pytest.raises(ValueError):
        custom_mod.preparar_loaders(_cfg_rapida(), ruta_fotos=catalogo, val_fraccion=0.0)
    with pytest.raises(ValueError):
        custom_mod.preparar_loaders(_cfg_rapida(), ruta_fotos=catalogo, val_fraccion=1.0)


# ---------------------------------------------------------- 3. entrenar_custom
def test_entrenar_custom_punta_a_punta(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", por_clase=8)
    salida = tmp_path / "salida"
    cfg = _cfg_rapida()  # epochs=1, mobilenet_v3_large, pretrained=False, debug=True

    resultado = custom_mod.entrenar_custom(
        cfg, ruta_fotos=catalogo, val_fraccion=0.25, out_dir=salida
    )

    assert set(resultado) == {"checkpoint", "class_names", "historial", "informe", "out_dir"}
    assert resultado["out_dir"] == salida
    assert resultado["checkpoint"] == salida / "best.pt"
    assert resultado["checkpoint"].is_file()
    assert resultado["class_names"] == list(CLASES)

    # history.json con las 13 claves del contrato de src.train.
    historial = json.loads((salida / "history.json").read_text(encoding="utf-8"))
    assert len(HISTORY_KEYS) == 13
    assert set(historial) == set(HISTORY_KEYS)
    assert historial["epochs_run"] == 1
    assert set(resultado["historial"]) == set(HISTORY_KEYS)

    # catalogo.json con la trazabilidad del catalogo propio.
    catalogo_json = json.loads((salida / "catalogo.json").read_text(encoding="utf-8"))
    assert catalogo_json["clases"] == list(CLASES)
    assert set(catalogo_json["fotos_por_clase"]) == set(CLASES)
    assert catalogo_json["total"] == 24
    assert catalogo_json["backbone"] == "mobilenet_v3_large"
    assert catalogo_json["ruta_fotos"] == str(catalogo)
    assert catalogo_json["val_fraccion"] == 0.25
    assert catalogo_json["semilla"] == cfg.seed
    assert catalogo_json["fecha"]
    assert catalogo_json["num_classes"] == 3

    # El checkpoint se recarga con el cargador del proyecto, sin nada especial.
    modelo, nombres, meta = load_checkpoint(resultado["checkpoint"])
    assert meta["num_classes"] == 3
    assert nombres == list(CLASES)
    assert [str(n) for n in meta["class_names"]] == list(CLASES)
    assert int(meta["img_size"]) == 224
    salida_modelo = modelo(torch.zeros(2, 3, 224, 224))
    assert tuple(salida_modelo.shape) == (2, 3)


# ----------------------------------------------------------- 4. evaluar_custom
def test_evaluar_custom_metricas(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", por_clase=8, tamano=32)
    salida = tmp_path / "salida"
    cfg = _cfg_rapida(img_size=96)
    resultado = custom_mod.entrenar_custom(cfg, ruta_fotos=catalogo, out_dir=salida)

    cfg_eval = Config(out_dir=str(salida), pretrained=False, debug=True, img_size=96)
    metricas = custom_mod.evaluar_custom(
        resultado["checkpoint"], cfg_eval, ruta_fotos=catalogo, split="val"
    )

    assert {"top1", "top5", "macro_f1"} <= set(metricas)
    assert metricas["split"] == "val"
    assert metricas["num_classes"] == 3
    assert metricas["img_size"] == 96
    assert 0.0 <= float(metricas["top1"]) <= 1.0
    assert 0.0 <= float(metricas["top5"]) <= 1.0
    assert 0.0 <= float(metricas["macro_f1"]) <= 1.0

    guardado = salida / "metrics.json"
    assert guardado.is_file()
    assert json.loads(guardado.read_text(encoding="utf-8"))["top1"] == metricas["top1"]


# ------------------------------------------- 5. predict_image con modelo propio
def test_predict_image_con_checkpoint_propio(tmp_path):
    catalogo = _catalogo(tmp_path / "fotos", por_clase=8, tamano=32)
    salida = tmp_path / "salida"
    cfg = _cfg_rapida(img_size=96)
    resultado = custom_mod.entrenar_custom(cfg, ruta_fotos=catalogo, out_dir=salida)

    limpiar_cache()  # el checkpoint es nuevo: no reutilizar un modelo cacheado
    foto = sorted(catalogo.rglob("*.png"))[0]
    predicciones = predict_image(
        foto, resultado["checkpoint"], topk=5, min_conf=0.0, device="cpu"
    )

    assert predicciones
    etiquetas = [nombre for nombre, _ in predicciones]
    assert set(etiquetas) <= set(resultado["class_names"])
    assert len(etiquetas) == 3  # topk se recorta a las 3 especies del catalogo
    assert abs(sum(prob for _, prob in predicciones) - 1.0) < 1e-4


# ------------------------------------------------------------------- 6. CLI
def test_cli_revisar(tmp_path, capsys):
    catalogo = _catalogo(tmp_path / "fotos", por_clase=8)
    codigo = custom_mod.main(["--revisar", "--fotos", str(catalogo)])
    salida = capsys.readouterr().out

    assert codigo == 0
    assert "total de fotos: 24" in salida
    assert "rosa" in salida
    assert "estado        : OK" in salida


def test_cli_revisar_carpeta_inexistente(tmp_path, capsys):
    codigo = custom_mod.main(["--revisar", "--fotos", str(tmp_path / "no_existe")])
    salida = capsys.readouterr().out

    assert codigo == 1
    assert "CON PROBLEMAS" in salida


def test_cli_sin_argumentos_usa_los_defaults_del_catalogo(monkeypatch, capsys):
    """`python -m src.custom` (sin argumentos) entrena con los valores documentados."""
    capturado: dict = {}

    def _entrenar_falso(cfg=None, ruta_fotos=None, val_fraccion=None, out_dir=None):
        capturado.update(cfg=cfg, ruta_fotos=ruta_fotos, out_dir=out_dir)
        return {}

    monkeypatch.setattr(custom_mod, "entrenar_custom", _entrenar_falso)

    assert custom_mod.main([]) == 0
    assert capturado["ruta_fotos"] == custom_mod.RUTA_FOTOS_POR_DEFECTO
    assert capturado["out_dir"] == custom_mod.OUT_DIR_POR_DEFECTO
    assert capturado["cfg"].epochs == 15
    assert capturado["cfg"].batch_size == 16
    assert capturado["cfg"].backbone == "efficientnet_b0"
    assert capturado["cfg"].pretrained is True

    # --debug sin --out-dir no pisa los resultados definitivos del catalogo propio.
    assert custom_mod.main(["--debug"]) == 0
    assert capturado["out_dir"] == custom_mod.OUT_DIR_DEBUG_POR_DEFECTO
    assert "Aviso" in capsys.readouterr().out
