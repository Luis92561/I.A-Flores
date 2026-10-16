"""Pruebas de `src/app.py`: demo de Gradio con uno o dos catalogos (SA-E3).

Auditoria: `src/app.py` era el unico modulo de `src/` sin tests. Aqui se cubre

  * el contrato historico de `predecir` (sin imagen, checkpoint inexistente,
    top-k del modelo base, aviso de confianza baja),
  * la integracion del **catalogo propio** (`custom_checkpoint`): enrutado por
    confianza, ruta inexistente, checkpoint ilegible y `build_app`,
  * la CLI (`--custom-checkpoint` y su propagacion hasta `build_app`).

Nada de red, nada de dataset real y nada de `gr.Interface`: los checkpoints son
sinteticos y reales (`build_model(4, "mobilenet_v3_large", pretrained=False)` +
`save_checkpoint`), igual que en `tests/test_eval_predict.py`.

IMPORTANTE: no se usa `TestClient(app).get("/")` sin `launch()`: en Gradio 6.30
eso falla por un bug ajeno al proyecto (el `theme` solo se inicializa dentro de
`launch()`). Por eso la app se comprueba construyendola y verificando que el
objeto devuelto tiene `.launch`, que es el contrato real de `build_app`.

Ejecutar:
    .venv\\Scripts\\python.exe -m pytest tests\\test_app.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Permite `import src...` aunque pytest se ejecute desde otro directorio.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(2)  # no saturar la maquina mientras otros subagentes trabajan

import src.app as app_mod  # noqa: E402
import src.predict as predict_mod  # noqa: E402
from src.app import (  # noqa: E402
    AVISO_PROPIO_FALLA,
    AVISO_PROPIO_INEXISTENTE,
    MSG_SIN_IMAGEN,
    TEXTO_BASE,
    TEXTO_PROPIO,
    build_app,
    main,
    parse_args,
    predecir,
)
from src.config import Config  # noqa: E402
from src.model import build_model, save_checkpoint  # noqa: E402

# Clases del "modelo base" (simula las 102 especies de Oxford).
NOMBRES_BASE = ["rosa", "tulipan", "girasol", "margarita"]
# Clases del "catalogo propio": distintas a proposito para poder distinguirlas.
NOMBRES_PROPIO = ["orquidea_del_autor", "dalia_del_jardin", "loto_casero"]

N_CLASES_BASE = len(NOMBRES_BASE)
N_CLASES_PROPIO = len(NOMBRES_PROPIO)
IMG_SIZE = 64  # pequeno a proposito: los tests tienen que ser rapidos en CPU
BACKBONE = "mobilenet_v3_large"  # el mas rapido de los tres soportados


# --------------------------------------------------------------------------- #
# Utilidades de test
# --------------------------------------------------------------------------- #
def _crear_checkpoint(
    destino: Path,
    nombres: list[str],
    img_size: int = IMG_SIZE,
    backbone: str = BACKBONE,
    subcarpeta: str = "resultados",
) -> Path:
    """Checkpoint sintetico real (sin red) con las clases indicadas."""
    nombres = [str(n) for n in nombres]
    out_dir = destino / subcarpeta
    cfg = Config(
        num_classes=len(nombres),
        img_size=img_size,
        debug=True,  # num_workers=0, amp=False: no descarga nada
        out_dir=str(out_dir),
        data_root=str(destino / "data"),
        pretrained=False,
    )
    modelo = build_model(len(nombres), backbone, pretrained=False).eval()
    ckpt = out_dir / "best.pt"
    save_checkpoint(ckpt, modelo, nombres, cfg, epoch=1, val_acc=1.0)
    return ckpt


def _imagen_roja() -> Image.Image:
    """Imagen PIL sintetica: suficiente para atravesar todo el pipeline."""
    return Image.new("RGB", (200, 150), "red")


@pytest.fixture(autouse=True)
def _cache_limpia():
    """Cada test arranca con la cache de modelos de `src.predict` vacia."""
    predict_mod.limpiar_cache()
    yield
    predict_mod.limpiar_cache()


@pytest.fixture
def ckpt_base(tmp_path: Path) -> Path:
    """Checkpoint del catalogo base (4 clases "de Oxford")."""
    return _crear_checkpoint(tmp_path, NOMBRES_BASE)


@pytest.fixture
def ckpt_propio(tmp_path: Path) -> Path:
    """Checkpoint de un catalogo propio (3 clases distintas de las del base)."""
    return _crear_checkpoint(
        tmp_path, NOMBRES_PROPIO, subcarpeta="resultados/custom"
    )


# --------------------------------------------------------------------------- #
# 1) build_app sin checkpoints
# --------------------------------------------------------------------------- #
def test_build_app_sin_checkpoints_no_lanza(tmp_path: Path) -> None:
    """La app se construye aunque no exista ningun checkpoint (criterio SA-07)."""
    app = build_app(str(tmp_path / "no_existe.pt"))

    assert app is not None
    assert hasattr(app, "launch")
    assert callable(app.launch)


# --------------------------------------------------------------------------- #
# 2) predecir sin imagen
# --------------------------------------------------------------------------- #
def test_predecir_sin_imagen_pide_una_imagen(tmp_path: Path) -> None:
    etiquetas, tabla = predecir(None, str(tmp_path / "no_existe.pt"))

    assert etiquetas == {}
    assert tabla == MSG_SIN_IMAGEN
    assert "Sube una imagen" in tabla


# --------------------------------------------------------------------------- #
# 3) predecir con checkpoint inexistente
# --------------------------------------------------------------------------- #
def test_predecir_checkpoint_inexistente_pide_entrenar(tmp_path: Path) -> None:
    ruta = tmp_path / "no_existe.pt"

    etiquetas, tabla = predecir(_imagen_roja(), str(ruta))

    assert etiquetas == {}
    assert "Entrena primero" in tabla
    assert "python -m src.train" in tabla
    assert str(ruta) in tabla


# --------------------------------------------------------------------------- #
# 4) predecir con el modelo base
# --------------------------------------------------------------------------- #
def test_predecir_devuelve_topk_del_modelo_base(ckpt_base: Path) -> None:
    topk = N_CLASES_BASE

    etiquetas, tabla = predecir(
        _imagen_roja(), str(ckpt_base), topk=topk, min_conf=0.0
    )

    assert isinstance(etiquetas, dict)
    assert len(etiquetas) == topk, "min_conf=0.0 debe devolver exactamente topk etiquetas"
    assert set(etiquetas) == set(NOMBRES_BASE)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in etiquetas.values())
    for nombre in NOMBRES_BASE:
        assert nombre in tabla, f"falta {nombre!r} en la tabla"
    assert "| Especie | Probabilidad |" in tabla
    # Sin catalogo propio no se menciona ningun otro catalogo.
    assert "catalogo propio" not in tabla.lower()


# --------------------------------------------------------------------------- #
# 5) enrutado al catalogo propio
# --------------------------------------------------------------------------- #
def _predict_falso_por_ruta(conf_base: float = 0.05, conf_propio: float = 0.90):
    """`predict_image` falso con confianzas controladas (base flojo, propio seguro).

    Las etiquetas imitan la forma que devuelve el checkpoint correspondiente, de
    modo que el test comprueba de verdad de que catalogo viene la respuesta.
    """

    def _falso(image_path, checkpoint_path, topk=5, min_conf=0.25, device="auto"):
        ruta = str(checkpoint_path).replace("\\", "/").lower()
        if "custom" in ruta:
            return [(nombre, conf_propio) for nombre in NOMBRES_PROPIO[:topk]]
        return [(nombre, conf_base) for nombre in NOMBRES_BASE[:topk]]

    return _falso


def _con_confianza_base_falsa(confianza_base: float, umbral: float):
    """Envuelve el `predict_image` REAL forzando la confianza del modelo base.

    Se sigue ejecutando la inferencia real (carga del checkpoint, transforms,
    softmax); solo se sustituye la probabilidad del primer resultado por
    `confianza_base`, que es lo que mira `predecir` (via `max`) para enrutar. Con
    pesos aleatorios esa probabilidad maxima es ~1/num_classes y haria el enrutado
    dependiente del azar. El catalogo propio ("custom" en la ruta) NO se toca.
    """
    real = predict_mod.predict_image

    def _envuelto(image_path, checkpoint_path, topk=5, min_conf=0.25, device="auto"):
        resultados = real(image_path, checkpoint_path, topk=topk, min_conf=min_conf, device=device)
        if "custom" in str(checkpoint_path).replace("\\", "/").lower():
            return resultados
        etiquetas = [str(nombre) for nombre, _ in resultados]
        if not etiquetas:
            return resultados
        # El primer resultado se queda con `confianza_base`; el resto por debajo
        # del umbral (asi el maximo es exactamente el valor pedido).
        resto = max(0.0, min(umbral, confianza_base) / 2.0)
        return [(etiquetas[0], confianza_base)] + [
            (nombre, resto) for nombre in etiquetas[1:]
        ]

    return _envuelto


def test_enrutado_al_catalogo_propio_con_min_conf_imposible(
    ckpt_base: Path, ckpt_propio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con `min_conf=1.1` y checkpoint propio, la respuesta sale del catalogo propio.

    El propio se predice DE VERDAD (checkpoint sintetico real) y con `min_conf=1.1`
    ninguno de los dos puede superar el umbral. Para que el enrutado no dependa del
    azar de los pesos aleatorios se fija la confianza del base en 0.06; como la del
    propio es mayor, el `gr.Label` (que nunca mezcla catalogos) se queda con SUS
    etiquetas y la tabla muestra las dos listas.
    """
    monkeypatch.setattr(
        predict_mod, "predict_image", _con_confianza_base_falsa(0.06, umbral=1.1)
    )

    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=N_CLASES_PROPIO,
        min_conf=1.1,  # imposible de superar: fuerza la ruta del catalogo propio
        custom_checkpoint=str(ckpt_propio),
    )

    # Las etiquetas del propio existen porque se calculo de verdad su top-k.
    for nombre in NOMBRES_PROPIO:
        assert nombre in tabla, (
            f"{nombre!r} falta: la lista del catalogo propio debe mostrarse"
        )
    assert set(etiquetas) == set(NOMBRES_PROPIO)
    assert set(etiquetas).isdisjoint(NOMBRES_BASE), "no se mezclan los dos catalogos"

    assert "catalogo propio" in tabla.lower()
    assert "ninguna de las dos listas es fiable" in tabla.lower()
    assert str(ckpt_propio.name) == "best.pt" and "best.pt" in tabla


def test_catalogo_propio_responde_cuando_el_base_no_llega(
    ckpt_base: Path, ckpt_propio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el base no llega al umbral y el propio si, se muestran SUS etiquetas."""
    # `predecir` importa `predict_image` de forma perezosa desde `src.predict`,
    # asi que se parchea en su modulo de origen (y no en `src.app`).
    monkeypatch.setattr(predict_mod, "predict_image", _predict_falso_por_ruta(0.05, 0.90))

    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=3,
        min_conf=0.30,
        custom_checkpoint=str(ckpt_propio),
    )

    assert set(etiquetas) == set(NOMBRES_PROPIO)
    assert TEXTO_PROPIO.split("(")[0].strip() in tabla
    assert "Reconocida por el catalogo propio" in tabla
    # El aviso de confianza baja NO aplica: el modelo propio si esta seguro.
    assert "AVISO: confianza baja" not in tabla
    # No aparece la lista del base (su 5 % no alcanza el 30 %).
    assert "rosa" not in tabla


def test_base_responde_y_lo_indica_cuando_si_llega_al_umbral(
    ckpt_base: Path, ckpt_propio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el base alcanza el umbral, responde el base y se indica en la tabla."""
    # El wrapper conserva las etiquetas REALES del base y solo fija su confianza
    # maxima; el catalogo propio ni siquiera se llega a consultar.
    monkeypatch.setattr(
        predict_mod, "predict_image", _con_confianza_base_falsa(0.80, umbral=0.30)
    )

    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=4,
        min_conf=0.30,
        custom_checkpoint=str(ckpt_propio),
    )

    assert set(etiquetas) == set(NOMBRES_BASE)
    # La confianza maxima es exactamente la que fijo el wrapper (el primer lugar
    # del modelo varia entre ejecuciones porque los pesos son aleatorios).
    assert max(etiquetas.values()) == pytest.approx(0.80)
    assert "Flowers-102" in tabla
    assert TEXTO_BASE.split("(")[0].strip() in tabla
    assert "Reconocida por el catalogo propio" not in tabla
    assert not any(nombre in tabla for nombre in NOMBRES_PROPIO)


def test_ningun_catalogo_llega_muestra_las_dos_listas(
    ckpt_base: Path, ckpt_propio: Path
) -> None:
    """Con `min_conf=1.1` en ambos, se muestran las dos listas y el aviso."""
    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=3,
        min_conf=1.1,
        custom_checkpoint=str(ckpt_propio),
    )

    # El `gr.Label` recibe las etiquetas del que mas confianza tuvo sin superar el
    # umbral. Con dos checkpoints de pesos aleatorios el ganador varia, asi que no
    # se fija cual: lo exigible es que sea UNO de los dos catalogos, sin mezclarlos.
    etiquetas_esperadas = (set(NOMBRES_BASE), set(NOMBRES_PROPIO))
    assert set(etiquetas) in etiquetas_esperadas, "las etiquetas deben ser de un solo catalogo"

    assert "ninguna de las dos listas es fiable" in tabla.lower()
    assert "Catalogo Flowers-102" in tabla
    assert "Catalogo propio" in tabla
    assert any(nombre in tabla for nombre in NOMBRES_BASE)
    assert any(nombre in tabla for nombre in NOMBRES_PROPIO)
    assert "AVISO: confianza baja" in tabla


# --------------------------------------------------------------------------- #
# 6) custom_checkpoint inexistente -> comportamiento del base
# --------------------------------------------------------------------------- #
def test_custom_checkpoint_inexistente_usa_el_base_sin_lanzar(
    ckpt_base: Path, tmp_path: Path
) -> None:
    inexistente = tmp_path / "custom" / "no_existe.pt"
    assert not inexistente.exists()

    esperadas, _ = predecir(_imagen_roja(), str(ckpt_base), topk=4, min_conf=0.0)
    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=4,
        min_conf=0.0,
        custom_checkpoint=str(inexistente),
    )

    assert etiquetas == esperadas, "debe comportarse igual que sin catalogo propio"
    assert set(etiquetas) == set(NOMBRES_BASE)
    assert AVISO_PROPIO_INEXISTENTE.split("**")[1] in tabla  # titulo del aviso
    assert "no se encontro el checkpoint del catalogo propio" in tabla


def test_custom_checkpoint_vacio_se_ignora(ckpt_base: Path) -> None:
    """Una ruta vacia o solo espacios equivale a no pasar catalogo propio."""
    esperadas, _ = predecir(_imagen_roja(), str(ckpt_base), topk=4, min_conf=0.0)

    for ruta in ("", "   "):
        etiquetas, tabla = predecir(
            _imagen_roja(), str(ckpt_base), topk=4, min_conf=0.0, custom_checkpoint=ruta
        )
        assert etiquetas == esperadas
        assert "catalogo propio" not in tabla.lower()


# --------------------------------------------------------------------------- #
# 7) custom_checkpoint que no se puede cargar
# --------------------------------------------------------------------------- #
def test_custom_checkpoint_ilegible_no_lanza_y_avisa(
    ckpt_base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roto = tmp_path / "custom" / "best.pt"
    roto.parent.mkdir(parents=True, exist_ok=True)
    roto.write_text("esto no es un checkpoint de torch", encoding="utf-8")
    assert roto.is_file()

    esperadas, _ = predecir(_imagen_roja(), str(ckpt_base), topk=4, min_conf=0.0)
    # El base se queda por debajo del umbral para que la demo SI intente cargar
    # el catalogo propio, que es el que esta roto (fallo real de `torch.load`).
    monkeypatch.setattr(
        predict_mod, "predict_image", _con_confianza_base_falsa(0.10, umbral=0.6)
    )

    etiquetas, tabla = predecir(
        _imagen_roja(),
        str(ckpt_base),
        topk=4,
        min_conf=0.6,
        custom_checkpoint=str(roto),
    )

    # El wrapper cambia las probabilidades, no las etiquetas: se comparan las
    # etiquetas del base con y sin catalogo propio.
    assert set(etiquetas) == set(esperadas) == set(NOMBRES_BASE)
    assert "el checkpoint del catalogo propio no se pudo cargar" in tabla
    assert AVISO_PROPIO_FALLA.split("**")[1] in tabla
    # El resultado del base sigue estando: la demo no se cae ni se queda vacia.
    assert "| Especie | Probabilidad |" in tabla
    assert "best.pt" in tabla


# --------------------------------------------------------------------------- #
# 8) build_app con catalogo propio
# --------------------------------------------------------------------------- #
def test_build_app_con_catalogo_propio(ckpt_base: Path, ckpt_propio: Path) -> None:
    app = build_app(str(ckpt_base), 5, 0.25, str(ckpt_propio))

    assert app is not None
    assert hasattr(app, "launch")
    assert callable(app.launch)


def test_build_app_con_catalogo_propio_inexistente_no_lanza(tmp_path: Path) -> None:
    app = build_app(
        str(tmp_path / "no_existe.pt"),
        custom_checkpoint=str(tmp_path / "custom" / "no_existe.pt"),
    )

    assert hasattr(app, "launch")


# --------------------------------------------------------------------------- #
# 9) CLI: --custom-checkpoint
# --------------------------------------------------------------------------- #
def test_cli_acepta_custom_checkpoint() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "resultados/best.pt",
            "--custom-checkpoint",
            "resultados/custom/best.pt",
            "--topk",
            "3",
            "--min-conf",
            "0.1",
        ]
    )

    assert args.custom_checkpoint == "resultados/custom/best.pt"
    assert args.topk == 3
    assert args.min_conf == pytest.approx(0.1)
    # Opcional: sin el flag no se activa ningun catalogo propio.
    assert parse_args([]).custom_checkpoint is None


def test_main_propaga_el_custom_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`main` valida la ruta y se la pasa a `build_app` (y `launch` se simula)."""
    ckpt = _crear_checkpoint(tmp_path, NOMBRES_PROPIO, subcarpeta="resultados/custom")
    capturado: dict = {}

    class _AppFalsa:
        def launch(self, **kwargs):
            capturado["kwargs"] = kwargs
            return None

    def _build_app_falso(checkpoint_path="resultados/best.pt", topk=5, min_conf=0.25, custom_checkpoint=None):
        capturado["args"] = (str(checkpoint_path), int(topk), float(min_conf), custom_checkpoint)
        return _AppFalsa()

    monkeypatch.setattr(app_mod, "build_app", _build_app_falso)

    main(
        share=False,
        checkpoint_path=str(ckpt),
        topk=2,
        min_conf=0.5,
        custom_checkpoint=str(ckpt),
    )

    assert capturado["args"] == (str(ckpt), 2, 0.5, str(ckpt))
    assert capturado["kwargs"]["share"] is False

    # Ruta inexistente: solo avisa por consola, no lanza.
    inexistente = tmp_path / "otro" / "best.pt"
    main(checkpoint_path=str(ckpt), custom_checkpoint=str(inexistente))
    assert capturado["args"][3] == str(inexistente)
