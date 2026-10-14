"""Demo interactiva (Gradio) para clasificar especies de flores.

CONTRATO (Fase 0, propietario SA-07):
    build_app(checkpoint_path="resultados/best.pt", topk=5) -> gr.Interface
    main(share=False, checkpoint_path="resultados/best.pt") -> None
    CLI: python -m src.app [--checkpoint resultados/best.pt] [--topk 5] [--share]

Notas de implementacion:
  - La demo NUNCA debe crashear si falta el checkpoint: en ese caso la funcion
    interna devuelve un aviso de texto ("entrena primero: python -m src.train").
  - `src.predict` se importa de forma perezosa (dentro de la funcion) para que la
    app se pueda construir aunque el modulo de inferencia o `torch` fallen al
    importar. Esto mantiene el criterio de aceptacion de `build_app`.
  - Se escribio contra la API realmente instalada: gradio 6.30.0.
  - DESVIACION (justificada): `build_app` devuelve un `gr.Blocks`, no un
    `gr.Interface`. En Gradio 6 el evento `submit` desaparecio de los componentes
    (solo quedan `click`/`change`/`upload`), y con `gr.Interface` no se puede
    mostrar el aviso de "falta el checkpoint" de forma permanente en el layout.
    El objeto devuelto cumple el contrato funcional: tiene `.launch(...)`.
    Para volver a `gr.Interface` bastaria reconstruirlo con `fn`, `inputs`,
    `outputs`, `title`, `description` y `article` (ya estan todos definidos).
"""

from __future__ import annotations

import argparse
import html
import inspect
import sys
from pathlib import Path
from typing import Any

import gradio as gr

# --------------------------------------------------------------------------- #
# Textos de la interfaz (sin acentos, como el resto del codigo del proyecto)
# --------------------------------------------------------------------------- #
TITULO = "Clasificador de flores (Flowers102)"

DESCRIPCION = (
    "Sube una foto de una flor o tomala con la webcam y el modelo mostrara las "
    "especies mas probables con su nivel de confianza. "
    "Modelo: EfficientNet-B0 entrenado sobre Oxford Flowers102 (102 especies)."
)

ARTICULO = (
    "### Como usarlo\n"
    "1. Sube una imagen (arrastrar y soltar) **o** activa la pestana de camara.\n"
    "2. Pulsa **Enviar**.\n"
    "3. Revisa las etiquetas con su probabilidad y la tabla de las top-k.\n\n"
    "Si aparece *no reconocida / fuera del catalogo*, la confianza maxima quedo por "
    "debajo del umbral (`min_conf`) y la imagen probablemente no sea una de las 102 "
    "especies del dataset.\n\n"
    "En Google Colab puedes lanzar la demo con: "
    "`python -m src.app --share` (genera un enlace publico temporal)."
)

AVISO_ENTRENA = (
    "**No hay modelo entrenado todavia.**\n\n"
    "Entrena primero:\n\n"
    "```bash\n"
    "python -m src.train\n"
    "```\n\n"
    "Si el checkpoint esta en otra ruta, indicala con "
    "`python -m src.app --checkpoint RUTA/al/best.pt`."
)

MSG_SIN_IMAGEN = (
    "**Sube una imagen primero.** Usa el panel de la izquierda "
    "(subida o webcam) y vuelve a pulsar Enviar."
)

MSG_SIN_RESULTADOS = "No se pudo obtener ninguna prediccion para esa imagen."

# Aviso que se muestra de forma permanente en la propia app cuando el
# checkpoint no existe: asi el usuario lo ve sin necesidad de enviar una imagen.
AVISO_CHECKPOINT_UI = (
    "### No hay modelo entrenado\n\n"
    "No se encontro el checkpoint en `{ruta}`. Entrena primero con:\n\n"
    "```bash\n"
    "python -m src.train\n"
    "```\n\n"
    "La demo se abre igual, pero al enviar una imagen mostrara este mismo aviso "
    "en lugar de la prediccion."
)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def mensaje_entrena_primero(checkpoint_path: str) -> str:
    """Mensaje en Markdown que se muestra cuando falta el checkpoint."""
    return (
        f"{AVISO_ENTRENA}\n\n"
        f"Ruta buscada: `{checkpoint_path}`"
    )


def _fmt_pct(valor: Any) -> str:
    """Formatea una probabilidad como porcentaje; tolera None y valores raros."""
    try:
        return f"{float(valor) * 100:.2f} %"
    except (TypeError, ValueError):
        return "s/d"


def _celda(texto: Any) -> str:
    """Sanea un valor para que no rompa una celda de tabla Markdown."""
    plano = " ".join(str(texto).split())
    return html.escape(plano, quote=False).replace("|", "\\|") or "s/d"


def _normalizar(resultados: Any, topk: int) -> list[tuple[str, float]]:
    """Normaliza la salida de `predict_image` a una lista de (etiqueta, prob).

    Acepta listas de tuplas, listas de listas o diccionarios; descarta entradas
    mal formadas y ordena de mayor a menor probabilidad.
    """
    if resultados is None:
        return []

    if isinstance(resultados, dict):
        candidatos: list[Any] = list(resultados.items())
    else:
        candidatos = list(resultados)

    normalizados: list[tuple[str, float]] = []
    for item in candidatos:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            etiqueta, prob = item[0], item[1]
        else:
            continue
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            continue
        normalizados.append((str(etiqueta), prob))

    normalizados.sort(key=lambda par: par[1], reverse=True)
    return normalizados[:topk] if topk > 0 else normalizados


def tabla_markdown(resultados: list[tuple[str, float]]) -> str:
    """Construye la tabla Markdown (especie | probabilidad %)."""
    if not resultados:
        return MSG_SIN_RESULTADOS

    filas = [
        "| Especie | Probabilidad |",
        "| :--- | ---: |",
    ]
    for etiqueta, prob in resultados:
        filas.append(f"| {_celda(etiqueta)} | {_fmt_pct(prob)} |")

    ganadora = _celda(resultados[0][0])
    encabezado = f"**Prediccion principal:** {ganadora} ({_fmt_pct(resultados[0][1])})\n\n"
    return encabezado + "\n".join(filas)


def _cargar_predict_image():
    """Importa `predict_image` de forma perezosa.

    Devuelve `(funcion, error)`: exactamente uno de los dos es None.
    """
    try:
        from src.predict import predict_image  # import perezoso a proposito
    except Exception as exc:  # ImportError, fallos de torch, etc.
        return None, exc
    if not callable(predict_image):
        return None, AttributeError("src.predict no define predict_image()")
    return predict_image, None


def _imagen_de_ejemplo() -> str | None:
    """Primera imagen disponible de `data/` o `assets/` para el panel de ejemplos."""
    for carpeta in ("data", "assets"):
        raiz = Path(carpeta)
        if not raiz.is_dir():
            continue
        for patron in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for ruta in sorted(raiz.rglob(patron)):
                if ruta.is_file():
                    return str(ruta.resolve())
    return None


def _es_notebook() -> bool:
    """True si estamos dentro de Jupyter/Colab (para no bloquear el hilo)."""
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


# --------------------------------------------------------------------------- #
# Funcion interna de la demo (visible para verificacion directa)
# --------------------------------------------------------------------------- #
def predecir(
    imagen: Any,
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> tuple[dict[str, float], str]:
    """Funcion que ejecuta la demo.

    Devuelve `(etiquetas_para_gr_Label, tabla_markdown)` y **nunca** lanza
    excepciones: cualquier fallo se traduce en un mensaje claro para el usuario.

    Sobre `min_conf`: se pide la lista top-k con `min_conf=0.0` y el umbral se
    aplica DESPUES, como AVISO, no como borrado de la prediccion. Motivo: con un
    modelo poco entrenado (por ejemplo en MODO_DEBUG, o entrenado sin pesos
    preentrenados) el softmax es muy plano y su maximo ronda el 2 %, de modo que
    aplicar el umbral antes hacia que la demo respondiera siempre "no reconocida"
    aunque la especie propuesta fuera la correcta. Asi el usuario ve la mejor
    suposicion y, ademas, sabe si hay que desconfiar de ella.
    """
    if imagen is None:
        return {}, MSG_SIN_IMAGEN

    if not Path(checkpoint_path).exists():
        return {}, mensaje_entrena_primero(checkpoint_path)

    predict_image, error_import = _cargar_predict_image()
    if predict_image is None:
        return {}, (
            "**No se pudo cargar el modulo de inferencia** (`src.predict`).\n\n"
            f"Detalle: `{type(error_import).__name__}: {error_import}`"
        )

    try:
        crudos = predict_image(imagen, checkpoint_path, topk=topk, min_conf=0.0)
    except FileNotFoundError as exc:
        return {}, mensaje_entrena_primero(checkpoint_path) + f"\n\nDetalle: `{_celda(exc)}`"
    except Exception as exc:
        return {}, (
            "**Error al predecir la imagen.**\n\n"
            f"`{type(exc).__name__}: {_celda(exc)}`\n\n"
            "Revisa que la imagen sea valida (jpg/png) y que el checkpoint "
            "corresponda al mismo backbone."
        )

    resultados = _normalizar(crudos, topk)
    if not resultados:
        return {}, MSG_SIN_RESULTADOS

    etiquetas = {nombre: prob for nombre, prob in resultados}
    tabla = tabla_markdown(resultados)

    confianza = max(prob for _, prob in resultados)
    if confianza < min_conf:
        tabla = (
            f"**AVISO: confianza baja ({confianza * 100:.1f} % < {min_conf * 100:.0f} %).** "
            "La lista de arriba es la mejor suposicion, no una certeza. Puede ocurrir "
            "por dos motivos: (a) la flor no esta entre las 102 especies del catalogo, "
            "o (b) el modelo esta poco entrenado (por ejemplo si entrenaste en "
            "MODO_DEBUG, o si aviso de que no pudo descargar los pesos preentrenados).\n\n"
        ) + tabla
    return etiquetas, tabla


# Alias estable para verificaciones/test manuales.
app_fn = predecir


# --------------------------------------------------------------------------- #
# Construccion de la interfaz
# --------------------------------------------------------------------------- #
def build_app(
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> gr.Blocks:
    """Construye la demo de Gradio (no lanza el servidor).

    Devuelve un `gr.Blocks` con `.launch(...)` (ver la desviacion documentada en
    el docstring del modulo: en Gradio 6 `gr.Interface` no permite mostrar el
    aviso de checkpoint ausente dentro del layout).

    Si `checkpoint_path` no existe, la app se construye igual, muestra un aviso
    visible en el propio layout y, al predecir, devuelve ese aviso en lugar de
    lanzar una excepcion.
    """
    checkpoints = str(checkpoint_path)
    falta_checkpoint = not Path(checkpoints).exists()

    def _fn(imagen: Any):
        return predecir(imagen, checkpoints, topk, min_conf)

    ejemplo = _imagen_de_ejemplo()

    with gr.Blocks(title=TITULO, fill_width=True) as demo:
        gr.Markdown(f"# {TITULO}")
        gr.Markdown(DESCRIPCION)
        if falta_checkpoint:
            gr.Markdown(AVISO_CHECKPOINT_UI.format(ruta=checkpoints))

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                entrada = gr.Image(
                    label="Imagen de la flor",
                    type="pil",  # PIL.Image: formato aceptado por predict_image
                    sources=["upload", "webcam", "clipboard"],
                    height=380,
                )
                with gr.Row():
                    boton_enviar = gr.Button("Enviar", variant="primary")
                    boton_limpiar = gr.Button("Limpiar")
                if ejemplo:
                    gr.Examples(examples=[ejemplo], inputs=entrada)

            with gr.Column(scale=1, min_width=320):
                salida_etiquetas = gr.Label(
                    label=f"Top-{topk} especies mas probables",
                    num_top_classes=topk,
                )
                salida_tabla = gr.Markdown(
                    value=(
                        AVISO_CHECKPOINT_UI.format(ruta=checkpoints)
                        if falta_checkpoint
                        else "Sube una imagen para ver el detalle de las top-k."
                    ),
                    label="Detalle del top-k",
                )

        gr.Markdown(ARTICULO)

        entradas = [entrada]
        salidas = [salida_etiquetas, salida_tabla]
        # Gradio 6 ya no expone el evento `submit` en los componentes: se usa el
        # boton (y ademas `Image.upload`, para predecir al subir la imagen).
        # `api_name=None` evita publicar los handlers auxiliares en la API HTTP.
        boton_enviar.click(fn=_fn, inputs=entradas, outputs=salidas, api_name="predict")
        entrada.upload(fn=_fn, inputs=entradas, outputs=salidas, api_name=None)
        boton_limpiar.click(
            fn=lambda: (None, {}, "Sube una imagen para ver el detalle de las top-k."),
            inputs=None,
            outputs=[entrada, salida_etiquetas, salida_tabla],
            api_name=None,
        )

    return demo


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos de linea de comandos segun el contrato (seccion 3.b)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.app",
        description=(
            "Abre la demo interactiva (Gradio) del clasificador de flores. "
            "No requiere el checkpoint para abrirse: si falta, la demo lo avisa."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="resultados/best.pt",
        help="Ruta del checkpoint del modelo (por defecto: resultados/best.pt).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Cuantas especies mostrar como maximo (por defecto: 5).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help=(
            "Umbral de confianza por debajo del cual la demo muestra un AVISO de "
            "confianza baja (por defecto: 0.25). Bajalo si tu modelo esta poco "
            "entrenado y quieres ver la prediccion sin el aviso."
        ),
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help=(
            "Genera un enlace publico temporal de Gradio (util en Google Colab; "
            "requiere salida a internet). Por defecto es False."
        ),
    )
    return parser.parse_args(argv)


def main(
    share: bool = False,
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> None:
    """Construye y lanza la demo.

    Imprime la URL local y (si `share=True`) la publica. En Jupyter/Colab no
    bloquea el hilo; en un script normal si (hasta que se interrumpa con Ctrl+C).
    """
    if not Path(checkpoint_path).exists():
        print(
            f"[aviso] No se encontro el checkpoint: {checkpoint_path}\n"
            "        La demo se abrira igual, pero mostrara el aviso "
            "'entrena primero: python -m src.train'.\n"
            "        Entrena con:  python -m src.train",
            flush=True,
        )

    app = build_app(checkpoint_path, topk, min_conf)

    # En notebooks/Colab no bloqueamos la celda; en script el servidor sigue vivo.
    kwargs: dict[str, Any] = {"share": share, "prevent_thread_lock": _es_notebook()}

    # `inbrowser` no aporta nada en Colab.
    if "inbrowser" in inspect.signature(app.launch).parameters:
        kwargs["inbrowser"] = not _es_notebook()

    resultado = app.launch(**kwargs)

    url_local: str | None = None
    url_publica: str | None = None
    if isinstance(resultado, tuple) and len(resultado) >= 3:
        url_local, url_publica = resultado[1], resultado[2]
    elif isinstance(resultado, str):
        url_local = resultado

    if url_local:
        print(f"Demo local:   {url_local}", flush=True)
    if share and url_publica:
        print(f"Demo publica: {url_publica}", flush=True)
    elif share:
        print(
            "[aviso] Se pidio --share pero Gradio no devolvio enlace publico "
            "(revisa la conexion a internet).",
            flush=True,
        )
    print("Pulsa Ctrl+C para detener la demo.", flush=True)


def cli(argv: list[str] | None = None) -> None:
    """Punto de entrada de `python -m src.app`."""
    args = parse_args(argv)
    main(
        share=bool(args.share),
        checkpoint_path=str(args.checkpoint),
        topk=int(args.topk),
        min_conf=float(args.min_conf),
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
