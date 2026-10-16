"""Demo interactiva (Gradio) para clasificar especies de flores.

CONTRATO (Fase 0, propietario SA-07; ampliado por SA-E3 con catalogo propio):
    predecir(imagen, checkpoint_path="resultados/best.pt", topk=5,
             min_conf=0.25, custom_checkpoint=None) -> tuple[dict[str, float], str]
    build_app(checkpoint_path="resultados/best.pt", topk=5, min_conf=0.25,
              custom_checkpoint=None) -> gr.Blocks
    main(share=False, checkpoint_path="resultados/best.pt", topk=5, min_conf=0.25,
         custom_checkpoint=None) -> None
    CLI: python -m src.app [--checkpoint resultados/best.pt] [--topk 5]
                           [--min-conf 0.25] [--custom-checkpoint RUTA] [--share]

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

Dos catalogos (integracion de SA-E3):
  - Sin `custom_checkpoint` (o si la ruta no existe) todo funciona exactamente
    como antes: solo el modelo base de las 102 especies de Oxford.
  - Con `custom_checkpoint` valido se predice primero con el modelo base; si su
    confianza maxima llega a `min_conf` responde el base y se indica en la tabla
    que respondio el **catalogo Flowers-102**. Si no llega, se prueba el modelo
    propio: si este si llega, se muestran sus etiquetas y su tabla; si tampoco
    llega, se muestran las dos listas precedidas del AVISO de confianza baja.
  - Si el checkpoint propio existe pero no se puede cargar (por ejemplo entrenado
    con otro backbone), se avisa en la tabla y se sigue mostrando el resultado del
    base: la demo nunca se cae.
  - `app.py` NO importa `src.custom`: la demo solo necesita la RUTA del
    checkpoint, que `src.predict.predict_image` ya sabe usar tal cual.
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
    "Ademas del catalogo Flowers-102 puedes anadir tu **catalogo propio**: si has "
    "entrenado tus propias especies (por ejemplo con `python -m src.custom`), pasa "
    "la ruta de ese checkpoint con `--custom-checkpoint RUTA/al/best.pt` y la demo "
    "lo usara como respaldo cuando el catalogo de 102 especies no este seguro.\n\n"
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

# --- textos de la integracion de dos catalogos ----------------------------- #
#: Fila que se anyade a la tabla cuando responde el catalogo base.
TEXTO_BASE = (
    "*Resultado del catalogo **Flowers-102** (el modelo base del proyecto): "
    "respondio este porque su confianza maxima alcanzo el umbral "
    "`min_conf = {min_conf:.2f}`.*"
)

#: Fila que se anyade a la tabla cuando responde el catalogo propio.
TEXTO_PROPIO = (
    "**Reconocida por el catalogo propio** "
    "(`{ruta}`, confianza maxima {confianza})."
)

#: Encabezado de la lista del catalogo base dentro del aviso de doble resultado.
TEXTO_LISTA_BASE = "**Catalogo Flowers-102** (modelo base del proyecto)"
#: Encabezado de la lista del catalogo propio dentro del aviso de doble resultado.
TEXTO_LISTA_PROPIO = "**Catalogo propio** (`{ruta}`)"

#: Aviso de checkpoint propio inexistente (se ignora, nunca lanza).
AVISO_PROPIO_INEXISTENTE = (
    "**Aviso: no se encontro el checkpoint del catalogo propio.**\n\n"
    "No existe `{ruta}`: se ignora y la demo responde solo con el catalogo "
    "Flowers-102. Corrige la ruta o entrena tu catalogo con "
    "`python -m src.custom`."
)

#: Aviso de checkpoint propio que existe pero no se puede cargar.
AVISO_PROPIO_FALLA = (
    "**Aviso: el checkpoint del catalogo propio no se pudo cargar.**\n\n"
    "Ruta: `{ruta}`\n\n"
    "Detalle: `{error}`\n\n"
    "Puede estar corrupto o entrenado con otro backbone. Se muestra el resultado "
    "del catalogo Flowers-102."
)

#: Aviso (barra lateral) cuando la demo arranca con catalogo propio configurado.
AVISO_PROPIO_UI = (
    "### Catalogo propio activo\n\n"
    "Se usara `{ruta}` como respaldo cuando el catalogo Flowers-102 no llegue al "
    "umbral de confianza. Si esa ruta no existe o el checkpoint no se puede "
    "cargar, la demo lo avisa en la tabla y responde solo con Flowers-102."
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


def checkpoint_existe(ruta: Any) -> bool:
    """True si `ruta` apunta a un archivo existente (tolerante a None y a basura).

    Se usa para decidir si hay modelo base y si hay catalogo propio. Nunca lanza:
    una ruta vacia, un tipo raro o un `OSError` del sistema de archivos se
    traducen en `False`.
    """
    if ruta is None:
        return False
    texto = str(ruta).strip()
    if not texto:
        return False
    try:
        return Path(texto).exists()
    except OSError:  # rutas imposibles en Windows (caracteres invalidos, etc.)
        return False


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


def _codigo(texto: Any) -> str:
    """Sanea un valor para mostrarlo dentro de un `code span` de Markdown."""
    plano = " ".join(str(texto).split())
    return plano.replace("`", "'") or "s/d"


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


def _separar_aviso(resultados: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], float]:
    """Separa la etiqueta centinela "no reconocida" de las etiquetas reales.

    `predict_image` responde `[("no reconocida / fuera del catalogo", prob_max)]`
    cuando `max(prob) < min_conf`. Cuando la demo pide `min_conf=0.0` ese caso no
    aparece; solo surge si el usuario configura un `min_conf > 0`. Se devuelve
    `(etiquetas_reales, confianza_maxima)` leyendo la constante `NO_RECONOCIDA` de
    `src.predict` (nunca una copia local de la cadena).
    """
    if not resultados:
        return [], 0.0

    centinela: str | None = None
    try:
        from src.predict import NO_RECONOCIDA  # import perezoso a proposito

        centinela = str(NO_RECONOCIDA)
    except Exception:  # pragma: no cover - sin src.predict no hay prediccion
        centinela = None

    confianza = max(prob for _, prob in resultados)
    if centinela is not None and any(nombre == centinela for nombre, _ in resultados):
        reales = [(nombre, prob) for nombre, prob in resultados if nombre != centinela]
        return reales, confianza
    return list(resultados), confianza


def _confianza(resultados: list[tuple[str, float]]) -> float:
    """Confianza maxima de una lista de resultados (0.0 si esta vacia)."""
    return max((prob for _, prob in resultados), default=0.0)


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


def _avisar_si_falta(checkpoint_path: str) -> None:
    """Aviso por consola (una sola vez, al arrancar) si falta un checkpoint."""
    print(
        f"[aviso] No se encontro el checkpoint: {checkpoint_path}\n"
        "        La demo se abrira igual, pero mostrara el aviso "
        "'entrena primero: python -m src.train'.\n"
        "        Entrena con:  python -m src.train",
        flush=True,
    )


def _avisar_custom_inexistente(custom_checkpoint: str) -> None:
    """Aviso por consola si se pidio un catalogo propio que no existe.

    Nunca lanza: el catalogo propio es opcional y una ruta mal escrita solo debe
    degradar al comportamiento del modelo base.
    """
    print(
        f"[aviso] No se encontro el checkpoint del catalogo propio: "
        f"{custom_checkpoint}\n"
        "        Se ignora: la demo respondera solo con el catalogo Flowers-102.\n"
        "        Entrena tu catalogo con:  python -m src.custom",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Funcion interna de la demo (visible para verificacion directa)
# --------------------------------------------------------------------------- #
def predecir(
    imagen: Any,
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
    custom_checkpoint: str | None = None,
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

    Sobre `custom_checkpoint` (catalogo propio, opcional):
      - `None` o ruta inexistente -> comportamiento historico (solo base).
      - Si existe, se predice primero con el base; solo si el base NO alcanza
        `min_conf` se consulta el modelo propio.
      - Si el propio responde, sus etiquetas y su tabla son las que se muestran
        (con la linea "Reconocida por el catalogo propio").
      - Si tampoco alcanza el umbral, se muestran las DOS listas con el AVISO de
        confianza baja. El `gr.Label` recibe siempre las etiquetas del modelo con
        mayor confianza (las del base si ninguno llega) para no mezclar barras.
      - Si el propio existe pero falla al cargar, se avisa en la tabla y se
        muestra el resultado del base.
    """
    if imagen is None:
        return {}, MSG_SIN_IMAGEN

    if not checkpoint_existe(checkpoint_path):
        return {}, mensaje_entrena_primero(str(checkpoint_path))

    predict_image, error_import = _cargar_predict_image()
    if predict_image is None:
        return {}, (
            "**No se pudo cargar el modulo de inferencia** (`src.predict`).\n\n"
            f"Detalle: `{type(error_import).__name__}: {error_import}`"
        )

    # --- 1) modelo base (Flowers-102) -------------------------------------- #
    try:
        crudos_base = predict_image(imagen, checkpoint_path, topk=topk, min_conf=0.0)
    except FileNotFoundError as exc:
        return {}, mensaje_entrena_primero(str(checkpoint_path)) + f"\n\nDetalle: `{_celda(exc)}`"
    except Exception as exc:
        return {}, (
            "**Error al predecir la imagen.**\n\n"
            f"`{type(exc).__name__}: {_celda(exc)}`\n\n"
            "Revisa que la imagen sea valida (jpg/png) y que el checkpoint "
            "corresponda al mismo backbone."
        )

    reales_base, confianza_base = _separar_aviso(_normalizar(crudos_base, topk))

    # --- 2) catalogo propio (opcional) ------------------------------------- #
    ruta_propia: str | None = None
    reales_propio: list[tuple[str, float]] = []
    confianza_propio: float | None = None
    error_propio: str | None = None
    aviso_extra: str | None = None

    if custom_checkpoint is not None and str(custom_checkpoint).strip():
        ruta_propia = str(custom_checkpoint).strip()
        if not checkpoint_existe(ruta_propia):
            # Ruta mal escrita: se ignora con un aviso, jamas una excepcion.
            aviso_extra = AVISO_PROPIO_INEXISTENTE.format(ruta=_codigo(ruta_propia))
            _avisar_custom_inexistente(ruta_propia)
            ruta_propia = None
        elif confianza_base < float(min_conf):
            # Solo se consulta el modelo propio si el base no esta seguro.
            try:
                crudos_propio = predict_image(imagen, ruta_propia, topk=topk, min_conf=0.0)
                reales_propio, confianza_propio = _separar_aviso(
                    _normalizar(crudos_propio, topk)
                )
            except Exception as exc:  # checkpoint roto, otro backbone, etc.
                error_propio = f"{type(exc).__name__}: {_celda(exc)}"
                confianza_propio = None
                aviso_extra = AVISO_PROPIO_FALLA.format(
                    ruta=_codigo(ruta_propia), error=error_propio
                )

    if not reales_base and not reales_propio:
        return {}, aviso_extra or MSG_SIN_RESULTADOS

    # --- 3) quien responde -------------------------------------------------- #
    # El base manda siempre que llegue al umbral (el catalogo propio solo es un
    # respaldo cuando el base no esta seguro). Si el propio llega y el base no,
    # la respuesta es del propio. Si ninguno llega, el `gr.Label` recibe las
    # etiquetas del que tuvo mas confianza (las del base si empatan) para que las
    # barras no mezclen dos catalogos distintos.
    propio_responde = (
        confianza_propio is not None and confianza_propio >= float(min_conf)
    )
    base_responde = confianza_base >= float(min_conf)

    if ruta_propia is not None and not base_responde and confianza_propio is not None:
        if propio_responde or confianza_propio > confianza_base:
            elegidas = reales_propio
        else:
            elegidas = reales_base
    else:
        elegidas = reales_base or reales_propio

    # El encabezado del catalogo propio solo se pone si el propio es quien
    # responde de verdad (si no, lo que se muestra es el base o la doble lista).
    encabezado_propio: str | None = None
    if ruta_propia is not None and not base_responde and propio_responde:
        encabezado_propio = (
            TEXTO_PROPIO.format(
                ruta=_codigo(ruta_propia), confianza=_fmt_pct(confianza_propio)
            )
            + "\n\n"
        )

    etiquetas = {nombre: prob for nombre, prob in elegidas}
    tabla = tabla_markdown(elegidas)

    if ruta_propia is not None and not base_responde and not propio_responde:
        # Ni el catalogo base ni el propio llegan al umbral: se muestran las DOS
        # listas completas, dejando claro que ninguna de las dos es fiable. Este
        # aviso sustituye al generico de "confianza baja" (lo incluye y lo amplia).
        confianza_mostrada = _confianza(elegidas)
        catalogo_propio = (
            f"{TEXTO_LISTA_PROPIO.format(ruta=_codigo(ruta_propia))}\n\n"
            f"{tabla_markdown(reales_propio) if reales_propio else MSG_SIN_RESULTADOS}"
        )
        catalogo_base = (
            f"{TEXTO_LISTA_BASE}\n\n"
            f"{tabla_markdown(reales_base) if reales_base else MSG_SIN_RESULTADOS}"
        )
        if confianza_propio is not None and confianza_propio > confianza_base:
            cuerpo = f"{catalogo_propio}\n\n{catalogo_base}"
        else:
            cuerpo = f"{catalogo_base}\n\n{catalogo_propio}"
        tabla = (
            f"**AVISO: confianza baja ({confianza_mostrada * 100:.1f} % < "
            f"{float(min_conf) * 100:.0f} %).** "
            "Ninguna de las dos listas es fiable: ningun catalogo alcanzo el umbral "
            f"`min_conf = {float(min_conf):.2f}` (el catalogo Flowers-102 se quedo "
            f"en {_fmt_pct(confianza_base)} y el catalogo propio en "
            f"{_fmt_pct(confianza_propio)}). La mejor suposicion de cada uno es:\n\n"
            f"{cuerpo}\n\n"
            "Usa la lista con mas confianza como orientacion, nunca como certeza. "
            "Si la flor es de tus propias especies, entrena mas tu catalogo propio; "
            "si es una de las 102 de Oxford, revisa que el modelo base este bien "
            "entrenado.\n\n"
            + tabla
        )
    else:
        confianza_mostrada = _confianza(elegidas)
        if confianza_mostrada < float(min_conf):
            tabla = (
                f"**AVISO: confianza baja ({confianza_mostrada * 100:.1f} % < "
                f"{float(min_conf) * 100:.0f} %).** "
                "La lista de arriba es la mejor suposicion, no una certeza. Puede ocurrir "
                "por dos motivos: (a) la flor no esta entre las 102 especies del catalogo, "
                "o (b) el modelo esta poco entrenado (por ejemplo si entrenaste en "
                "MODO_DEBUG, o si aviso de que no pudo descargar los pesos preentrenados).\n\n"
            ) + tabla

        if ruta_propia is not None and base_responde and not propio_responde:
            tabla = tabla + "\n\n" + TEXTO_BASE.format(min_conf=float(min_conf))

    # --- 4) avisos de procedencia ------------------------------------------ #
    if encabezado_propio is not None:
        tabla = encabezado_propio + tabla

    if aviso_extra:
        tabla = aviso_extra + "\n\n" + tabla

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
    custom_checkpoint: str | None = None,
) -> gr.Blocks:
    """Construye la demo de Gradio (no lanza el servidor).

    Devuelve un `gr.Blocks` con `.launch(...)` (ver la desviacion documentada en
    el docstring del modulo: en Gradio 6 `gr.Interface` no permite mostrar el
    aviso de checkpoint ausente dentro del layout).

    Si `checkpoint_path` no existe, la app se construye igual, muestra un aviso
    visible en el propio layout y, al predecir, devuelve ese aviso en lugar de
    lanzar una excepcion.

    `custom_checkpoint` es opcional: si se indica, la demo anade un catalogo
    propio como respaldo (la ruta se le pasa tal cual a `predecir`, que decide en
    cada clic si el checkpoint se puede usar). La demo nunca comprueba ni carga
    ese checkpoint al construirse, asi que una ruta invalida no impide abrirla.
    """
    checkpoints = str(checkpoint_path)
    falta_checkpoint = not checkpoint_existe(checkpoints)

    ruta_propia: str | None = None
    if custom_checkpoint is not None and str(custom_checkpoint).strip():
        ruta_propia = str(custom_checkpoint).strip()

    def _fn(imagen: Any):
        return predecir(imagen, checkpoints, topk, min_conf, ruta_propia)

    ejemplo = _imagen_de_ejemplo()

    with gr.Blocks(title=TITULO, fill_width=True) as demo:
        gr.Markdown(f"# {TITULO}")
        gr.Markdown(DESCRIPCION)
        if falta_checkpoint:
            gr.Markdown(AVISO_CHECKPOINT_UI.format(ruta=checkpoints))
        if ruta_propia is not None:
            gr.Markdown(AVISO_PROPIO_UI.format(ruta=_codigo(ruta_propia)))

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
        "--custom-checkpoint",
        dest="custom_checkpoint",
        default=None,
        help=(
            "Ruta del checkpoint de un catalogo propio (por ejemplo "
            "resultados/custom/best.pt, generado con python -m src.custom). Es "
            "opcional: si no se indica, la demo usa solo el catalogo Flowers-102. "
            "Si la ruta no existe o el checkpoint no se puede cargar, la demo lo "
            "avisa y sigue funcionando con el modelo base."
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
    custom_checkpoint: str | None = None,
) -> None:
    """Construye y lanza la demo.

    Imprime la URL local y (si `share=True`) la publica. En Jupyter/Colab no
    bloquea el hilo; en un script normal si (hasta que se interrumpa con Ctrl+C).
    """
    if not checkpoint_existe(checkpoint_path):
        _avisar_si_falta(str(checkpoint_path))

    ruta_propia: str | None = None
    if custom_checkpoint is not None and str(custom_checkpoint).strip():
        ruta_propia = str(custom_checkpoint).strip()
        if not checkpoint_existe(ruta_propia):
            _avisar_custom_inexistente(ruta_propia)

    app = build_app(checkpoint_path, topk, min_conf, ruta_propia)

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
        custom_checkpoint=args.custom_checkpoint,
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
