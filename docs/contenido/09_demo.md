# La demo: la ventana donde cualquiera puede probarlo

## Para qué sirve este archivo (en lenguaje llano)

Este archivo explica `src/app.py`, el programa que abre una **ventana con botones** donde cualquier persona puede subir una foto de una flor, pulsar un boton y ver que responde el modelo. No hace falta saber programar para usarla: solo hay que abrir la pagina, elegir la foto y mirar el resultado.

Dos palabras que se repiten y conviene entender antes de seguir:

- **interfaz grafica**: una pantalla con elementos visuales (cajas, botones, etiquetas) que se manejan con el raton, en lugar de escribir comandos de texto. El programa de la consola de la seccion anterior tambien funciona, pero exige escribir la ruta de la foto a mano cada vez.
- **Gradio**: la libreria (un conjunto de piezas ya hechas) que se usa para construir esa pantalla. Con Gradio se describe "quiero una caja para subir una imagen y una etiqueta de salida" y la libreria crea la pagina web, la publica en el ordenador y la deja lista para usar. Gracias a eso no hubo que escribir ni una linea de la parte web.

### Decision de fondo: la demo nunca debe fallar

Todo el archivo esta escrito alrededor de una regla: **pase lo que pase, la ventana se abre y explica lo que ocurre en castellano**. Si falta el modelo entrenado, si falta una libreria, si la foto es ilegible o si la prediccion revienta, el usuario ve un mensaje comprensible. Nunca una pantalla en blanco ni un texto tecnico incomprensible.

## La función que hace la predicción: predecir

Aunque el usuario solo ve botones, por debajo hay una funcion que hace el trabajo. Se llama `predecir` y es la que se ejecuta cada vez que se pulsa **Enviar**.

```python
def predecir(
    imagen: Any,
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> tuple[dict[str, float], str]:
```

- `imagen` es la foto que subio el usuario. Si no hay ninguna, se devuelve el aviso `MSG_SIN_IMAGEN` ("Sube una imagen primero") en vez de fallar.
- `checkpoint_path` es el archivo del modelo entrenado.
- `topk` son cuantas especies se muestran (5).
- `min_conf` es el umbral de confianza, que aqui tiene un papel especial que se explica mas abajo.
- Devuelve dos cosas a la vez: un diccionario `especie -> probabilidad` que alimenta la etiqueta de barras, y un texto en formato Markdown que alimenta el recuadro de detalle.

### El aviso de "falta el modelo entrenado"

```python
    if not Path(checkpoint_path).exists():
        return {}, mensaje_entrena_primero(checkpoint_path)
```

```python
def mensaje_entrena_primero(checkpoint_path: str) -> str:
    """Mensaje en Markdown que se muestra cuando falta el checkpoint."""
    return (
        f"{AVISO_ENTRENA}\n\n"
        f"Ruta buscada: `{checkpoint_path}`"
    )
```

- La comprobacion se hace **antes** de intentar nada: si el archivo del modelo no esta, ni se intenta predecir.
- El mensaje no solo dice que falta: dice como arreglarlo (`python -m src.train`) y en que ruta se estaba buscando el archivo. Esa ultima parte ahorra el caso tipico de "si lo entrene, pero en otra carpeta".
- El texto esta en Markdown, el formato de texto sencillo que Gradio sabe mostrar con titulos y negritas.

### La decision importante: avisar en vez de borrar la prediccion

Esta es la parte del archivo que mas veces se cambio, y merece una explicacion larga porque es una leccion practica y no un detalle tecnico.

El problema real, visto en las pruebas: con un modelo poco entrenado, el `softmax` queda **muy plano**. El `softmax` es la operacion que convierte la salida del modelo en porcentajes que suman 100 %. Cuando el modelo esta bien entrenado, la especie correcta se lleva un porcentaje alto (por ejemplo 70 %). Cuando esta poco entrenado, reparte la confianza casi a partes iguales entre las 102 especies y su maximo ronda el 2 %. En modo debug (2 epocas de entrenamiento) el maximo queda en torno al 2 % aunque la especie propuesta sea la correcta.

Con el umbral puesto antes de mirar la lista, la demo respondia "no reconocida / fuera del catalogo" incluso cuando estaba acertando la especie, porque un 2 % nunca llega al 25 %. El usuario recibia un "no la reconozco" que era falso.

La solucion que se adopto: **la demo pide siempre el top-5 con umbral 0 y ensena la respuesta; despues, si la confianza es baja, anade un AVISO encima de la lista en lugar de borrarla.**

```python
    try:
        crudos = predict_image(imagen, checkpoint_path, topk=topk, min_conf=0.0)
```

```python
    confianza = max(prob for _, prob in resultados)
    if confianza < min_conf:
        tabla = (
            f"**AVISO: confianza baja ({confianza * 100:.1f} % < {min_conf * 100:.0f} %).** "
            "La lista de arriba es la mejor suposicion, no una certeza. Puede ocurrir "
            "por dos motivos: (a) la flor no esta entre las 102 especies del catalogo, "
            "o (b) el modelo esta poco entrenado (por ejemplo si entrenaste en "
            "MODO_DEBUG, o si aviso de que no pudo descargar los pesos preentrenados).\n\n"
        ) + tabla
```

- La llamada usa `min_conf=0.0`: se le pide al motor de prediccion que no filtre nada, que devuelva las cinco mejores apuestas aunque su confianza sea minima.
- Despues, `confianza` es el porcentaje de la primera candidata, y si no llega al umbral se antepone un texto de aviso que dice exactamente los dos motivos posibles: o la flor no esta en el catalogo de 102 especies, o el modelo esta poco entrenado.
- La consecuencia para el usuario es que **siempre ve una respuesta util**: la mejor suposicion, su porcentaje, y la advertencia de que hay que desconfiar. Antes veia "no reconocida" y no aprendia nada; ahora ve "creo que es una petunia, pero con un 2 % de confianza, asi que tomalo con pinzas".
- El umbral cambia de significado: deja de ser un interruptor que borra y pasa a ser una senal de calidad. Por eso `--min-conf` sigue existiendo, pero ahora sirve para decidir a partir de que porcentaje no hace falta el aviso.

### Todo lo demas se convierte en un mensaje, no en un error

```python
    except FileNotFoundError as exc:
        return {}, mensaje_entrena_primero(checkpoint_path) + f"\n\nDetalle: `{_celda(exc)}`"
    except Exception as exc:
        return {}, (
            "**Error al predecir la imagen.**\n\n"
            f"`{type(exc).__name__}: {_celda(exc)}`\n\n"
            "Revisa que la imagen sea valida (jpg/png) y que el checkpoint "
            "corresponda al mismo backbone."
        )
```

- `except` significa "si ocurre este problema, haz esto en lugar de romperte". Aqui se capturan dos casos: que falte el archivo del modelo y cualquier otro fallo.
- En los dos casos la funcion devuelve un texto, no lanza el error hacia arriba. Por eso la ventana sigue viva y muestra la explicacion.
- El mensaje incluye el nombre tecnico del error y una recomendacion concreta: revisar que la imagen sea un jpg o png valido y que el modelo corresponda al mismo tipo de red.
- `_celda` limpia el texto para que no rompa la tabla en Markdown.

### La lista de resultados y la tabla

```python
    resultados = _normalizar(crudos, topk)
    if not resultados:
        return {}, MSG_SIN_RESULTADOS

    etiquetas = {nombre: prob for nombre, prob in resultados}
    tabla = tabla_markdown(resultados)
```

- `_normalizar` ordena las candidatas de mayor a menor probabilidad, descarta las que vienen mal formadas y se queda con las `topk` primeras. Existe porque la prediccion puede llegar en varios formatos y aqui se unifican.
- `etiquetas` es el diccionario que Gradio dibuja como barras de porcentaje.
- `tabla_markdown` construye la tabla de texto:

```python
    filas = [
        "| Especie | Probabilidad |",
        "| :--- | ---: |",
    ]
    for etiqueta, prob in resultados:
        filas.append(f"| {_celda(etiqueta)} | {_fmt_pct(prob)} |")

    ganadora = _celda(resultados[0][0])
    encabezado = f"**Prediccion principal:** {ganadora} ({_fmt_pct(resultados[0][1])})\n\n"
    return encabezado + "\n".join(filas)
```

- Se genera una tabla con dos columnas: el nombre de la especie y su probabilidad en porcentaje con dos decimales.
- Antes de la tabla se escribe en negrita cual es la prediccion principal, porque es el dato que la gente busca primero.
- Que existan las dos salidas (barras y tabla) es a proposito: las barras dan la impresion rapida de la confianza, la tabla da los numeros exactos.

## build_app: construir la ventana

`build_app` monta toda la pantalla. No abre el servidor: solo deja la ventana preparada y la devuelve, para que sea el llamador quien decida cuando encenderla.

```python
def build_app(
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> gr.Blocks:
```

```python
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
```

- `falta_checkpoint` se calcula una sola vez, al construir la pantalla, y se usa despues para decidir si se ensena el aviso.
- `_fn` es la funcion concreta que se ejecutara en cada clic: recuerda la ruta del modelo, cuantas especies y que umbral se pidieron al construir la app.
- `gr.Blocks` es la pieza de Gradio que permite describir una pantalla por partes y colocarlas como se quiera.
- `ejemplo` busca la primera imagen que haya en las carpetas `data` o `assets` para ofrecer un ejemplo de un clic. Si no hay ninguna, no se muestra ese panel.
- El aviso de "no hay modelo entrenado" se coloca **dentro del diseno de la pantalla**, no como respuesta a un clic. Asi el usuario lo ve nada mas abrir, sin tener que probar.

### La caja para subir o fotografiar la imagen

```python
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
```

- `entrada` es la caja de la imagen. Se admiten tres formas de dar la foto: **upload** (subir un archivo del ordenador), **webcam** (hacer la foto con la camara) y **clipboard** (pegarla desde el portapapeles). Cuantas mas vias, menos gente se queda fuera.
- `type="pil"` indica en que formato llega la imagen al programa: una imagen ya abierta en memoria, que es justo lo que sabe interpretar `predict_image` de la seccion anterior.
- **Enviar** lanza la prediccion; **Limpiar** vacia la caja y los resultados para probar otra foto. Tener un boton de limpiar evita que el resultado anterior se confunda con el nuevo.
- `gr.Examples` coloca una fila de ejemplos ya cargados; al pulsarla, la imagen se rellena sola.
- La pantalla se organiza en dos columnas dentro de una fila: a la izquierda la entrada, a la derecha los resultados. Se fija un ancho minimo de 320 pixels por columna para que en una pantalla pequena las columnas se apilen en lugar de aplastarse.

### La etiqueta con las cinco especies y la tabla de detalle

```python
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
```

- `gr.Label` es el recuadro de barras: muestra hasta `topk` especies con su porcentaje. De un vistazo se ve cual gana y por cuanto.
- `gr.Markdown` es el recuadro de texto: muestra el aviso de confianza baja, la prediccion principal en negrita y la tabla especie-probabilidad.
- La tabla arranca con un texto de instrucciones ("Sube una imagen para ver el detalle de las top-k") en lugar de vacia, para que el usuario sepa que hay que hacer. Si falta el modelo, arranca directamente con el aviso.

### Los botones que disparan la prediccion

```python
        boton_enviar.click(fn=_fn, inputs=entradas, outputs=salidas, api_name="predict")
        entrada.upload(fn=_fn, inputs=entradas, outputs=salidas, api_name=None)
        boton_limpiar.click(
            fn=lambda: (None, {}, "Sube una imagen para ver el detalle de las top-k."),
            inputs=None,
            outputs=[entrada, salida_etiquetas, salida_tabla],
            api_name=None,
        )
```

- La primera linea conecta el boton **Enviar** con la funcion de prediccion: `inputs` son las entradas y `outputs` las salidas que se rellenan.
- La segunda hace que la prediccion salga **tambien al subir la imagen**, sin esperar al clic. Es comodidad pura: subir y ver el resultado a la vez.
- El boton **Limpiar** no predice nada: deja la caja vacia, las barras vacias y el recuadro con el texto de instrucciones.
- `api_name="predict"` publica la funcion de prediccion con un nombre estable en la direccion web de la demo, de modo que se puede llamar desde fuera. `api_name=None` en los atajos evita publicar automatismos internos que no aportan nada a nadie.
- Nota de implementacion: en la version de Gradio instalada (6.30.0) el evento `submit` ya no existe en los componentes, por eso se usa el clic del boton y la subida de la imagen. Esta desviacion esta documentada en la cabecera del archivo, junto con el motivo de que `build_app` devuelva un `gr.Blocks` en lugar del `gr.Interface` que preveia el contrato original: con `gr.Interface` no se puede dejar el aviso de "falta el checkpoint" fijo en la pantalla, y ese aviso se considero mas importante que la forma exacta del objeto devuelto.

## main y cli: encender la ventana

`build_app` construye; `main` enciende. La separacion permite probar la pantalla sin abrir un servidor.

```python
def main(
    share: bool = False,
    checkpoint_path: str = "resultados/best.pt",
    topk: int = 5,
    min_conf: float = 0.25,
) -> None:
```

```python
    if not Path(checkpoint_path).exists():
        print(
            f"[aviso] No se encontro el checkpoint: {checkpoint_path}\n"
            "        La demo se abrira igual, pero mostrara el aviso "
            "'entrena primero: python -m src.train'.\n"
            "        Entrena con:  python -m src.train",
            flush=True,
        )
```

- Si falta el modelo, `main` no se niega a arrancar: avisa por consola, abre la ventana igual y deja que sea la propia pantalla la que explique la situacion.
- `flush=True` obliga a que el mensaje salga inmediatamente en pantalla. Sin eso, en algunos entornos el texto se queda retenido y el usuario cree que el programa se colgo.

```python
    app = build_app(checkpoint_path, topk, min_conf)

    # En notebooks/Colab no bloqueamos la celda; en script el servidor sigue vivo.
    kwargs: dict[str, Any] = {"share": share, "prevent_thread_lock": _es_notebook()}

    # `inbrowser` no aporta nada en Colab.
    if "inbrowser" in inspect.signature(app.launch).parameters:
        kwargs["inbrowser"] = not _es_notebook()
```

```python
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
```

- `_es_notebook` detecta si el programa corre dentro de un cuaderno tipo Colab. En ese caso no se bloquea la ejecucion de la celda; en un programa normal el servidor se queda encendido hasta que se interrumpe con **Ctrl+C**, y eso es lo que se avisa al final.
- La direccion local se imprime siempre, para poder copiarla y abrirla. En las pruebas se levanto un servidor real y quedo escuchando en `127.0.0.1:7860`; `127.0.0.1` significa "este mismo ordenador" y `7860` es el numero de puerta que usa Gradio.
- Si se pidio enlace publico y no llego, se avisa igual en lugar de quedarse en silencio: normalmente significa que no hay conexion a internet.

### Las opciones de la linea de comandos

```python
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
```

```python
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
```

- El comando tipico es `python -m src.app` y la ventana queda disponible en la direccion local que se imprime.
- `--checkpoint` apunta al archivo del modelo; sirve para usar uno entrenado en otra carpeta.
- `--topk 5` decide cuantas especies se ensenan.
- `--min-conf 0.25` es el umbral a partir del cual ya no aparece el aviso de confianza baja. El propio texto de ayuda recomienda bajarlo si el modelo esta poco entrenado.
- `--share` genera un **enlace publico temporal**: Gradio levanta un tunel hacia internet y devuelve una direccion que cualquiera puede abrir desde su navegador durante un rato, sin estar en la misma red ni instalar nada. En Google Colab es imprescindible, porque la maquina que ejecuta el programa esta en la nube y su `127.0.0.1` no lo ve nadie. El comando alli es `python -m src.app --share`.
- `--share` es opcional y por defecto esta apagado: `share=False`. La razon es que publicar una direccion accesible desde internet es una decision consciente, no algo que deba pasar por descuido.

```python
def cli(argv: list[str] | None = None) -> None:
    """Punto de entrada de `python -m src.app`."""
    args = parse_args(argv)
    main(
        share=bool(args.share),
        checkpoint_path=str(args.checkpoint),
        topk=int(args.topk),
        min_conf=float(args.min_conf),
    )
```

- `parse_args` lee lo que se escribio en la consola y `cli` se lo pasa a `main`. Son dos funciones separadas para que se pueda probar la lectura de opciones sin abrir ninguna ventana.

## Por qué se hizo asi

- **Se uso Gradio y no una pagina web propia** porque Gradio ya trae hecha toda la parte dificil: subir archivos, usar la webcam, pegar desde el portapapeles, dibujar las barras de porcentaje, mostrar Markdown y servir la pagina en el ordenador. Escribir eso a mano habria costado muchisimo mas tiempo y habria anadido mucho mas codigo que revisar, sin ninguna ventaja para el objetivo del proyecto, que es que se vea el modelo funcionando.
- **La demo muestra siempre el top-5 y el umbral pasa a ser un aviso** por lo que se aprendio en las pruebas: con un modelo poco entrenado el `softmax` es plano, el maximo ronda el 2 %, y filtrar por umbral producia "no reconocida" incluso acertando la especie. Ensenar la respuesta con una advertencia siempre es mas util que ocultarla: el usuario ve la mejor suposicion, ve su porcentaje y sabe si puede fiarse.
- **El aviso de confianza baja explica las dos causas posibles** ("no esta en el catalogo" o "el modelo esta poco entrenado") porque desde fuera las dos situaciones se ven igual: porcentajes bajos. Sin esa explicacion, quien prueba la demo concluiria que el sistema no sirve, cuando en realidad puede estar funcionando con un modelo a medio entrenar.
- **La app no debe fallar nunca** porque una demo que se rompe delante de alguien que no programa no se vuelve a probar. Por eso se comprueba el checkpoint antes de predecir, se capturan los errores y se traducen a mensajes con instrucciones concretas, y se importa el motor de prediccion de forma perezosa (dentro de la funcion), de modo que la ventana se pueda abrir aunque falte una libreria.
- **La aplicacion se abre igual aunque falte el checkpoint**, mostrando el aviso dentro de la pantalla. La alternativa (negarse a arrancar) dejaria al usuario sin saber que pasa; asi ve la interfaz y lee exactamente el comando que tiene que ejecutar.
- **`share=False` por defecto** porque abrir un enlace publico en internet es una decision con consecuencias y no debe ocurrir porque alguien haya olvidado una opcion. En Colab se pide de forma explicita con `--share`, que es donde de verdad hace falta.
- **`main` y `build_app` estan separadas** para poder construir y probar la pantalla sin levantar un servidor: es lo que permite verificar que la app se arma correctamente en unos segundos, sin dejar un proceso escuchando en un puerto.
- **Se imprime siempre la direccion local y el recordatorio de Ctrl+C** porque quien usa la demo necesita saber donde mirar y como pararla; son dos frases que evitan muchas preguntas.
- **La desviacion respecto al contrato original esta documentada en la cabecera del archivo**: la version instalada de Gradio es la 6.30.0, en la que el evento `submit` desaparecio, y con `gr.Interface` no se podia dejar el aviso de "falta el checkpoint" fijo en la pantalla. Se prefirio una desviacion explicada y justificada a mantener la forma prevista perdiendo funcionalidad visible para el usuario, y el objeto devuelto sigue cumpliendo lo esencial: tiene `.launch(...)`.
- **Se cachea el modelo y se usa un unico motor de prediccion compartido con la consola** para que la demo no duplique logica: la misma funcion que responde en la ventana es la que responde en el comando `python -m src.predict`. Asi lo que se prueba en la pantalla y lo que se mide en la evaluacion son exactamente lo mismo.
