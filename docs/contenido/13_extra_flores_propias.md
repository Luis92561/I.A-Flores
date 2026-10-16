# El extra: entrenar con tus propias flores

## Para que sirve este archivo (en lenguaje llano)

Este archivo explica `src/custom.py`, el modulo que permite ensenar al sistema **tus** flores. El proyecto principal reconoce 102 especies de flores, pero son las 102 que eligio la Universidad de Oxford: probablemente ninguna sea la de tu jardin. Con este extra puedes entrenar un modelo nuevo con las especies que tu quieras, usando tus propias fotos, **sin tocar el modelo original**.

Tres palabras que conviene entender antes de seguir:

- **catalogo**: el conjunto de cosas que el sistema sabe reconocer. El proyecto trae dos: el de las 102 especies de Oxford (el original) y el tuyo, el que tu construyes con tus fotos.
- **backbone**: la parte grande y ya entrenada de la red, la que se encarga de mirar la imagen y detectar formas, colores y texturas. Es como el ojo del sistema, y ya viene ensenado de fabrica.
- **checkpoint**: una foto fija del modelo entrenado: sus ajustes internos, sus clases y su configuracion, guardados en un solo archivo. Cuando ves una ruta terminada en `.pt` (por ejemplo `resultados/custom/best.pt`), eso es un checkpoint.

La idea de fondo es sencilla: **el sistema solo puede responder lo que ha aprendido**. Si le ensenas fotos de tus flores, etiquetadas con su nombre, el sistema aprende a reconocerlas tambien. Este modulo es el que hace ese trabajo de principio a fin, y convive con el proyecto original sin modificarle ni una linea.

## Que hay dentro

### El problema que resuelve y la idea clave

El modelo original esta entrenado con 102 especies concretas. Su respuesta, ante cualquier foto, es siempre una de esas 102: si le ensenas una flor de tu casa que no esta en la lista, devolvera la especie de Oxford que mas se le parezca, con mayor o menor acierto. No es un fallo del sistema: es que **nunca le han ensenado tu flor**.

La solucion de este extra es entrenar un **modelo aparte** con tus fotos, en lugar de intentar ampliar el de las 102. Y aqui aparece la idea clave que hace que todo funcione sin cambiar codigo: **cuando se guarda un modelo, se guardan dentro sus clases y su backbone**. Se puede ver en la funcion `save_checkpoint` del archivo `src/model.py`:

```python
def save_checkpoint(
    path: str | Path,
    model: FlowerClassifier | nn.Module,
    class_names: Iterable[str],
    cfg: Config,
    **extra: Any,
) -> None:
```

```python
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "class_names": list(class_names),
        "backbone": backbone_name,
        "num_classes": num_classes,
        "img_size": int(getattr(cfg, "img_size", IMAGE_SIZE)),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "epoch": int(extra.pop("epoch", -1)),
        "val_acc": float(extra.pop("val_acc", 0.0)),
    }
```

- El checkpoint no guarda solo los numeros aprendidos (`model_state`): guarda tambien **la lista de nombres de las clases** (`class_names`), **cuantas son** (`num_classes`), **que backbone se uso** y **de que tamano eran las fotos** (`img_size`).
- La consecuencia practica es enorme y conviene decirla despues: **el modelo se guarda con su propia etiqueta**. Las funciones que leen un checkpoint no necesitan que nadie les diga cuantas especies hay ni como se llaman, porque lo leen del propio archivo.
- Por eso `predict_image` (prediccion de una foto suelta), `evaluate` (la medicion de resultados) y la demo funcionan con un modelo entrenado con tus flores **sin cambiar una sola linea de esas piezas**. Para ellas un modelo es un modelo: un archivo con sus pesos, sus clases y su configuracion.

### La funcion que revisa tus fotos: revisar_carpeta

Antes de gastar tiempo entrenando, conviene saber si las fotos estan bien puestas. De eso se encarga `revisar_carpeta`, que mira la carpeta y devuelve un informe en forma de diccionario (una lista de parejas etiqueta-valor; por ejemplo, la etiqueta `total` con el numero de fotos encontradas).

```python
def revisar_carpeta(
    ruta: str | Path = RUTA_FOTOS_POR_DEFECTO,
    min_por_clase: int = MIN_POR_CLASE_POR_DEFECTO,
) -> dict:
```

```python
        carpetas_vacias: list[str] = []
        sueltas: list[str] = []
        try:
            entradas = sorted(base.iterdir())
        except OSError as exc:  # carpeta ilegible: se informa, no se lanza
            entradas = []
            problemas.append(f"no se pudo leer la carpeta '{base}' ({type(exc).__name__}: {exc}).")
        for entrada in entradas:
            if entrada.is_dir():
                fotos = _fotos_de(entrada)
                clases[entrada.name] = len(fotos)
                if not fotos:
                    carpetas_vacias.append(entrada.name)
            elif entrada.is_file() and entrada.suffix.lower() in EXTENSIONES_FOTO:
                sueltas.append(entrada.name)
```

- Recorre la carpeta que le digas y, por cada **subcarpeta**, cuenta las fotos que hay dentro. La subcarpeta es una especie y su nombre sera la etiqueta que aprenda el modelo.
- Se fija en dos errores tipicos: **carpetas de especie vacias** (creadas pero sin fotos dentro) y **fotos sueltas** metidas directamente en la carpeta principal, sin su subcarpeta de especie. Las dos cosas impiden entrenar y el informe lo dice con el nombre del archivo o de la carpeta.
- Solo cuenta los formatos admitidos (`.jpg`, `.jpeg`, `.png` y `.webp`). Un archivo suelto de otro tipo (por ejemplo un documento de texto) se ignora sin mayor problema.

```python
    for nombre in sorted(con_fotos):
        n = con_fotos[nombre]
        if n < minimo:
            avisos.append(
                f"la especie '{nombre}' tiene {n} foto(s): menos de {minimo}. Con tan "
                f"pocas fotos el resultado sera pobre; anade mas o usa el "
                f"{MODO_MATRICULACION}."
            )
```

```python
        if minimo_real > 0 and maximo / float(minimo_real) > DESEQUILIBRIO_UMBRAL:
            mejor = max(sorted(con_fotos), key=lambda nombre: con_fotos[nombre])
            peor = min(sorted(con_fotos), key=lambda nombre: con_fotos[nombre])
            avisos.append(
                f"las clases estan desequilibradas: '{mejor}' tiene {maximo} fotos y "
                f"'{peor}' solo {minimo_real} (x{maximo / float(minimo_real):.1f}). El "
                "modelo tendera a confundir la especie con menos fotos; iguala el "
                "numero de fotos por especie."
            )
```

- **Pocas fotos**: si una especie tiene menos de 10 fotos, se anota un aviso. El numero 10 es el minimo que se considera razonable, y se puede cambiar al llamar a la funcion.
- **Desequilibrio**: si la especie con mas fotos tiene mas del triple que la que menos, se anota otro aviso. El motivo es que el modelo aprende mas de lo que mas ve: con 80 fotos de una especie y 8 de otra, acabara confundiendo la segunda. La solucion que propone el informe es igualar el numero de fotos por especie.

```python
    if avisos:
        return (
            f"Se puede entrenar con {n_clases} especie(s) y {total} foto(s), pero el "
            "resultado mejorara anadiendo fotos: lo ideal son 20-50 por especie. "
            f"Con menos de {MIN_POR_CLASE_POR_DEFECTO} fotos por especie conviene "
            f"esperar al {MODO_MATRICULACION}."
        )
    return (
        f"Carpeta lista: {n_clases} especie(s) y {total} foto(s). "
        f"Entrena con: `python -m src.custom --fotos {ruta}`."
    )
```

- Al final el informe siempre trae una **recomendacion en una frase**: que hacer con la carpeta tal como esta, con el comando exacto que toca ejecutar.
- **Aviso no es lo mismo que problema.** Un problema (por ejemplo "no hay ninguna foto") impide entrenar y bloquea el paso siguiente; un aviso (pocas fotos, desequilibrio) solo advierte de que el resultado sera mejorable. El campo `ok` del informe es verdadero solo cuando no hay ningun problema.
- La decision de diseno es **informar en vez de fallar**: si la funcion se negara a seguir con 8 fotos por especie, no podrias hacer la prueba rapida para ver si el flujo funciona. Te dice lo que hay, te avisa de lo que falta y te deja decidir.
- La recomendacion honesta es esta, y no conviene endulzarla:

| Fotos por especie | Que esperar |
|---|---|
| Menos de 10 | Resultado pobre: el modelo casi no tiene material con el que aprender. |
| 20 a 50 | Resultado decente: es el rango recomendado. |
| Mas de 50 | Mejor todavia, sobre todo si varian el fondo, la luz y el angulo. |

- Con **2 o 3 fotos por especie** el sistema aprende de memoria esas fotos concretas y falla con cualquier otra. Ahi no hay magia posible: un modelo de reconocimiento de imagenes necesita variedad para aprender la especie y no la foto.
- El mensaje de aviso menciona un "modo de matriculacion" para catalogos muy pequenos: es una idea apuntada en el codigo, pero **todavia no esta implementada**. Dicho claro: si tienes menos de 10 fotos por especie, la unica via que existe hoy es anadir mas fotos.

### La funcion que prepara las fotos: preparar_loaders

`preparar_loaders` convierte la carpeta de fotos en la forma que el entrenamiento entiende. Un **DataLoader** (cargador de datos) es el encargado de servir las fotos al modelo: las busca en el disco, las prepara y las entrega ya agrupadas en lotes (grupos de fotos que el modelo mira de golpe antes de corregirse).

```python
def preparar_loaders(
    cfg: Config,
    ruta_fotos: str | Path = RUTA_FOTOS_POR_DEFECTO,
    val_fraccion: float = VAL_FRACCION_POR_DEFECTO,
    semilla: int | None = None,
) -> dict:
```

- El formato esperado de las fotos es **`mis_flores/<especie>/<foto>`**: una carpeta por especie, con el nombre de la especie, y dentro sus fotos. Esto no es un capricho: es el formato estandar que espera la pieza de PyTorch llamada `ImageFolder` (una utilidad que construye un catalogo leyendo carpetas, y que toma el nombre de cada carpeta como etiqueta).
- `val_fraccion` decide que parte de cada especie se reserva para **validacion**: el trozo de fotos que no se usa para aprender, sino para vigilar si el modelo mejora de verdad o solo memoriza. Lo habitual es 0,2, es decir, un 20 por ciento.
- `semilla` es el numero inicial que gobierna los sorteos. Fijandolo, el reparto de fotos sale siempre igual, y una corrida se puede repetir.

```python
    informe = revisar_carpeta(ruta_fotos)
    if not informe["ok"]:
        raise ValueError(
            "El catalogo de fotos no es utilizable: " + " | ".join(informe["problemas"])
        )

    escasas = sorted(n for n, total in informe["clases"].items() if 0 < total < 2)
    if escasas:
        raise ValueError(
            "Cada especie necesita al menos 2 fotos para poder reservar una en "
            f"validacion sin dejar la especie sin entrenamiento: {escasas}."
        )
```

- Primero reutiliza la revision anterior. Si el catalogo no sirve (no hay fotos, o solo hay una especie), se detiene con un mensaje que lista los problemas. `raise ValueError` es la forma que tiene el programa de decir "no puedo seguir, y este es el motivo exacto".
- **El unico caso que bloquea por numero de fotos es que una especie tenga menos de 2.** La razon es aritmetica: si una especie tiene una sola foto y esa foto se reserva para validacion, la especie se queda sin nada con lo que aprender. Con 2 fotos ya se puede hacer la prueba, aunque el resultado sea pobre.
- Fijate en que el modulo no se inventa un minimo mas alto: con 3 fotos por especie te deja entrenar, aunque el informe te haya avisado de que el resultado no sera bueno. La puerta esta abierta y el cartel de aviso puesto.

```python
    def __call__(self, imagen):
        if isinstance(imagen, Image.Image):
            imagen = ImageOps.exif_transpose(imagen)
            if imagen.mode != "RGB":
                imagen = imagen.convert("RGB")
        return self.transform(imagen)
```

- Aqui esta la **limpieza de las fotos de movil**, que es la misma para todas las fotos antes de que el modelo las vea.
- `ImageOps.exif_transpose` corrige el giro. Cuando haces una foto vertical con el movil, el aparato no reordena los pixeles: guarda la imagen tumbada y anota en los **metadatos EXIF** (unos datos ocultos dentro del archivo de imagen) que hay que girarla 90 grados para verla bien. El movil y el visor de fotos lo hacen al mostrarla, pero el programa no, y veria la flor tumbada. Esta orden aplica ese giro de verdad, de modo que no haya fotos tumbadas entre las que se aprende.
- `convert("RGB")` garantiza que cada foto tenga los tres canales de color normales (rojo, verde y azul). Una foto en blanco y negro, en tonos de paleta o con transparencia tiene otra estructura, y el modelo solo entiende tres canales. Sin esta conversion, esas fotos harian fallar el entrenamiento.
- Las dos lineas estan juntas y antes de las transformaciones, porque el recorte y el cambio de tamano tienen que aplicarse despues del giro, no antes.

```python
    generador = random.Random(int(semilla))
    indices_train: list[int] = []
    indices_val: list[int] = []
```

```python
    for etiqueta in sorted(por_clase):
        indices = list(por_clase[etiqueta])
        generador.shuffle(indices)  # deterministico con la semilla
        total = len(indices)
        n_val = max(1, int(round(total * float(val_fraccion))))
        n_val = min(n_val, total - 1)  # train conserva siempre al menos una foto
        indices_val.extend(indices[:n_val])
        indices_train.extend(indices[n_val:])
```

- El reparto se hace **especie por especie**, no echando todas las fotos en un monton. Asi cada especie aporta su parte a la validacion, y ninguna se queda fuera de ella.
- Es **determinista**: el mismo numero inicial produce siempre el mismo reparto. Se barajan los numeros de posicion de las fotos de esa especie y se corta en dos trozos, uno para entrenamiento y otro para validacion.
- Y es **sin fuga**. La fuga de datos es el error clasico de este tipo de proyectos: dejar la misma foto en el grupo de entrenamiento y en el de validacion. Si eso ocurre, el modelo ya ha visto esa foto y la validacion da un resultado artificialmente bueno, que no se repite con fotos nuevas. Aqui los indices se cortan en dos trozos que no se solapan, y por construccion una foto no puede estar en los dos lados.
- Fijate tambien en `n_val = min(n_val, total - 1)`: el grupo de entrenamiento conserva **siempre al menos una foto** de cada especie, aunque la fraccion de validacion sea alta.

### La funcion que entrena: entrenar_custom

`entrenar_custom` es la pieza que hace el trabajo largo. Monta su propia configuracion, entrena y deja el modelo guardado junto a un archivo de trazabilidad.

```python
    base = _cfg_base(cfg)
    destino = Path(out_dir or OUT_DIR_POR_DEFECTO)
    paquete = preparar_loaders(
        base, ruta_fotos=ruta_fotos, val_fraccion=val_fraccion, semilla=None
    )
    class_names = list(paquete["class_names"])

    cfg_final = replace(
        base,
        num_classes=int(len(class_names)),
        out_dir=str(destino),
    )
```

- `_cfg_base` hace una **copia** de la configuracion recibida, o crea una con los valores propios del modulo si no se le pasa ninguna. Copiar importa: si se modificara la configuracion original del llamador, se estarian cambiando ajustes del proyecto principal por la puerta de atras.
- Los valores propios de este modulo son **15 epocas, lotes de 16 fotos, backbone EfficientNet-B0 y pesos preentrenados**. Son mas modestos que los de la corrida grande del proyecto (25 epocas y lotes de 32) a proposito: un catalogo propio suele tener muchas menos fotos, y con pocas fotos conviene dar menos vueltas y mirar menos fotos de golpe para no memorizar.
- `num_classes` se ajusta al **numero de especies que hayas puesto**, y la salida va a `resultados/custom` por defecto. Nada de esto toca la carpeta `resultados/` del proyecto original: los dos modelos viven en carpetas distintas y ninguno pisa al otro.
- `out_dir` es la carpeta donde quedaran el checkpoint, el historial de entrenamiento y las figuras.

```python
    historial = _entrenar_con_reintento(
        cfg_final, _loaders_inyectables(paquete), ruta_fotos, val_fraccion
    )
```

```python
    try:
        return train(cfg, loaders=loaders)
    except Exception as exc:  # noqa: BLE001 - se decide con _parece_fallo_de_workers
        workers = int(getattr(cfg, "num_workers", 0) or 0)
        if workers <= 0 or not _parece_fallo_de_workers(exc):
            raise
```

- Aqui esta la **unica pieza del codigo ya verificado que hubo que tocar** para que este extra existiera. La funcion de entrenamiento del proyecto, `train`, ahora acepta un parametro opcional llamado `loaders`, que son los lotes ya preparados que se le entregan desde fuera:

```python
def train(cfg: Config, *, loaders: tuple | None = None) -> dict:
```

```python
    if loaders is None:
        train_dl, val_dl, _test_dl, class_names = build_dataloaders(cfg)
    else:
        train_dl, val_dl, _test_dl, class_names = loaders
```

- Si nadie le pasa lotes, `train` los construye como siempre a partir del catalogo de Oxford: el proyecto original se comporta **exactamente igual que antes**. Si se los pasan, usa esos y ni siquiera mira el catalogo de Oxford.
- Por que se hizo asi, en vez de copiar el bucle de entrenamiento dentro de este modulo: duplicar ese bucle habria significado mantener dos copias de la misma logica. La copia se desincroniza sola con el tiempo, y entonces el modelo de tus flores y el de las 102 especies se entrenarian con reglas distintas sin que nadie lo note. Tocar **una linea** de la funcion verificada (el parametro nuevo) y reutilizarla entera es menos codigo nuevo, menos sitios donde puede haber un fallo y una sola manera de entrenar en todo el proyecto.
- El parametro se declara despues del asterisco (`*, loaders`), lo que en Python significa que hay que escribirlo con su nombre al llamar. Asi añadirlo no cambio la forma de llamar a `train` en ningun sitio del proyecto.
- El `try` de arriba es un reintento defensivo: si los ayudantes que leen las fotos del disco fallan por una limitacion del sistema operativo, la corrida se repite una vez sin esos ayudantes (leyendo las fotos en el proceso principal) en lugar de perderse. En Colab y Linux no se activa nunca.

```python
    catalogo = {
        "clases": class_names,
        "fotos_por_clase": dict(paquete["informe"]["clases"]),
        "reparto_por_clase": paquete["n_por_clase"],
        "total": int(paquete["informe"]["total"]),
        "backbone": str(cfg_final.backbone),
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ruta_fotos": str(Path(ruta_fotos)),
        "val_fraccion": float(val_fraccion),
        "semilla": int(cfg_final.seed),
```

- Ademas del modelo, se escribe un archivo llamado `catalogo.json` en la misma carpeta. Es un **archivo de texto plano** que guarda la procedencia de los datos: que especies habia, cuantas fotos por especie, como se repartieron entre entrenamiento y validacion, que backbone se uso, la fecha y la semilla.
- Sirve para dos cosas muy concretas. La primera es saber dentro de un mes con que fotos se entreno ese modelo. La segunda es tecnica: al evaluar, el programa lee de `catalogo.json` con que fraccion y con que semilla se repartieron las fotos, de modo que evalue **sobre el mismo grupo de validacion** con el que se entreno, y no sobre uno distinto.
- Tambien guarda la configuracion completa de la corrida, para poder saber despues con cuantas epocas y con que tamano de lote se entreno.

### La funcion que mide el resultado: evaluar_custom

`evaluar_custom` responde a la pregunta "que tal quedo el modelo". Y la respuesta corta es: reutiliza la evaluacion del proyecto tal cual, sin logica nueva.

```python
def evaluar_custom(
    checkpoint_path: str | Path,
    cfg: Config | None = None,
    ruta_fotos: str | Path = RUTA_FOTOS_POR_DEFECTO,
    split: str = "val",
) -> dict:
```

- `split` elige sobre que fotos se mide: `"val"` es el grupo de validacion (las fotos que el modelo no uso para aprender) y `"train"` es el grupo de entrenamiento. Mirar las dos cosas es util: si sobre entrenamiento el resultado es perfecto y sobre validacion es malo, el modelo ha memorizado en lugar de aprender, y eso se llama **sobreajuste**.
- El catalogo propio solo reserva fotos de validacion, por eso `"val"` y `"test"` miden sobre las mismas fotos. No es un descuido: el catalogo de Oxford trae un grupo de prueba oficial separado y el tuyo no lo tiene, porque las fotos las pones tu.

```python
    paquete = preparar_loaders(
        cfg_efectiva, ruta_fotos=ruta_fotos, val_fraccion=fraccion, semilla=semilla
    )
    cargador = _cargador(cfg_efectiva, paquete)

    try:
        return evaluate(ckpt, cfg_efectiva, split=nombre_split, dataloader=cargador)
```

- Se preparan las fotos, se elige el cargador del grupo pedido y se le entrega a la funcion de evaluacion del proyecto (`evaluate`), que es la misma que produce las cifras del informe.
- El detalle tecnico importante es que `evaluate` acepta **un cargador ya construido** (ese es el parametro `dataloader`), igual que `train` acepta unos lotes ya construidos. Gracias a esa puerta, la evaluacion de tus flores es literalmente la misma pieza que la de las 102 especies: las mismas cifras (acierto a la primera, acierto entre las cinco primeras, medida equilibrada por especie), el mismo archivo de resultados (`metrics.json`), la misma tabla de confusiones y el mismo listado en hoja de calculo.
- Al no haber logica nueva no hay dos maneras distintas de medir en el proyecto, y por tanto no hay forma de que las cifras de un catalogo y las del otro no sean comparables entre si.

### La demo con dos modelos

La demo es la ventana con botones donde cualquiera sube una foto y ve la respuesta. Con el extra, esa ventana puede llevar **dos modelos a la vez**: el de las 102 especies de Oxford y el tuyo. Se configura con una opcion llamada `custom_checkpoint`, que es la ruta de tu checkpoint.

```python
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
```

- El orden es deliberado: **primero responde el modelo de las 102 especies**. Si su confianza maxima llega al umbral (0,25 por defecto, es decir, un 25 por ciento), esa es la respuesta y el modelo propio ni se consulta.
- Solo si el modelo grande **no esta seguro** entra el tuyo: se le pregunta por la misma foto y, si su confianza si llega al umbral, la respuesta que se muestra es tu especie, acompanada de la linea "Reconocida por el catalogo propio".
- Si **ninguno de los dos** llega al umbral, la ventana ensena las dos listas encabezadas por un aviso de confianza baja. Es decir: no oculta la duda, la declara. El mensaje explica que la mejor suposicion no es una certeza.
- El motivo del orden es simple: el modelo de Oxford esta entrenado con miles de fotos profesionales y 102 especies, asi que en su terreno es mucho mas fiable que un modelo tuyo con 20 fotos por especie. Preguntarle primero no le resta nada a tu catalogo: solo evita que un modelo pequeno y con pocos datos opaque una respuesta mucho mas solida.
- Y si tu checkpoint no existe o esta mal escrito, la demo no se cae: lo avisa y sigue respondiendo con el modelo de las 102 especies. Lo mismo si tu checkpoint esta corrupto o se entreno con otro backbone.

### La linea de comandos

Todo el extra se puede usar sin escribir programa, con tres ordenes de texto. Son estas:

```bash
python -m src.custom --revisar
python -m src.custom --fotos mis_flores --epochs 15
python -m src.custom --evaluar resultados/custom/best.pt
```

- La primera **solo revisa**: mira la carpeta, imprime el informe (especies, fotos por especie, avisos y recomendacion) y no entrena nada. Es el paso que conviene dar siempre antes de gastar tiempo de entrenamiento, y termina con exito o con error segun el informe.
- La segunda **entrena**. `--fotos` indica la carpeta (por defecto `mis_flores`) y `--epochs` cuantas vueltas dar. El modelo queda en `resultados/custom/`. Si no se indica nada mas, se usan los valores propios del modulo (15 epocas, lotes de 16 fotos, EfficientNet-B0 con pesos preentrenados), porque la linea de comandos parte de esos mismos valores para que la orden sin opciones haga exactamente lo que dice la documentacion.
- La tercera **evalua** un checkpoint ya entrenado y sobre las fotos de validacion, e imprime el resultado en pantalla.
- Hay mas opciones, todas con valor por defecto razonable, para quien necesite afinar:

```python
    parser.add_argument("--epochs", type=int, default=None, help="numero de epocas")
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size", help="tamano de lote"
    )
```

```python
    parser.add_argument("--semilla", type=int, default=None, help="semilla del reparto y del entrenamiento")
    parser.add_argument("--debug", action="store_true", help="corrida corta (smoke test)")
```

- `--batch-size` cambia cuantas fotos se miran de golpe (bajarlo ayuda si el ordenador se queda sin memoria), `--backbone` cambia la red que mira las imagenes, `--device` elige procesador o tarjeta grafica, `--val-fraccion` que parte de cada especie se reserva para validar y `--semilla` fija el sorteo.
- Aqui se ve de donde salen esos valores cuando no se indica ninguna opcion: la linea de comandos arranca con los mismos numeros que usa el modulo por dentro, y solo los cambia si se los pides de forma explicita.

```python
    overrides: dict[str, Any] = {
        "epochs": EPOCHS_POR_DEFECTO,
        "batch_size": BATCH_POR_DEFECTO,
        "backbone": BACKBONE_POR_DEFECTO,
        "pretrained": PRETRAINED_POR_DEFECTO,
        "debug": bool(args.debug),
    }
```

- `--debug` hace una **corrida corta de prueba**: pocas epocas y sin ayudantes de lectura. Sirve para comprobar que todo el flujo funciona antes de lanzar el entrenamiento de verdad, y por precaucion escribe en una carpeta aparte (`resultados/custom_debug`) para no pisar el modelo bueno.
- La opcion de ayuda de cualquier comando (`--help`, que significa "explicame las opciones") esta disponible y esta escrita en espanol, con ejemplos al final.

### Como se usa desde el cuaderno de Colab

El cuaderno del proyecto (`notebooks/flores_colab.ipynb`, el documento que se abre en Google Colab y se ejecuta celda a celda) tiene un **Paso 11 (extra)** dedicado a esto, que ocupa las celdas 25 a 31 y es opcional: los pasos 1 a 10 son el proyecto de las 102 especies y funcionan igual si no se ejecuta nada del Paso 11.

- La celda 11.a puede **crear fotos de ejemplo** (colores planos de dibujo, no flores de verdad) para probar el flujo completo sin tener fotos propias. Nunca pisa fotos que ya existan.
- La celda 11.b llama a `revisar_carpeta` e imprime el informe con las fotos por especie y los avisos.
- Las celdas 11.c y 11.d entrenan y evaluan el modelo propio, con la misma advertencia escrita en pantalla: en modo de prueba las cifras solo demuestran que el flujo funciona, no que el modelo sea bueno.
- La celda 11.e vuelve a construir la demo, ahora con los dos modelos, y la abre en un puerto distinto para no chocar con la demo del paso 9.
- El cuaderno cierra con un **checklist final** (una lista de comprobaciones) que incluye un apartado propio para el Paso 11: que es opcional, donde queda el modelo propio, y la advertencia de que la carpeta de fotos `mis_flores/` no se sube al repositorio.

## Por que se hizo asi

- **Un modelo aparte, y no ampliar el de las 102 especies.** La alternativa tentadora era reentrenar el modelo grande anadiendo tus especies a las 102. Se descarto por tres motivos concretos. El primero es que **protege lo que ya funciona**: el modelo de las 102 esta entrenado, evaluado y auditado, y la entrega del proyecto depende de el; si se reentrenara con tus fotos, cada resultado ya publicado dejaria de corresponder a ese archivo. Con un modelo aparte, un catalogo mal montado no puede romper nada del original, y `resultados/custom/` se puede borrar entero sin consecuencias.
- El segundo motivo es que **con pocas fotos se entrena mejor un modelo pequeno y aparte**. Meter 20 fotos de tu rosa en una red que ya distingue 102 especies obliga a competir contra miles de imagenes de otros catalogos, y el ajuste se vuelve inestable. Entrenar un modelo propio con tus 20 fotos, en cambio, tiene un trabajo mucho mas simple: distinguir tus 3 o 4 especies entre si.
- El tercer motivo es el **olvido catastrofico**: cuando una red entrenada se reajusta con datos nuevos y muy pocos, tiende a perder lo que ya sabia. Es un fenomeno conocido y real, y significa que ampliar el modelo grande con tus fotos podria **empeorar** su rendimiento en las 102 especies de Oxford. Un modelo aparte no corre ese riesgo, porque no toca al otro.
- A lo anterior se suma el **desequilibrio**: tus 20 fotos frente a las miles de Oxford pesarian muy poco en el aprendizaje conjunto, asi que el modelo grande aprenderia tus especies a medias. Separando los catalogos, cada modelo ve la cantidad de datos que le corresponde.
- **Se reutiliza el bucle de entrenamiento y la evaluacion en vez de duplicarlos.** Anadir el parametro de lotes a `train` y usar la puerta que ya existia en `evaluate` costo unas pocas lineas; duplicar el bucle habria costado cientos y habria creado dos versiones de la misma logica destinadas a separarse. Todo lo que ya estaba probado (precision mixta, parada temprana, curvas, matriz de confusiones, hoja de calculo de predicciones) funciona igual para tus flores sin programarlo otra vez.
- **Las fotos no se suben al repositorio.** La carpeta `mis_flores/` esta excluida del sistema de control de versiones (el mecanismo que guarda el historial del proyecto). Los motivos son el tamano y la privacidad: un catalogo de fotos puede pesar cientos de megas, y no tiene sentido que las fotos de tu casa viajen con el codigo. La carpeta viaja con un archivo de instrucciones (`mis_flores/LEEME.md`) que explica el formato y se queda vacia de fotos.
- **La revision informa en lugar de bloquear** porque bloquear impide aprender. Con 8 fotos por especie no se obtiene un buen modelo, pero si se comprueba que el flujo completo funciona, y eso permite corregir el formato de las carpetas antes de hacer 200 fotos mal colocadas. Avisar y dejar seguir es mas util que negarse.
- **Cada cifra de este extra se lee como lo que es: una prueba de que el flujo funciona, no un resultado de calidad.** En una comprobacion con un catalogo de prueba de 2 especies y 8 fotos por especie, entrenar 2 epocas en procesador normal tardo **1,8 segundos**, el checkpoint resultante ocupa **12 MB**, se carga con `num_classes=2` y las clases propias, la evaluacion sobre validacion dio **top1 = 0,5 y top5 = 1,0 sobre 4 imagenes**, y pedir una prediccion devolvio las dos especies propias con un 50,24 por ciento y un 49,76 por ciento. Esos numeros demuestran que el circuito completo (crear carpetas, revisar, entrenar, guardar, cargar, evaluar y predecir) esta bien montado. No dicen nada sobre la calidad del reconocimiento: con 8 fotos por especie y 2 epocas el modelo apenas ha tenido con que aprender, y los dos porcentajes casi identicos son la senal tipica de un modelo que aun no distingue bien las dos especies.
- **Lo que este extra NO hace**, dicho sin rodeos: no aprende porque le ensenes una foto en la demo (la demo clasifica imagenes, no se entrena con ellas; para ensenarle algo hay que pasar por el entrenamiento); no hace magia con 5 fotos por especie; no reconoce especies de las que no tenga ninguna foto; no mejora solo porque pasen mas horas de entrenamiento con las mismas pocas fotos; y no sustituye al modelo de las 102 especies, sino que convive con el.
