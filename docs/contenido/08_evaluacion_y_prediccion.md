# Evaluar y predecir: cómo sabemos si acierta

## Para qué sirve este archivo (en lenguaje llano)

Este archivo explica dos programas del proyecto: `src/evaluate.py` y `src/predict.py`. El primero sirve para poner a prueba al modelo ya entrenado y sacar notas ("de cada 100 fotos, cuantas acierta"); el segundo sirve para usar el modelo entrenado con una foto suelta y obtener una respuesta.

Antes de seguir, tres palabras que se repiten todo el rato y conviene tener claras desde el principio:

- **modelo**: el programa que ha aprendido a reconocer flores. Se guarda en un archivo llamado **checkpoint**, con extension `.pt`. Es como el "cerebro entrenado" guardado en un fichero.
- **conjunto de prueba** (`split` "test"): una lista de fotos con su respuesta correcta escrita al lado, que el modelo nunca ha usado para estudiar. Es el examen final.
- **prediccion**: la respuesta que da el modelo. Por ejemplo, "esta foto es una rosa".

### Por qué "evaluar" y por que se hace sobre el conjunto de prueba

**Evaluar** significa pasarle al modelo un monton de fotos cuya respuesta correcta ya conocemos y comparar lo que responde con lo que deberia responder. El resultado de esa comparacion es la nota del modelo.

Se hace sobre el conjunto de prueba porque es el unico dato honesto. Si examinaramos al modelo con las mismas fotos que uso para estudiar, se pareceria a dejarle el examen copiado: sacaria una nota altisima y no sabriamos si de verdad reconoce flores nuevas. El proyecto usa los tres grupos oficiales del conjunto Oxford Flowers-102: `train` (estudiar), `val` (ajustar durante el estudio) y `test` (el examen final). El examen final tiene 6149 imagenes repartidas en 102 especies.

```python
SPLITS_VALIDOS: tuple[str, str, str] = ("train", "val", "test")
MENSAJE_ENTRENA = "entrena primero: `python -m src.train`"
MAX_CONFUSIONES: int = 20
CABECERA_CSV: tuple[str, ...] = ("ruta", "etiqueta_real", "prediccion", "prob", "correcto")
```

- `SPLITS_VALIDOS` es la lista de los tres grupos permitidos; si alguien pide otro nombre, el programa se queja en vez de hacer algo raro.
- `MENSAJE_ENTRENA` es el texto unico de "todavia no hay modelo entrenado", para que siempre se explique lo mismo.
- `MAX_CONFUSIONES` pone un tope de 20 filas a la lista de confusiones, para no llenar los informes de cientos de casos.
- `CABECERA_CSV` son los cinco nombres de columna del archivo de resultados que se explica mas abajo.

### La decision importante antes de escribir nada: trabajar con lo que trae el checkpoint

`evaluate` no da por hecho como es el modelo: lee del propio checkpoint el tamano de imagen y el numero de especies. Asi, si un checkpoint se entreno con imagenes de 224 pixels, se evalua siempre con 224, aunque quien lo llame se equivoque. Los nombres de las especies se recortan o se rellenan para que la posicion 1 sea la especie 1.

```python
    n_clases = int(meta.get("num_classes") or len(nombres_ckpt) or getattr(cfg, "num_classes", 0))
    if n_clases <= 0:
        raise ValueError(
            f"El checkpoint '{ckpt}' no define num_classes y no se pudo inferir "
            f"(class_names={len(nombres_ckpt)})."
        )
    nombres = _ajustar_nombres(nombres_ckpt, n_clases)

    # El tamano de imagen del checkpoint manda (no asumir 224).
    img_size = int(meta.get("img_size") or getattr(cfg, "img_size", IMAGE_SIZE))
    cfg = _cfg_con_img_size(cfg, img_size)
```

- `meta` es la "etiqueta" que el entrenamiento guardo dentro del checkpoint: con cuantas especies se entreno, con que tamano de imagen y que ajustes se usaron.
- `_ajustar_nombres` garantiza que exista exactamente un nombre por especie; si el checkpoint trae menos nombres de los necesarios, inventa rellenos como `clase_007`.
- Si el checkpoint no dice cuantas especies hay y no se puede deducir, se levanta un error claro en vez de dar resultados sin sentido.

### evaluate: el examen completo

`evaluate` es la funcion principal del archivo. Recibe la ruta del checkpoint, la configuracion, el grupo a examinar (`test` por defecto) y, opcionalmente, un cargador de datos ya hecho para las pruebas automaticas. Devuelve un diccionario con todas las notas y ademas escribe tres cosas en disco: las notas en `metrics.json`, las figuras y el archivo CSV con una fila por imagen.

```python
def evaluate(
    checkpoint_path: str | Path,
    cfg: Config | None = None,
    split: str = "test",
    *,
    dataloader=None,
) -> dict:
```

- `checkpoint_path` es el archivo `.pt` con el modelo entrenado. Si no existe, se lanza `FileNotFoundError` con el recordatorio "entrena primero: `python -m src.train`".
- `cfg` son los ajustes de la corrida. Si no se pasa ninguno, se reconstruye leyendo el propio checkpoint, de modo que los resultados se guardan en la misma carpeta donde se entreno.
- `split` elige el grupo: `train`, `val` o `test`.
- `dataloader` es un parametro solo para pruebas automaticas. En uso normal no se toca.

### _inferir: recorrer todas las fotos y apuntar la respuesta

`_inferir` es la parte que de verdad pasa las fotos por el modelo. Va lote a lote (grupos de imagenes, no una a una, para ir mas rapido) y apunta cinco listas: la respuesta correcta, la respuesta del modelo, la probabilidad que le dio, si acerto entre sus cinco mejores opciones y la ruta del archivo de imagen.

```python
            logits = modelo(imagenes)
            probs = torch.softmax(logits.detach().float(), dim=1)
            k_top5 = max(1, min(5, int(probs.shape[1])))
            valores, indices = probs.topk(k_top5, dim=1)

            reales.extend(int(v) for v in etiquetas.cpu().tolist())
            predichas.extend(int(v) for v in indices[:, 0].cpu().tolist())
            probabilidades.extend(float(v) for v in valores[:, 0].cpu().tolist())
            aciertos = (indices == etiquetas.unsqueeze(1)).any(dim=1)
```

- Las **probabilidades** se obtienen con `softmax`, que convierte las salidas crudas del modelo en numeros entre 0 y 1 que suman 1. Es la forma de leer "cuanta confianza tiene" en cada especie.
- `topk` pide las cinco especies mas probables y las devuelve ya ordenadas de mayor a menor, con sus nombres de posicion.
- `modelo.eval()` y `torch.no_grad()` (unas lineas mas arriba) ponen el modelo en modo examen y le dicen que no hace falta calcular nada para entrenar: eso evita errores y ahorra tiempo.

### Las metricas: como se leen las notas

`_calcular_metricas` es la funcion que traduce todas esas listas a numeros entendibles.

```python
    top1 = float(np.mean(y_true == y_pred))
    top5 = float(np.mean(np.asarray(aciertos_top5, dtype=float))) if len(aciertos_top5) else top1
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=etiquetas, average="macro", zero_division=0)
    )
```

- **top1** (exactitud o acierto directo): de cada 100 fotos, cuantas acierta la especie correcta como primera respuesta. El sistema obtiene 37,8 % sobre las 6149 imagenes del conjunto de prueba oficial.
- **top5**: de cada 100 fotos, en cuantas aparece la especie correcta dentro de las cinco primeras propuestas. El sistema obtiene 67,4 %, mucho mas alto que el top-1 porque acertar "de entre cinco" es mas facil. Es util porque en la demo se muestran cinco candidatas: si la correcta esta en la lista, el usuario la ve.
- **macro_f1**: una nota que promedia el rendimiento de las 102 especies dando el mismo peso a todas, incluidas las que tienen pocas fotos. El sistema obtiene 0,354.
- `zero_division=0` evita que el programa falle cuando una especie no aparece nunca en las predicciones: en ese caso cuenta como cero, no como error.

```python
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=etiquetas, zero_division=0
    )
    per_class = {
        nombres[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(int(n_clases))
    }
    per_class_f1 = {nombres[i]: float(f1[i]) for i in range(int(n_clases))}
```

- **per_class** es el cuadro de rendimiento especie por especie, con un diccionario por especie que contiene `precision`, `recall`, `f1` y `support`.
- **precision** responde a "cuando el modelo dice que es una rosa, cuantas veces tenia razon".
- **recall** responde a "de todas las rosas que habia, cuantas encontro".
- **f1** combina las dos anteriores en un solo numero (0 es pesimo, 1 es perfecto).
- **support** no es una nota, es el numero de fotos de esa especie en el examen. Sirve para saber si la nota se apoya en 40 fotos o en 2.
- **per_class_f1** es lo mismo pero mas simple: solo el numero de f1 por especie, porque el cuaderno de analisis lo consume asi.

### Por qué el macro-F1 puede ser mucho menor que la exactitud

No es un error de cuentas, es consecuencia de como se promedian:

- La exactitud (top-1, 37,8 %) se calcula sumando todos los aciertos y dividiendo entre 6149. Una especie con muchas fotos pesa mucho, y las especies frecuentes suelen ser las que mejor se reconocen. Esa nota sube sola.
- El macro-F1 (0,354) da exactamente el mismo peso a las 102 especies. Una especie rara con dos fotos, en la que el modelo falla, baja la media tanto como si fallara una especie con cien fotos.

Por eso los dos numeros difieren: la exactitud mide "que proporcion del examen apruebo" y el macro-F1 mide "como de bien me porto con todas las especies por igual". La diferencia entre 0,378 y 0,354 es la senal de que las especies con menos ejemplos son las que peor van.

### La matriz de confusion: quien se confunde con quien

La **matriz de confusion** es una tabla cuadrada donde las filas son la respuesta correcta y las columnas lo que respondio el modelo. Cada casilla cuenta cuantas fotos de la fila acabaron en la columna. La diagonal (fila y columna iguales) son los aciertos; todo lo que queda fuera de la diagonal son confusiones. Es la forma de saber, de un vistazo, que el modelo confunde sobre todo petunia con wallflower (58 casos) y rosa con camelia (45 casos).

```python
    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas)
    return top1, top5, macro_f1, per_class, per_class_f1, matriz
```

- Con 102 especies la tabla es de 102 por 102, asi que tambien se genera una version corta y legible con solo las 20 confusiones mas frecuentes.

```python
def _top_confusiones(matriz: np.ndarray, nombres: Sequence[str]) -> list[list]:
    pares: list[list] = []
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if i == j:
                continue
            n = int(matriz[i, j])
            if n > 0:
                pares.append([str(nombres[i]), str(nombres[j]), n])
    pares.sort(key=lambda fila: (-fila[2], fila[0], fila[1]))
    return pares[:MAX_CONFUSIONES]
```

- Se salta la diagonal (`if i == j`) porque ahi estan los aciertos, y aqui solo interesan los errores.
- Cada confusion se guarda como tres datos: especie real, especie predicha y numero de veces.
- Se ordenan de mas a menos casos y se cortan las 20 primeras. En una lista asi aparece arriba petunia hacia wallflower con 58.

### Las figuras y el archivo de predicciones

El programa guarda dos imagenes dentro de `resultados/figuras/`. La primera, `confusion_matrix.png`, dibuja la tabla completa como un mapa de calor: cuanto mas oscura la casilla, mas imagenes. Se usa una escala logaritmica a proposito, porque sin ella la diagonal seria tan dominante que no se veria ninguna confusion.

```python
    figura, eje = plt.subplots(figsize=(13, 11))
    norma = LogNorm(vmin=1, vmax=max(1, maximo)) if maximo > 0 else None
    imagen = eje.imshow(datos, interpolation="nearest", cmap="Blues", norm=norma)
```

- La escala `LogNorm` comprime las diferencias para que los valores pequenos sigan siendo visibles.
- El backend `Agg` se fija al principio del archivo. Significa que las figuras se dibujan directamente en el archivo, sin abrir ninguna ventana: asi el programa funciona igual en un servidor sin pantalla que en un ordenador normal.

La segunda figura, `top_confusions.png`, dibuja barras horizontales con las confusiones mas frecuentes, la mas repetida arriba, para poder citarlas en el informe.

```python
        eje.barh(posiciones, valores, color="#c0392b")
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas, fontsize=7)
        eje.set_xlabel("Numero de imagenes")
        eje.set_title("Confusiones mas frecuentes (real -> predicho)")
```

- Cada barra es una confusion y su etiqueta se lee "especie real -> especie predicha".
- La mas frecuente de todas, petunia confundida con wallflower (58 casos), sale arriba del todo.

Ademas se escribe un **CSV de predicciones** con una fila por cada imagen examinada:

```python
CABECERA_CSV: tuple[str, ...] = ("ruta", "etiqueta_real", "prediccion", "prob", "correcto")
```

```python
            escritor.writerow(
                [
                    str(ruta),
                    str(nombres[real]),
                    str(nombres[predicha]),
                    f"{float(prob):.6f}",
                    bool(real == predicha),
                ]
            )
```

- `ruta` es el archivo de imagen concreto, `etiqueta_real` la respuesta correcta, `prediccion` lo que dijo el modelo, `prob` la confianza de esa prediccion y `correcto` un si o un no.
- Para que sirve: permite auditar caso por caso. Si alguien pregunta por que la rosa se confunde con camelia, se abre el CSV, se filtran las filas donde `etiqueta_real` es rosa y `correcto` es falso, y se miran esas fotos concretas. Sin este archivo solo habria un numero global y no se podria investigar nada.
- El archivo se guarda como `resultados/predicciones/test_predicciones.csv`, con el nombre del grupo examinado delante para no pisar los resultados de `val`.

### Si fallan los procesos auxiliares, el programa se reintenta solo

Para leer las fotos rapido se usan varios **procesos auxiliares** (`num_workers`), que son copias del programa trabajando en paralelo. En Windows esas copias a veces no pueden comunicarse y la evaluacion se cae con un error de permisos, aunque el modelo no tiene ninguna culpa. La solucion es no morir: se avisa y se repite el examen con cero procesos auxiliares, que siempre funciona.

```python
        aviso = (
            f"El DataLoader con num_workers={workers} fallo ({type(exc).__name__}: {exc}). "
            "Se reintenta una sola vez con num_workers=0; esto es habitual en Windows "
            "(los workers de multiprocessing necesitan named pipes). Para evitar el "
            "reintento usa `--num-workers 0`."
        )
        warnings.warn(aviso, RuntimeWarning, stacklevel=2)
        print(f"[evaluate] aviso: {aviso}")
        return _intento(replace(cfg, num_workers=0))
```

- El reintento es **una sola vez** y solo si el error tiene pinta de venir de esos procesos auxiliares; si el fallo es de otra cosa, se deja ver el error de verdad. Esto ultimo es importante: si se ocultaran todos los errores, seria imposible enterarse de un problema real.
- El aviso se emite por dos vias (`warnings` y `print`) para que quede en el registro aunque nadie vea la consola.
- Si el cargador de datos lo habia puesto quien llamo a la funcion (pruebas automaticas), no se reintenta nada: ese fallo es de las pruebas y debe verse entero.
- En la misma idea, `--num-workers` es una opcion de la linea de comandos para pedirlo a mano y evitar el reintento.

```python
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=None,
        help=(
            "workers del DataLoader (por defecto: los de Config, 2). Usa 0 en Windows "
            "o dentro de un sandbox donde los workers no puedan abrir named pipes"
        ),
    )
```

- Si no se indica nada, se respeta el valor normal de la configuracion, que es el correcto en Colab y Linux.

### Como se lanza la evaluacion desde la consola

```python
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evalua un checkpoint de flores sobre el split val o test.",
    )
    parser.add_argument(
        "--checkpoint",
        default="resultados/best.pt",
        help="ruta del checkpoint (por defecto: resultados/best.pt)",
    )
```

- El comando completo es `python -m src.evaluate --checkpoint resultados/best.pt --split test`. `python -m` significa "ejecuta este archivo del proyecto como programa".
- `--split test` examina con el conjunto de prueba; con `--split val` se examina el grupo de validacion.
- `--out-dir` cambia la carpeta donde se escriben las notas y las figuras, `--device` elige si se usa la tarjeta grafica o el procesador, `--debug` usa un subconjunto pequeno de fotos.
- `main` devuelve 0 si todo fue bien y 1 si faltaba el checkpoint o hubo un error de datos. Ese numero es lo que usan los sistemas automaticos para saber si el comando funciono.

## Predecir: dar una respuesta a una foto suelta

`src/predict.py` es el programa que usa el modelo ya entrenado con una imagen concreta. Aqui ya no hay notas ni examenes: hay una pregunta y una respuesta.

### predict_image: que recibe y que devuelve

```python
def predict_image(
    image_path,
    checkpoint_path,
    topk: int = 5,
    min_conf: float = 0.25,
    device: str = "auto",
) -> list[tuple[str, float]]:
```

- **Recibe** `image_path`: la foto. Puede ser la ruta de un archivo, una imagen ya abierta en memoria, un arreglo de numeros que representa pixeles, o directamente lo que entrega la ventana de la demo. Toda esa variedad se admite a proposito, para que quien llama no tenga que preocuparse por el formato.
- **Recibe** `checkpoint_path`: el archivo del modelo entrenado.
- **Recibe** `topk`: cuantas especies quiere de vuelta (5 por defecto).
- **Recibe** `min_conf`: el umbral de confianza, que se explica justo despues.
- **Devuelve** una lista de parejas ordenadas de mayor a menor probabilidad, por ejemplo `[("rosa", 0.31), ("camelia", 0.12), ...]`. Cada pareja es el nombre de la especie y un numero entre 0 y 1 que es su probabilidad. La primera de la lista es la apuesta principal.

```python
    with torch.no_grad():
        logits = modelo(tensor)
    probabilidades = torch.softmax(logits.detach().float(), dim=1).flatten().cpu().numpy()

    n_clases = int(probabilidades.shape[0])
    nombres = _nombres_ajustados(nombres_ckpt, n_clases)
    mejor = float(probabilidades.max()) if n_clases else 0.0

    if mejor < float(min_conf):
        return [(NO_RECONOCIDA, mejor)]

    if n_clases == 0:
        return []

    k = max(1, min(int(topk), n_clases))
    indices = np.argsort(-probabilidades, kind="stable")[:k]
    return [(nombres[int(i)], float(probabilidades[int(i)])) for i in indices]
```

- `probabilidades` es la confianza en cada una de las 102 especies, y suman 1. `mejor` es la mas alta de todas.
- La linea `if mejor < float(min_conf)` es el umbral: si ni siquiera la mejor apuesta llega al minimo exigido, en vez de una lista de especies se responde con la frase `no reconocida / fuera del catalogo`. La idea es decir "esto no parece ninguna de las 102 flores que conozco" en lugar de dar una respuesta poco fiable.
- `NO_RECONOCIDA` es exactamente ese texto, definido una sola vez en todo el proyecto:

```python
NO_RECONOCIDA = "no reconocida / fuera del catálogo"
```

- `k = max(1, min(int(topk), n_clases))` evita pedir mas especies de las que existen y garantiza que al menos se devuelva una.
- `np.argsort` ordena las probabilidades de mayor a menor y devuelve las posiciones; despues esas posiciones se traducen a nombres de especie.
- El tamano de imagen que se usa para preparar la foto se lee del checkpoint, no se da por hecho. Es la misma precaucion que en la evaluacion.

### El umbral min_conf y su problema

El umbral `min_conf` vale 0,25 por defecto: si la mejor probabilidad no llega al 25 %, el resultado es "no reconocida". El razonamiento es sensato (mejor no responder que responder mal), pero en la practica tiene una trampa que se descubrio al probar la demo y que se explica en la seccion 09 de este documento. En resumen: con un modelo poco entrenado, el `softmax` queda muy plano (el maximo ronda el 2 %), y entonces el umbral borra predicciones que podian ser correctas. La conclusion practica es que este umbral hay que elegirlo mirando la confianza real del modelo entrenado, no copiarlo de un ejemplo.

### La cache: no recargar el modelo en cada clic

Un detalle que no se ve pero se nota mucho: el modelo no se vuelve a leer de disco cada vez que se pide una prediccion. Se guarda en memoria con una **cache**.

Una **cache** es una memoria temporal: la primera vez que se necesita algo caro se guarda, y las siguientes veces se reutiliza en lugar de repetir el trabajo.

```python
@functools.lru_cache(maxsize=_CACHE_SIZE)
def _modelo_cacheado(checkpoint_path: str, device: str):
    """Carga (una sola vez por ``checkpoint_path`` + ``device``) el modelo y sus metadatos.

    Returns:
        ``(modelo, nombres_de_clase, img_size, torch.device)``.
    """
    torch_device = resolve_device(device)
    modelo, class_names, meta = load_checkpoint(checkpoint_path, device=torch_device)
    img_size = int(meta.get("img_size") or getattr(modelo, "img_size", 224) or 224)
    return modelo, tuple(str(n) for n in class_names), img_size, torch_device
```

- `lru_cache` significa "guarda los ultimos resultados y reutilizalos". La clave de la cache es la pareja **ruta del checkpoint mas dispositivo**: si se pide el mismo modelo dos veces, se carga una sola vez.
- Por que se cachea: leer un checkpoint de unos 20 MB del disco tarda bastante, y la demo recibe una peticion por cada clic. Sin cache, cada clic volveria a leer el archivo y la interfaz se sentiria lenta y pesada.
- `maxsize=_CACHE_SIZE` (valor 8) pone un tope: como maximo se recuerdan 8 modelos a la vez, para no llenar la memoria del ordenador.
- Consecuencia practica de la cache, que hay que conocer: si se reentrena el modelo y se sobreescribe el mismo archivo, el programa puede seguir usando la version antigua que ya tenia en memoria. Para eso existe `limpiar_cache()`, que vacia esa memoria y obliga a leer de nuevo.

```python
def limpiar_cache() -> None:
    """Vacia la cache de modelos cargados (util en tests y al reentrenar)."""
    _modelo_cacheado.cache_clear()
```

- Si no se vacia, despues de reentrenar se comparan resultados antiguos creyendo que son nuevos. Es la clase de error que hace perder una tarde entera.

### predict_batch: varias fotos de golpe

`predict_batch` aplica `predict_image` a una lista de imagenes. Como el modelo ya esta en la cache, se carga una sola vez aunque la lista tenga cien fotos.

```python
    return [
        predict_image(imagen, checkpoint_path, topk=topk, min_conf=DEFAULT_MIN_CONF, device=device)
        for imagen in image_paths
    ]
```

- Devuelve una lista de resultados, en el mismo orden en que se pidieron las imagenes.
- Aqui no se puede elegir el umbral: se usa siempre el valor por defecto `DEFAULT_MIN_CONF` de 0,25. Por eso algunas de las respuestas pueden ser "no reconocida".

### Y como se manda una prediccion desde la consola

```python
    parser.add_argument("--image", required=True, help="ruta de la imagen a clasificar")
    parser.add_argument(
        "--checkpoint",
        default="resultados/best.pt",
        help="ruta del checkpoint (por defecto: resultados/best.pt)",
    )
    parser.add_argument("--topk", type=int, default=5, help="numero de clases a mostrar (defecto: 5)")
```

```python
    print(f"imagen:     {args.image}")
    print(f"checkpoint: {args.checkpoint}")
    for posicion, (nombre, probabilidad) in enumerate(resultados, start=1):
        print(f"{posicion:2d}. {nombre:<42s} {probabilidad * 100:6.2f}%")
    return 0
```

- El comando tipico es `python -m src.predict --image fotos/mi_flor.jpg --topk 5`.
- `--image` es obligatorio: sin foto no hay nada que predecir.
- `--topk 5` pide cinco candidatas; `--min-conf` cambia el umbral y `--device` elige tarjeta grafica o procesador.
- La salida en pantalla es una lista numerada con el nombre de la especie y el porcentaje, con dos decimales.
- Igual que en la evaluacion, si falta el checkpoint se imprime el error, se recuerda el comando de entrenamiento y se devuelve 1 en lugar de dejar un rastro de error incomprensible.

## Por qué se hizo asi

- **Se evalua sobre el conjunto de prueba y no sobre el de entrenamiento** porque solo asi la nota es creible. Un modelo examinado con lo que ya estudio siempre parece mejor de lo que es.
- **Se guardan cuatro metricas distintas en lugar de una** porque cada una responde a una pregunta diferente: el top-1 dice la proporcion de aciertos directos, el top-5 dice si la respuesta correcta esta entre las candidatas que la demo ensena, el macro-F1 dice como de equilibrado es el rendimiento entre las 102 especies y el cuadro por especie dice donde falla exactamente. Con un solo numero no se podria explicar la diferencia entre 37,8 % de aciertos y 0,354 de macro-F1.
- **Se generan figuras y no solo numeros** porque una tabla de 102 por 102 no la lee nadie. Convertida en mapa de calor y en barras de las 20 confusiones mas frecuentes, la informacion se entiende de un vistazo y se puede citar en el informe con casos concretos.
- **Se guarda un CSV con una fila por imagen** para poder pasar del "el modelo falla un 62 % de las veces" al "estas cuatro fotos concretas de rosa se confundieron con camelia" y mirarlas.
- **El cargador de datos degrada solo a cero procesos auxiliares** porque el problema es del sistema operativo, no del proyecto. Abortar una evaluacion de varios minutos por un fallo de comunicacion entre procesos seria absurdo, pero ocultar cualquier error tambien: por eso se reintenta una sola vez, solo ante senales conocidas de ese fallo, y se avisa.
- **El tamano de imagen y el numero de especies se leen del checkpoint** en los dos archivos, porque es el unico sitio donde consta con que se entreno. Dar por hecho 224 pixels es la clase de suposicion que produce resultados malos sin dar ningun error.
- **El modelo se cachea** porque la demo y el uso interactivo repiten peticiones con el mismo modelo; recargar 20 MB por clic haria la interfaz inusable. El precio es que hay que vaciar la cache con `limpiar_cache()` al reentrenar, y ese precio esta documentado en el propio codigo.
- **Las predicciones devuelven probabilidades, no solo el ganador** porque la confianza es informacion util: permite al usuario decidir si se fia y permite que el resto del sistema aplique umbrales o avisos sin volver a calcular nada.
- **`predict_image` admite muchos formatos de entrada** para que la interfaz grafica no tenga que adaptar nada: lo que entrega la ventana de la demo se acepta tal cual.
- **Los errores se convierten en mensajes en lugar de excepciones sueltas** ("entrena primero: `python -m src.train`") porque cualquiera puede olvidar entrenar antes de predecir, y un texto claro ahorra tiempo a quien no programa.
