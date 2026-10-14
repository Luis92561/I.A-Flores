# Los datos: cómo entran las fotos al sistema

## Para qué sirve este archivo (en lenguaje llano)

Hasta ahora sabemos que el sistema aprende mirando fotos de flores y que las fichas tecnicas del proyecto estan escritas en una libreta de ajustes. Falta la pieza mas terrenal de todas: **de donde salen esas fotos y como se preparan antes de entrar**.

Este capitulo explica `src/data.py`, el archivo que se ocupa de esa puerta de entrada. Su trabajo es de portero y de cocinero a la vez. De portero, porque consigue las 8189 fotos del catalogo publico de la Universidad de Oxford y las reparte en los tres grupos de trabajo: las que sirven para aprender, las que sirven para vigilar el aprendizaje y las que se guardan para la medicion final. De cocinero, porque ninguna foto entra al sistema tal como salio de la camara: hay que cortarla al mismo tamano, ajustarle los colores y, durante el aprendizaje, variarla un poco para que el sistema no se acostumbre a verla siempre igual.

El archivo tiene tres funciones publicas, y son las tres que explican este capitulo. La primera, `build_transforms`, describe las operaciones que se le hacen a cada foto antes de usarla. La segunda, `build_dataloaders`, prepara el mecanismo que va sirviendo las fotos en tandas. La tercera, `get_class_names`, consigue la lista de los 102 nombres de especie para que el sistema pueda responder "petunia" y no "especie numero 53".

Una advertencia sobre las palabras, porque en este capitulo aparecen varios terminos nuevos y conviene no perderse. Llamamos **transformacion** a cada operacion que se aplica a una foto, como recortarla o cambiarle el brillo. Llamamos **lote** al grupo de fotos que se procesa de una vez. Y llamamos **catalogo** o **conjunto de datos** al total de fotos con sus etiquetas, que en ingles se dice *dataset*. La palabra **etiqueta** es simplemente el nombre correcto de la flor en cada foto: es lo que queremos que el sistema aprenda a deducir.

El archivo tiene 449 lineas, pero la mayor parte son funciones auxiliares y comentarios de seguridad. Las tres funciones que importan ocupan unas sesenta lineas en total.

## Qué hay dentro

### El catalogo de flores y sus tres particiones

Antes de mirar el codigo conviene entender que se esta manejando. **Oxford Flowers-102** es un catalogo publico de fotografias de flores publicado por la Universidad de Oxford y usado en investigacion desde hace anos. Contiene exactamente lo siguiente:

| Dato | Valor |
|---|---|
| Fotos en total | 8189 |
| Especies distintas | 102 |
| Fotos de entrenamiento | 1020 |
| Fotos de validacion | 1020 |
| Fotos de prueba | 6149 |

Esas tres particiones, en ingles *splits*, no las elegimos nosotros: vienen definidas en el propio catalogo y estan descritas en un archivo interno llamado `setid.mat`. El propio archivo de datos las declara asi:

```python
#: Splits oficiales, en el orden que espera el contrato.
SPLITS: tuple[str, str, str] = ("train", "val", "test")
```

- `train` es el grupo de **entrenamiento**. Estas 1020 fotos son las que el sistema mira para aprender. Son 10 fotos por especie, que es muy poco, y esa escasez es intencionada en el diseno de este catalogo.
- `val` es el grupo de **validacion**. Estas otras 1020 fotos no se usan para aprender, sino para ir comprobando como va el aprendizaje. Como el sistema nunca aprende de ellas, sirven para saber si de verdad esta mejorando o solo memorizando.
- `test` es el grupo de **prueba**, el mas numeroso con 6149 fotos. Se guarda intacto hasta el final y sirve para la medicion definitiva. Es el examen de verdad, el que nunca se vio antes ni directa ni indirectamente.
- Una **tupla** es una lista corta de valores que van juntos y no se modifican. Aqui guarda los tres nombres en el orden en que el resto del programa los espera.

**Por que se usan las particiones oficiales y no se reparten a mano.** Hay tres motivos y los tres pesan. El primero es la **comparabilidad**: los resultados publicados sobre este catalogo se miden sobre estas mismas particiones, asi que usar otras haria que nuestras cifras no se pudieran comparar con ninguna referencia. El segundo es la **reproducibilidad**: cualquiera puede obtener exactamente la misma particion ejecutando el mismo programa, sin depender de una semilla de reparto ni de nuestras decisiones. El tercero, y el mas importante, es evitar la **fuga de datos**, una expresion que conviene entender bien.

La fuga de datos ocurre cuando una foto usada para aprender aparece tambien en el examen. El sistema la reconoceria de memoria y el resultado seria falsamente bueno: pareceria que acierta mucho, cuando en realidad esta haciendo trampa sin saberlo. Con flores el riesgo es real, porque una misma planta puede haberse fotografiado varias veces con pocos segundos de diferencia: si una foto de esa planta cae en entrenamiento y su gemela en prueba, la medicion queda contaminada. Las particiones oficiales, que fueron construidas con cuidado por los autores del catalogo, reducen ese riesgo. Aqui conviene ser honestos: una comprobacion posterior por contenido encontro cuatro pares de imagenes identicas byte a byte, de los cuales dos cruzan particiones. Son dos imagenes sobre 8189, un efecto despreciable en las cifras finales, pero se documenta porque impide afirmar que la separacion sea absolutamente perfecta.

Que el grupo de prueba sea mas grande que el de entrenamiento sorprende a mucha gente, y es a proposito: en este catalogo la dificultad no esta en tener muchos datos para aprender, sino en aprender con muy pocos y demostrarlo sobre muchos.

### build_transforms: preparar cada foto antes de usarla

**Que es una transformacion.** Una transformacion es una operacion que se le hace a una foto. Recortarla es una transformacion. Saltarla es una transformacion. Cambiarle el brillo es una transformacion. Se encadenan unas detras de otras para formar una **tuberia** o *pipeline*: un recorrido con estaciones, por el que cada foto pasa en orden y del que sale preparada.

**Por que hay que recortar y normalizar.** Por dos razones muy concretas. La primera es que el sistema exige siempre el mismo tamano: la red que usamos espera imagenes de 224 por 224 pixeles, tal como explica la libreta de ajustes, y las fotos del catalogo tienen tamanos muy distintos entre si. La segunda es que la red que usamos aprendio primero con el banco de imagenes ImageNet, donde las fotos estaban preparadas de una forma concreta; para aprovechar lo que ya sabe hay que hablarle en el mismo idioma, y ese idioma incluye el tamano y los colores. **Normalizar** es precisamente eso: dejar los colores en la escala que la red ya conoce, restableciendo la media y la desviacion tipica que tiene grabadas.

La funcion tiene dos caminos, segun se este entrenando o evaluando:

```python
def build_transforms(cfg: Config, train: bool) -> Callable:
    transforms = _transforms()
    img_size = int(cfg.img_size)
    if img_size <= 0:
        raise ValueError(f"cfg.img_size debe ser positivo, se recibio {cfg.img_size!r}")

    ops: list[Any] = []
    if train:
        ops.append(transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)))
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        rand_augment = getattr(transforms, "RandAugment", None)
        if rand_augment is not None:
            ops.append(rand_augment(num_ops=2, magnitude=9))
        else:  # torchvision < 0.12: no hay RandAugment
            ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    else:
        ops.append(transforms.Resize(int(img_size * 1.14)))  # 224 -> 255
        ops.append(transforms.CenterCrop(img_size))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(ops)
```

- `def build_transforms(cfg: Config, train: bool) -> Callable:` declara la funcion. Recibe dos cosas: la libreta de ajustes, de donde lee el tamano de imagen y los valores de color, y un valor de si o no llamado `train` que dice si estamos entrenando. Devuelve un `Callable`, es decir, un objeto que se puede llamar como si fuera una funcion. Ese objeto es la tuberia ya montada y lista para aplicar a cualquier foto.
- `transforms = _transforms()` consigue la caja de herramientas de transformaciones de la libreria TorchVision. El nombre empieza por guion bajo, que es la convencion para las funciones internas del archivo, las que no se anuncian hacia fuera.
- `img_size = int(cfg.img_size)` copia el tamano de imagen de la libreta a una variable local, para no estar consultando la libreta en cada linea. El `int(...)` se asegura de que sea un numero entero.
- El bloque `if img_size <= 0: raise ValueError(...)` es un control de sentido comun: un tamano de imagen de cero o negativo es imposible, asi que la funcion se niega a seguir y explica por que. Detectar un disparate en este punto ahorra horas de busqueda despues. `raise` significa lanzar un error a proposito.
- `ops: list[Any] = []` crea una lista vacia donde se iran guardando las estaciones de la tuberia. Una **lista** es una coleccion ordenada de cosas; una **lista vacia** es una lista sin nada dentro todavia. Se empieza vacia y se van anadiendo estaciones con `ops.append(...)`, que significa "anade esto al final de la lista".
- `if train:` separa los dos caminos. Si estamos entrenando, la tuberia lleva variaciones; si no, lleva un recorrido fijo y predecible. Cada linea de este bloque y del siguiente se explica por separado mas abajo.
- Al final, las dos estaciones comunes se anaden siempre, tanto en entrenamiento como en evaluacion: convertir a numeros y normalizar. Es la parte que no admite variacion, porque es la que hace que la foto sea comprensible para la red.
- `return transforms.Compose(ops)` envuelve la lista en un objeto que la aplica de corrido, estacion tras estacion, en el orden en que se anadieron. `Compose` significa componer, y el orden importa: primero se recorta y se varia, y solo al final se convierte a numeros y se normaliza.

Ahora las estaciones una a una. Las cuatro primeras solo aparecen cuando se entrena.

**`RandomResizedCrop`, recortar al azar.** `transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0))` elige un trozo de la foto al azar y lo estira hasta el tamano pedido. El parametro `scale=(0.6, 1.0)` dice que el trozo elegido tendra entre el 60 y el 100 por ciento del area de la imagen original; si sale un recorte pequeno, el sistema ve la flor de cerca y con menos contexto alrededor, y si sale grande, la ve mas entera. El trozo cambia en cada vuelta, asi que el sistema nunca ve la misma foto exactamente igual dos veces. Esto le ensena a reconocer la flor aunque este descentrada, recortada o mas cerca de lo habitual.

**`RandomHorizontalFlip`, voltear como en un espejo.** `transforms.RandomHorizontalFlip(p=0.5)` voltea la foto de izquierda a derecha en la mitad de los casos, porque `p=0.5` es una probabilidad de una entre dos. Una margarita fotografiada desde el otro lado sigue siendo una margarita, asi que el volteo no cambia la respuesta correcta pero si duplica las formas en que el sistema ve la misma flor. Conviene notar que solo se voltea en horizontal: girarla en vertical seria como mirarla boca abajo, algo que no ocurre en la naturaleza y que le ensenaria cosas falsas.

**`RandAugment`, variaciones automaticas de aspecto.** `rand_augment(num_ops=2, magnitude=9)` aplica dos variaciones al azar de una lista amplia de retoques, entre los que hay cambios de brillo, contraste, color o nitidez. `num_ops=2` significa dos variaciones por foto, y `magnitude=9` regula su intensidad. El nombre viene de *random augmentation*, aumento al azar. Su utilidad es que multiplica artificialmente la variedad del entrenamiento sin necesidad de fotos nuevas: una misma foto, retocada de mil maneras, vale por muchas. En este proyecto importa doblemente, porque solo hay 10 fotos por especie.
- La linea `rand_augment = getattr(transforms, "RandAugment", None)` pregunta si la version instalada de TorchVision trae esta herramienta. `getattr` es una forma defensiva de preguntar "tienes esto?"; si no lo tiene, devuelve `None`, que significa "nada".
- El bloque `if rand_augment is not None` la usa cuando existe. Si no existe, entra el `else` y se aplica `ColorJitter`, una version mas modesta que solo retoca brillo, contraste, saturacion y tono. Es una red de seguridad para que el proyecto funcione tambien con versiones antiguas de la libreria, en lugar de romperse: el sistema pierde algo de variedad, pero sigue funcionando.

En evaluacion y en prueba, en cambio, la tuberia es fija:

**`Resize` y `CenterCrop`, el recorrido determinista.** `ops.append(transforms.Resize(int(img_size * 1.14)))` agranda la foto un 14 por ciento, de modo que 224 pasa a ser 255, y despues `transforms.CenterCrop(img_size)` recorta el cuadrado central de 224. La combinacion equivale a un leve acercamiento al centro. Se hace asi, y no recortando directamente, para imitar la preparacion que se uso al entrenar la red original: primero se agranda un poco y luego se toma el centro.
- La palabra **determinista** es clave en este camino. Significa que la misma foto produce siempre exactamente el mismo resultado. Es obligatorio al medir: si cada evaluacion recortara la foto de una forma distinta, dos mediciones del mismo modelo darian cifras diferentes y no se sabria si la diferencia viene del modelo o del recorte.

**`ToTensor`, convertir la foto en numeros.** `transforms.ToTensor()` transforma la imagen en un **tensor**. Un tensor es la estructura de datos que usa PyTorch: una tabla de numeros con varias dimensiones. En este caso pasa de una imagen a una tabla de tres capas (rojo, verde y azul) de 224 por 224 numeros cada una. Ademas, la conversion cambia la escala: los pixeles, que venian como numeros enteros de 0 a 255, pasan a ser numeros decimales entre 0 y 1. Es el paso sin el cual la red no puede hacer nada, porque la red solo sabe calcular con numeros.

**`Normalize`, dejar los colores en el idioma de la red.** `transforms.Normalize(MEAN, STD)` resta la media y divide por la desviacion tipica, usando los tres valores de color que estan guardados en la libreta de ajustes. El resultado es que la mayoria de los valores quedan alrededor de cero. Suena a complicacion innecesaria, y tiene una explicacion clara: la red que usamos como base fue entrenada con el banco de imagenes ImageNet usando exactamente esta correccion. Si le entregamos las fotos sin corregir, los numeros le llegan en una escala distinta de la que aprendio a interpretar y su rendimiento baja. Normalizar es, literalmente, hablarle en el mismo idioma.

**Por que en validacion y prueba no se aumenta nada.** Porque ahi no queremos que el sistema aprenda, sino que queremos medirlo con honestidad. Si durante la evaluacion recortaramos y voltearamos las fotos al azar, cada medicion seria distinta y ademas estariamos midiendo sobre versiones deformadas de las fotos, no sobre las fotos reales. La aumentacion es una herramienta de entrenamiento y solo se usa entrenando. Esta regla esta escrita en el contrato del proyecto y se cumple en este archivo.

> Nota: todas las estaciones de variacion son de la libreria TorchVision, que las aplica al azar en cada lectura de cada foto. Por eso el mismo catalogo produce, en la practica, una cantidad casi ilimitada de versiones distintas de las mismas 1020 fotos de entrenamiento.

### build_dataloaders: servir las fotos en tandas

**Que es un DataLoader.** Un *DataLoader* es el mecanismo que sirve las fotos al sistema. Su nombre se traduce como cargador de datos y su tarea es triple: coger el catalogo, ir tomando las fotos de una en una, aplicarles su tuberia de transformaciones y entregarlas agrupadas en lotes. Sin el, el programa de entrenamiento tendria que ocuparse a mano de abrir archivos, ordenarlos y agruparlos.

**Que son el lote y el barajado.** El **lote** es el grupo de fotos que se entrega de una vez; su tamano lo fija la libreta de ajustes y vale 32. Como ya se explico en el capitulo de la configuracion, el sistema no corrige su criterio despues de cada foto, sino despues de cada lote. El **barajado** es desordenar las fotos antes de entregarlas. Si el catalogo se recorriera siempre en el mismo orden, el sistema veria durante un rato solo especies parecidas y luego solo otras, y ese orden acabaria influyendo en lo que aprende. Desordenar en cada vuelta rompe ese patron.

```python
def build_dataloaders(cfg: Config) -> tuple["DataLoader", "DataLoader", "DataLoader", list[str]]:
    from torch.utils.data import DataLoader

    Flowers102 = _flowers102_class()
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)

    # pin_memory solo cuando hay CUDA (en CPU no aporta y consume RAM).
    pin_memory = bool(_torch().cuda.is_available())
    num_workers = int(cfg.num_workers)

    datasets: dict[str, Any] = {}
    loaders: dict[str, Any] = {}
    for split in SPLITS:
        dataset = Flowers102(
            root=str(root),
            split=split,
            transform=build_transforms(cfg, train=(split == "train")),
            download=True,
        )
        datasets[split] = dataset
        if cfg.debug:
            dataset = _debug_subset(dataset, per_class=cfg.debug_per_class, seed=int(cfg.seed) + 17 * SPLITS.index(split))
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=(split == "train"),  # val/test SIN barajar (contrato §3)
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    class_names = _resolve_class_names(cfg, dataset=datasets["train"])
    return loaders["train"], loaders["val"], loaders["test"], class_names
```

- `Flowers102 = _flowers102_class()` consigue la clase que sabe leer el catalogo de Oxford. Entrega el catalogo ya organizado segun las particiones oficiales, de modo que no hay que repartir nada a mano.
- `root = Path(cfg.data_root)` y `root.mkdir(parents=True, exist_ok=True)` preparan la carpeta donde se guardaran las fotos. El detalle importante es que la carpeta no existe de antemano: el programa la crea la primera vez.
- `pin_memory = bool(_torch().cuda.is_available())` decide una optimizacion llamada **memoria fijada**. Consiste en dejar las fotos en una zona de memoria preparada para que el envio a la tarjeta grafica sea mas rapido. Se activa solo si hay tarjeta, por el motivo que se explica en la seccion de decisiones: sin tarjeta no aporta nada y consume memoria del ordenador. La palabra `bool` convierte la respuesta en un si o no limpio.
- `num_workers = int(cfg.num_workers)` lee cuantos procesos ayudantes se usaran para preparar las fotos. El valor normal es dos, y cero en modo de prueba.
- `datasets` y `loaders` son dos **diccionarios**, es decir, colecciones de parejas "nombre: valor". El primero guardara los tres catalogos ya preparados, y el segundo los tres cargadores. Se usan diccionarios para poder consultarlos por el nombre de la particion, por ejemplo `loaders["train"]`, en lugar de por su posicion en una lista.
- `for split in SPLITS:` recorre las tres particiones, `train`, `val` y `test`, y hace lo mismo con cada una. Un **bucle** es una instruccion que repite un bloque de codigo una vez por cada elemento de una lista.
- `Flowers102(root=..., split=split, transform=..., download=True)` crea el catalogo de esa particion. De los cuatro parametros, dos merecen atencion. `transform=build_transforms(cfg, train=(split == "train"))` es el que conecta este cargador con la funcion anterior: la comparacion `(split == "train")` da si solo cuando la particion es la de entrenamiento, de manera que la tuberia con variaciones se aplica unicamente ahi y las otras dos usan el recorrido fijo. Y `download=True` significa que, si las fotos no estan todavia en la carpeta, se descarguen automaticamente.
- `datasets[split] = dataset` guarda el catalogo recien creado bajo su nombre en el diccionario, para poder recuperarlo al final.
- El bloque `if cfg.debug:` sustituye el catalogo completo por una muestra reducida, segun se explica en su propia seccion mas abajo.
- `DataLoader(dataset, ...)` construye el cargador. Ahi estan los tres ajustes que definen su comportamiento: `batch_size=int(cfg.batch_size)` fija el lote en 32 fotos; `shuffle=(split == "train")` activa el barajado solo en entrenamiento, de modo que las fotos de validacion y de prueba se sirven siempre en el mismo orden, para que las mediciones sean comparables entre si; y `drop_last=False` ordena no descartar el ultimo lote aunque quede incompleto. Este ultimo detalle importa mas de lo que parece: con `num_workers` a cero, cada foto se lee y se transforma en el momento de usarla, sin ayuda de nadie. Es trabajo lento, pero es exactamente lo que se quiere en una prueba de un minuto y lo que evita los problemas descritos en la siguiente seccion.

**Los tres cargadores que devuelve.** El `return` final entrega cuatro cosas: el cargador de entrenamiento, el de validacion, el de prueba y la lista de nombres de las 102 especies. Los tres cargadores se comportan igual salvo en dos detalles, y los dos son deliberados: el de entrenamiento baraja las fotos y las varia; los otros dos no hacen ninguna de las dos cosas, porque su mision es medir y no ensenar. El orden del retorno esta fijado en el contrato del proyecto y todos los demas programas lo dan por hecho.

**`num_workers` y por que en Windows puede fallar.** Los **procesos ayudantes** son copias reducidas del programa que se lanzan para ir preparando fotos mientras el proceso principal calcula. Preparar una foto es trabajo pesado: hay que abrir el archivo, recortarlo, variarlo y convertirlo. Repartiendo esa tarea entre dos ayudantes, el sistema deja de esperar. En Colab y en Linux funciona sin complicaciones.

En Windows, en cambio, puede fallar, y conviene saber por que. Windows no crea procesos nuevos de la misma forma que Linux: en lugar de duplicar el programa en marcha, arranca una copia nueva del programa y le pide que se reconstruya. En ese arranque hay codigo que se ejecuta de nuevo, y a veces los ayudantes se quedan esperando al proceso principal mientras el proceso principal espera a los ayudantes. El resultado es un bloqueo o un error de permisos que desconcierta, porque el mismo programa funciona perfectamente en Colab. Por eso la libreta de ajustes usa dos ayudantes por defecto, que es lo correcto donde el proyecto se ejecuta de verdad, y por eso existe la posibilidad de bajarlo a cero para trabajar en Windows. El proyecto, ademas, tiene previsto que el programa de evaluacion reintente una vez con cero ayudantes si tropieza con este problema, avisando en el registro, en lugar de morir con un error incomprensible.

### get_class_names: los nombres de las 102 especies

El sistema trabaja internamente con numeros: dice "especie 53", no "petunia". La ultima funcion del archivo se encarga de traducir esos numeros a nombres que una persona reconozca.

La dificultad es que ese nombre no esta garantizado en ningun sitio. La libreria TorchVision no siempre lo trae, el archivo oficial que lo contiene ha dejado de estar disponible en la direccion donde estaba, y el programa no puede permitirse el lujo de fallar por eso. La solucion es una **cadena de respaldos**: se intenta por un camino y, si no da resultado, se pasa al siguiente, hasta que alguno funcione. El ultimo recurso nunca falla porque son nombres inventados.

```python
def get_class_names(cfg: Config) -> list[str]:
    num_classes = int(cfg.num_classes)

    if dataset is not None:
        names = _names_from_dataset(dataset, num_classes)
        if names:
            return names
    else:
        try:  # 1a) `Flowers102.classes`: atributo de CLASE, sin instanciar ni red
            Flowers102 = _flowers102_class()
        except Exception:
            Flowers102 = None
        names = _names_from_dataset(Flowers102, num_classes)
        if names:
            return names

        try:  # 1b) instancia real, por si esa version expone los nombres de otra forma
            opened = (
                Flowers102(root=str(cfg.data_root), split="train", download=False) if Flowers102 is not None else None
            )
        except Exception:
            opened = None
        names = _names_from_dataset(opened, num_classes)
        if names:
            return names

    names = _names_from_cache(cfg)  # 2) cache local (offline-friendly)
    if names:
        return names

    names = _names_from_asset(num_classes)  # 3) respaldo offline versionado en el repo
    if names:
        return names

    names = _download_cat_to_name(cfg)  # 4) descarga de Oxford + cache
    if names:
        return names

    return _fallback_names(num_classes)  # 5) nunca fallar
```

Los cinco caminos, en el orden en que se intentan:

1. **Los nombres que trae la propia libreria.** `Flowers102.classes` es un apartado que en algunas versiones de TorchVision contiene la lista de especies. Es el camino preferido porque funciona sin internet y sin haber descargado nada. La anotacion del codigo lo llama **atributo de clase**, que significa que se puede consultar sobre el molde sin haber fabricado ningun objeto. Si esa lista existe pero contiene nombres genericos del estilo `class_7`, se descarta, porque no aporta nada a una persona.
2. **El archivo guardado en la carpeta de datos.** La primera vez que la descarga funciono, el archivo quedo copiado en la carpeta de datos del proyecto. Las siguientes veces se lee de ahi, sin usar la red.
3. **El respaldo que viaja con el proyecto.** En la carpeta `assets` hay una copia del archivo de nombres, guardada dentro del propio proyecto. Esta es la via mas fiable de todas: no depende de internet ni de la version de la libreria, y ha sido revisada. El propio codigo la prefiere a la descarga y lo deja escrito en un comentario.
4. **La descarga desde Oxford.** Esta es la direccion oficial del archivo de nombres, y hoy dia ya no responde. La peticion se hace con un limite de tiempo, y si falla, el programa lo apunta y sigue adelante sin protestar. Una vez intentado, no se vuelve a intentar en la misma corrida, para no perder tiempo esperando una respuesta que no va a llegar.
5. **Los nombres genericos.** El ultimo recurso fabrica nombres del tipo `especie_001`, `especie_002` y asi hasta `especie_102`. El codigo lo comenta con dos palabras que resumen su filosofia: "nunca fallar". Son nombres pobres pero honestos: el sistema sigue funcionando y avisa de que las especies no tienen nombre real, en lugar de interrumpir el trabajo.

**El detalle de las etiquetas que empiezan en 1.** Aqui esta la trampa mas fina de todo el archivo, y merece explicacion despacio porque es la clase de detalle que arruina resultados sin dar ningun error.

El catalogo original de Oxford numera sus 102 especies del 1 al 102. La libreria TorchVision, en cambio, tiene la costumbre de empezar a contar en cero, asi que a cada etiqueta le resta uno: donde Oxford dice 1, TorchVision dice 0, y donde Oxford dice 102, TorchVision dice 101. Como consecuencia, cuando el archivo de nombres asocia el nombre a la etiqueta interior 0, tiene que buscarlo en el archivo de nombres bajo la clave 1.

```python
    for offset in (1, 0):  # primero 1-based (contrato), luego 0-based (asset)
        names: list[str] = []
        for i in range(int(num_classes)):
            value = mapping.get(str(i + offset))
```

- `for offset in (1, 0)` prueba primero con el desplazamiento de uno, que corresponde al archivo oficial de Oxford, y despues con el de cero, que corresponde a la copia guardada en el proyecto. Asi el mismo codigo sirve para las dos convenciones.
- `str(i + offset)` construye la clave que hay que buscar. Para la etiqueta interior 0 y desplazamiento 1, busca la clave `"1"`. El `str(...)` convierte el numero en texto porque las claves del archivo son texto.
- `mapping.get(...)` busca en el diccionario de nombres. Si no encuentra el nombre esperado, el bucle se corta con `break` y se prueba el desplazamiento siguiente. Si logra completar las 102 especies, se queda con ese resultado.
- La comprobacion adicional de que unas pocas muestras reales coincidan con los nombres, que aparece en las funciones auxiliares, es una defensa contra esta misma trampa: si una version de la libreria cambiara la convencion, el programa lo detectaria y volveria a contar a mano en lugar de devolver nombres desplazados una posicion, que seria un desastre silencioso: todas las respuestas equivocadas por un puesto, sin ningun error de por medio.

**Los nombres genericos, reconocidos por su forma.** Para descartar una lista que no sirve, el archivo usa un patron de texto que reconoce los nombres vacios de contenido.

```python
_GENERIC_NAME_RE = re.compile(
    r"^(?:class|clase|label|etiqueta|category|categoria|species|especie|"
    r"n|img|image|idx|index)[_\- ]?\d+$",
    re.IGNORECASE,
)
```

- Una **expresion regular** es un patron que describe como tiene que ser un texto para coincidir con el. Aqui describe nombres compuestos por una palabra de relleno, un separador opcional y un numero: `class_7`, `n12`, `label 3`.
- `re.compile` prepara ese patron una sola vez, al cargar el archivo, para no reconstruirlo en cada comprobacion.
- La parte entre parentesis es la lista de palabras de relleno, en ingles y en espanol. La parte `[_\- ]?` admite un guion bajo, un guion o un espacio, y el signo de interrogacion significa que puede no haber nada. La parte final `\d+$` exige uno o mas digitos hasta el final del texto.
- `re.IGNORECASE` pide ignorar las mayusculas y minusculas, de modo que `Class_7` tambien se reconoce como nombre generico.
- Gracias a esta comprobacion, el programa distingue "un nombre de verdad" de "un relleno con pinta de nombre". Es la diferencia entre responder "petunia" y responder "clase 53", y ese detalle es el que hace util la funcion.

### El modo debug: 408 fotos en lugar de 8189

En el capitulo anterior se explico que `debug` reduce el trabajo para poder comprobar en un minuto que todo funciona. Aqui esta la parte que lo hace realidad, y el numero concreto: en modo de prueba no se usan 8189 fotos, sino **408**, que son **4 por cada una de las 102 especies**.

```python
    rng = random.Random(int(seed))
    selected: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.extend(sorted(indices[:per_class]))
    selected.sort()

    if not selected:  # dataset vacio: no fallar, devolver un Subset vacio
        return Subset(dataset, [])
    return Subset(dataset, selected)
```

- La funcion auxiliar que hace esto recibe el catalogo completo, el numero de fotos por especie y una semilla de azar. Devuelve un **subconjunto**, es decir, una vista sobre el catalogo original que solo deja ver las fotos elegidas. La palabra inglesa para esto es *Subset*. Es importante entender que no se copian fotos ni se reorganiza nada: se elige que fotos se miran.
- `rng = random.Random(int(seed))` crea un generador de azar propio, con su semilla. Se usa uno propio y no el general del programa para que la eleccion de fotos sea estable y no dependa de cuantas veces se haya usado el azar antes. La semilla sera la misma en cualquier ordenador, asi que las 408 fotos elegidas son siempre las mismas.
- El bucle recorre las especies una por una, en orden. Para cada una, desordena las fotos disponibles con `rng.shuffle(indices)` y se queda con las primeras `per_class`, que son cuatro.
- `selected.extend(...)` va acumulando los indices elegidos, y el `selected.sort()` final los pone en orden creciente. Ordenarlos es una precaucion: el resultado es siempre el mismo, independientemente del orden en que se recorrieran las especies.
- La proteccion `if not selected:` cubre el caso de un catalogo vacio y devuelve un subconjunto tambien vacio. Es coherente con la filosofia de "nunca fallar" que ya vimos en los nombres de especie.
- El detalle tecnico mas importante esta en la documentacion de la funcion: como solo se eligen indices del catalogo original, la correspondencia entre cada foto y su etiqueta se mantiene intacta. Esto parece obvio y no lo es: si el proceso de recorte mezclara las etiquetas, el modo de prueba estaria midiendo un sistema entrenado con respuestas falsas, y podria dar la impresion de que el proyecto esta roto cuando en realidad todo funciona.

Cada particion recibe su propia seleccion, calculada con una semilla ligeramente distinta, para que las fotos de entrenamiento y las de validacion no acaben siendo las mismas:

```python
            dataset = _debug_subset(dataset, per_class=cfg.debug_per_class, seed=int(cfg.seed) + 17 * SPLITS.index(split))
```

- `SPLITS.index(split)` devuelve la posicion de la particion en la lista: cero para entrenamiento, uno para validacion y dos para prueba.
- Multiplicar esa posicion por 17 y sumarla a la semilla base hace que cada particion use una semilla distinta. El numero 17 es arbitrario; lo unico que importa es que las tres semillas sean diferentes entre si.

**Para que sirve.** El modo de prueba permite comprobar en un minuto que toda la maquinaria gira: que las fotos se descargan, que las transformaciones no fallan, que el sistema aprende algo, que se guardan los resultados y que las mediciones se escriben. Con 408 fotos se cubren las 102 especies, de modo que ninguna queda fuera y el sistema no puede acertar por eliminacion. Lo que este modo no hace es dar un resultado presentable: sirve para comprobar que el mecanismo funciona, no para saber lo bueno que es.

## Por qué se hizo asi

**Por que los splits oficiales y no un reparto propio.** Repartir a mano las fotos tiene una ventaja aparente: se podria decidir que hubiera mas fotos de entrenamiento. Tiene tres inconvenientes, y pesan mas. El primero es que las cifras dejarian de ser comparables con las de cualquier trabajo publicado sobre este catalogo. El segundo es que habria que fijar una semilla de reparto y documentarla, con lo que la reproducibilidad dependeria de un detalle nuestro. El tercero, y decisivo, es el riesgo de fuga de datos: en un catalogo donde la misma planta aparece en varias fotos, un reparto descuidado pondria fotos de la misma planta a los dos lados del examen y las cifras saldrian infladas. Las particiones oficiales fueron construidas por los autores del catalogo con ese cuidado, asi que se respetan tal cual.

**Por que la aumentacion solo en entrenamiento.** Porque la aumentacion es una herramienta para ensenar, no para medir. Variar las fotos mientras se aprende hace que el sistema sea mas robusto, porque le obliga a reconocer la flor en condiciones distintas. Aplicar esa misma variacion al medir seria un error por partida doble: cada medicion daria un resultado ligeramente distinto segun lo que tocara al azar, y ademas se estaria midiendo sobre fotos deformadas, no sobre las fotos reales que el sistema vera cuando se use. Por eso la particion de entrenamiento tiene su propia tuberia con variaciones, y las de validacion y prueba usan un recorrido fijo y predecible.

**Por que los nombres de clase viven en un archivo propio del proyecto.** Al principio se intentaba descargar el archivo de nombres desde la direccion oficial de Oxford cada vez que hiciera falta. Cuando esa direccion dejo de responder, quedo claro que un dato tan basico no podia depender de un servidor ajeno. Guardar una copia revisada dentro del proyecto convierte ese dato en algo que viaja con el codigo: funciona sin internet, funciona en cualquier ordenador y no cambia bajo los pies. La descarga se conserva como ultimo intento de conseguir nombres reales, pero ya no es el camino principal. Detras queda todavia la red de seguridad de los nombres genericos, para que ni siquiera en el peor escenario el programa se detenga.

**Por que `pin_memory` se activa solo con tarjeta grafica.** La memoria fijada es una preparacion especial de la memoria del ordenador que acelera el envio de datos a la tarjeta grafica. Si no hay tarjeta, ese envio no existe, de modo que la preparacion no aporta ninguna velocidad y en cambio reserva memoria del ordenador que hace falta para otras cosas. Como las dos situaciones son la norma segun el entorno donde se ejecute, la decision se toma consultando si hay tarjeta disponible. Es el mismo criterio que sigue el proyecto en todo lo relacionado con el motor de calculo: adaptarse al entorno en lugar de dar por hecho cual es.

**Por que no hay rutas absolutas ni ordenes de tarjeta grafica en este archivo.** Dos reglas del contrato del proyecto se cumplen aqui de forma visible. Ninguna ruta empieza por la letra de una unidad de disco, porque todas se construyen a partir de la carpeta indicada en la libreta de ajustes; asi el proyecto funciona igual en Windows, en Linux y en Colab. Y ninguna linea ordena usar la tarjeta grafica, porque este archivo ni siquiera necesita saber donde se calcula: su tarea es servir los datos. La unica mencion a la tarjeta es la consulta de disponibilidad que decide la memoria fijada.

**Por que las importaciones de la libreria de inteligencia artificial son perezosas.** Cargar PyTorch y TorchVision ocupa tiempo y memoria. Este archivo lo hace dentro de las funciones, cuando de verdad hacen falta, y no en la primera linea. La consecuencia practica es valiosa: un programa que solo quiera preparar una imagen para la prediccion puede reutilizar la funcion de transformaciones sin cargar el catalogo de 8189 fotos ni la libreria que lee sus archivos internos. Ademas, cuando falta una pieza auxiliar llamada scipy, que sirve para leer los archivos con las etiquetas del catalogo, el programa no se limita a fallar: lo detecta y explica el comando exacto que hay que ejecutar para instalarla. Un mensaje util ahorra media hora de desconcierto.

**Por que se comprueba la forma de los nombres en lugar de confiar en ellos.** Porque un nombre generico que parece bueno es peor que no tener nombre. Si el sistema respondiera "clase 53" con toda naturalidad, nadie notaria que la traduccion fallo; en cambio, respondiendo "especie_053" queda claro de un vistazo que los nombres reales no se consiguieron. La comprobacion del patron de texto es la que permite distinguir una cosa de la otra antes de dar la lista por buena.

**Por que una prueba rapida usa 408 fotos y no 8189.** Porque el objetivo de esa prueba es ahorrar tiempo, y 8189 fotos exigen descargar cientos de megabytes y esperar. Con 408 fotos, cuatro por especie, la prueba recorre las 102 especies y comprueba toda la maquinaria en un minuto. Mantener la proporcion de cuatro por especie, en lugar de tomar las primeras 408 fotos del catalogo, es lo que evita que la prueba se centre en unas pocas especies y deje el resto sin comprobar: si se tomaran las primeras fotos de la lista, casi todas serian de las mismas especies y la prueba diria muy poco. Y para cerrar la idea con las cifras del proyecto: con solo cuatro epocas en la corrida completa, el sistema acierta el 37,8 por ciento de las 6149 fotos oficiales de prueba. Ese es el orden de magnitud de la prueba corta, y subraya que su papel es comprobar que todo funciona, no presumir de resultados.
