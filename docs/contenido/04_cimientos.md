# Los cimientos: configuración y utilidades

## Para qué sirve este archivo (en lenguaje llano)

Imagina que vas a montar una cocina. Antes de cocinar nada necesitas dos cosas: un lugar donde dejar escritas las cantidades de cada receta (cuanta harina, cuantos grados de horno, cuanto tiempo) y un cajon con las herramientas basicas que usaras en todas las recetas (el reloj, el termometro, el bote donde guardas las etiquetas).

Este capitulo explica esos dos cimientos del proyecto, que son dos archivos pequenos:

- `src/config.py` es la **libreta de ajustes**. No cocina nada: solo guarda los numeros y las rutas que usan todos los demas programas. Aqui estan escritas cosas como cuantas veces va a repasar el sistema las fotos, de cuantas en cuantas las va a mirar, o donde va a dejar el resultado.
- `src/utils.py` es el **cajon de herramientas**. Contiene siete funciones cortas y muy usadas: fijar el azar para que todo salga igual cada vez, elegir si el trabajo se hace con el procesador normal o con la tarjeta grafica, guardar y leer el resultado en un formato de texto sencillo, contar cuantas piezas ha aprendido el modelo y ponerle un nombre legible al motor que se esta usando.

Son los dos primeros archivos que se escribieron en el proyecto, y se escribieron antes que los demas a proposito. La razon es sencilla: si cada programa guardara sus propios ajustes, el dia que quisieras cambiar "25 repasos" por "10 repasos" tendrias que buscar ese numero por todo el proyecto. Con la libreta central solo cambias un sitio. Y como los demas archivos se apoyan en estas herramientas, sus nombres y su forma de usarlas quedaron fijados desde el primer dia y nadie los ha cambiado despues.

## Qué hay dentro

Los dos archivos son cortos y faciles de recorrer de arriba abajo. `src/config.py` tiene 105 lineas y `src/utils.py` tiene otras 105. En el primero encontraras cuatro constantes numericas, una clase con veinticinco ajustes, siete atajos para construir rutas de archivos y tres funciones pequenas. En el segundo, las siete herramientas que acabamos de mencionar.

Conviene aclarar ya tres palabras que van a aparecer todo el rato, porque el resto del capitulo se apoya en ellas:

- Una **variable** es una caja con nombre donde se guarda un valor. Si escribes `epochs = 25`, acabas de crear una caja llamada `epochs` que contiene el numero 25. Cuando el programa necesita saber cuantos repasos hacer, mira dentro de esa caja en lugar de repetir el numero por todas partes.
- Una **clase** es un molde para fabricar objetos. El molde dice que todo objeto fabricado con el tendra los mismos apartados, aunque cada objeto puede tener valores distintos dentro de cada apartado. Un molde de "ficha de alumno" produciria fichas con nombre, apellidos y nota; cada ficha rellena esos tres apartados con datos diferentes.
- Una **funcion** es una tarea con nombre y con instrucciones dentro. En lugar de repetir esas instrucciones cada vez que hacen falta, se escribe la funcion una vez y luego se la llama por su nombre. Por ejemplo, una funcion llamada `hacer_cafe` contendra los pasos de moler, calentar agua y colar; cada vez que la llamas, ocurren esos pasos en orden.

Un cuarto termino, **importar**, aparece en las primeras lineas de los dos archivos. Importar significa pedir prestado: cuando el archivo escribe `import json`, esta diciendo "voy a usar la herramienta llamada json que ya viene con Python", y no hay que escribirla de nuevo.

### Las constantes compartidas

Este es el primer bloque real de `src/config.py`, el que va justo despues de la linea `from dataclasses import asdict, dataclass`.

```python
# ---------------------------------------------------------------- constantes
IMAGE_SIZE: int = 224
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet
FLOWERS102_CLASSES: int = 102

@dataclass
class Config:
    """Hiperparametros y rutas de una corrida de entrenamiento/evaluacion."""
```

- Una **constante** es una variable que se escribe una sola vez y no se cambia durante el trabajo. Por convenio, en Python se escriben con mayusculas para que cualquiera las reconozca de un vistazo.
- `IMAGE_SIZE = 224` es el tamano de imagen que se usa en todo el proyecto: 224 pixeles de ancho por 224 de alto. Un **pixel** es el puntito de color mas pequeno del que se compone una foto; una imagen de 224 por 224 esta formada por 50176 puntitos.
- `MEAN` y `STD` son tres numeros cada uno. Significan media y desviacion tipica, y sirven para "normalizar" el color de las fotos, algo que explicamos con detalle en el capitulo de los datos. El comentario `# ImageNet` avisa de donde salen: del gran banco de imagenes ImageNet, porque la red que usamos se entreno primero con esas fotos y hay que hablarle en el mismo idioma de colores.
- `FLOWERS102_CLASSES = 102` dice cuantas especies distintas hay en el catalogo. En lugar de escribir el numero 102 repartido por el codigo, se escribe una vez aqui; el nombre de la constante deja claro de que 102 estamos hablando.
- La anotacion `: int` y `: float` no es un adorno: es una **anotacion de tipo**, una nota que dice "aqui va un numero entero" o "aqui va un numero con decimales". Python no obliga a cumplirla, pero sirve como documentacion para quien lee y como aviso automatico si algun editor de codigo la revisa.
- `tuple[float, float, float]` significa que el valor es una tupla de tres numeros decimales. Una **tupla** es una lista corta de valores que va junta y no se modifica; el orden importa, porque los tres numeros son la media del rojo, del verde y del azul, en ese orden.

### La clase de configuracion Config

Una clase de configuracion es, en palabras llanas, **una libreta con apartados fijos**. El molde se llama `Config`, y cada vez que fabricas una libreta con ese molde, la libreta sale con los mismos apartados: uno para el numero de repasos, otro para el tamano de imagen, otro para la carpeta de salida, y asi hasta veinticinco. Lo unico que puede cambiar entre una libreta y otra son los valores escritos dentro de los apartados.

Esto es util porque el proyecto entero habla de "la configuracion" y siempre se refiere a ese objeto. El programa que entrena recibe una libreta, mira dentro y actua; el que evalua recibe una libreta, mira dentro y actua. Ninguno de los dos necesita saber como se llama la carpeta de salida, porque lo lee de la libreta.

Cada apartado de la libreta es un **atributo**: la palabra tecnica para designar un apartado con nombre dentro de un objeto. En total esta clase declara veinticinco.

La pieza que convierte esto en comodidad tiene nombre propio y conviene reconocerla:

```python
from dataclasses import asdict, dataclass
```

```python
@dataclass
class Config:
```

- `@dataclass` es un **decorador**: una etiqueta que se pone encima de una clase para anadirle comportamiento sin escribir codigo. La arroba es lo que la marca como decorador. En este caso, decirle `@dataclass` a Python significa "genera por mi todo el trabajo repetitivo de esta clase": el que fabrica las libretas, el que guarda cada valor en su apartado y el que permite comparar dos libretas entre si.
- Ese trabajo repetitivo es el que, sin el decorador, obligaria a escribir a mano una funcion por cada apartado y a copiar veinticinco lineas de asignaciones. Con `@dataclass` basta con declarar los apartados y su valor de partida, que es exactamente lo que ves en el archivo.
- `asdict` se importa en la misma linea porque se usa al final del archivo, en el metodo `to_dict`, para convertir la libreta en un diccionario. Un **diccionario** es una coleccion de parejas "etiqueta: valor"; por ejemplo `{"epochs": 25, "batch_size": 32}`. Es el formato ideal para guardar los ajustes en un archivo de texto y poder revisar despues con que numeros se hizo una corrida.
- Los **valores por defecto** son los valores de partida, los que trae la libreta recien fabricada. Si alguien escribe `Config()` sin mas, obtiene una libreta con 25 repasos, lotes de 32 imagenes y todo lo demas tal como esta escrito en el archivo. Solo hay que escribir un valor a mano cuando se quiere algo distinto de lo normal, por ejemplo `Config(epochs=4)` para una corrida corta de comprobacion.
- El metodo `to_dict()` es la costumbre de poder volcar la libreta entera de golpe a un diccionario, para guardarla en un archivo. Aqui es una sola linea: `return asdict(self)`. La palabra `self` es la forma que tiene una clase de referirse a si misma, es decir, a la libreta concreta que se esta usando en ese momento.

### Los parametros reales, uno a uno

Estos son los veinticinco apartados tal como estan escritos en `src/config.py`, con su valor por defecto y su significado en lenguaje corriente. Un **parametro** es cada uno de esos ajustes que se puede tocar para cambiar el comportamiento del sistema.

| Parametro | Valor por defecto | Que significa |
|---|---|---|
| `data_root` | `"data"` | Carpeta donde se guardan y se buscan las fotos. |
| `out_dir` | `"resultados"` | Carpeta donde se dejan los resultados del trabajo. |
| `num_classes` | `102` | Cuantas especies distintas puede llegar a distinguir. |
| `img_size` | `224` | Lado de la imagen en pixeles, ya recortada y preparada. |
| `backbone` | `"efficientnet_b0"` | Nombre de la red preentrenada que sirve de base. |
| `dropout` | `0.3` | Porcentaje de conexiones que se apagan al azar durante el entrenamiento. |
| `pretrained` | `True` | Si la red base empieza ya sabiendo de ImageNet o desde cero. |
| `epochs` | `25` | Cuantas veces se repasa el conjunto de entrenamiento completo. |
| `batch_size` | `32` | Cuantas fotos se miran juntas antes de corregir. |
| `lr` | `3e-4` | Ritmo de aprendizaje de la parte final de la red. |
| `backbone_lr` | `3e-5` | Ritmo de aprendizaje de la red base cuando se descongela. |
| `weight_decay` | `1e-4` | Cuanto se castiga que la red se aprenda los datos de memoria. |
| `label_smoothing` | `0.1` | Cuanto se rebaja la seguridad de las etiquetas durante el aprendizaje. |
| `warmup_epochs` | `2` | Cuantas epocas iniciales el ritmo de aprendizaje sube poco a poco. |
| `freeze_epochs` | `3` | Cuantas epocas iniciales la red base queda congelada. |
| `early_stop_patience` | `6` | Cuantas epocas sin mejorar se aguantan antes de parar. |
| `amp` | `True` | Si se usa calculo en precision mixta para ir mas rapido. |
| `topk` | `5` | Cuantas especies propone el sistema en cada respuesta. |
| `min_conf` | `0.25` | Por debajo de esta seguridad, responde "no reconocida". |
| `seed` | `42` | Numero que fija el azar para que todo salga igual cada vez. |
| `num_workers` | `2` | Cuantos ayudantes preparan fotos a la vez. |
| `device` | `"auto"` | Con que motor calcular: el que haya disponible, GPU o CPU. |
| `debug` | `False` | Modo de prueba rapida con muy pocas fotos. |
| `debug_per_class` | `4` | Cuantas fotos por especie se usan en ese modo de prueba. |

Algunos de estos parametros merecen una explicacion mas larga, porque su nombre en ingles no ayuda nada a quien empieza.

**`epochs`, los repasos.** Una **epoca** es una vuelta completa por todas las fotos de entrenamiento. El sistema no aprende mirando una foto una sola vez; necesita repasar el conjunto muchas veces, corrigiendo su criterio en cada vuelta. Con 25 epocas, las 1020 fotos de entrenamiento se miran 25 veces, en orden distinto cada vez.

**`batch_size`, el lote.** Un **lote** es el grupo de fotos que se mira junto antes de corregir el criterio. El sistema no ajusta sus numeros despues de cada foto, sino despues de cada lote de 32. Es como estudiar: no cambias de metodo tras leer una sola frase, sino tras leer un parrafo. Un lote grande da una correccion mas estable pero exige mas memoria; uno pequeno es mas nervioso pero cabe en cualquier ordenador.

**`lr` y `backbone_lr`, los ritmos.** La abreviatura `lr` viene de *learning rate*, el ritmo al que el sistema corrige sus numeros. Un ritmo alto aprende rapido y puede pasarse de largo; uno bajo aprende fino pero tarda muchisimo. Aqui hay dos ritmos porque la red tiene dos partes: una parte final nueva, que empieza sin saber nada y necesita aprender deprisa (`3e-4`), y una parte base ya entrenada, que solo hay que retocar con suavidad (`3e-5`, diez veces mas lenta). La notacion `3e-4` es una forma compacta de escribir 0,0003.

**`dropout`, apagar conexiones.** Durante el entrenamiento, el sistema apaga al azar el 30 por ciento de las conexiones internas en cada lote. Suena a sabotearse a si mismo, y en parte lo es: es una tecnica deliberada para que la red no dependa de una sola conexion ni memorice las fotos de entrenamiento. Al evaluar, no se apaga nada.

**`pretrained`, partir de algo ya sabido.** Una red de este tipo tiene millones de numeros internos que hay que ajustar. Entrenarla desde cero con solo 1020 fotos de entrenamiento seria un desproposito. Con `pretrained = True` se parte de una red que ya fue entrenada con el enorme banco de imagenes ImageNet, y solo se le ensena el detalle nuevo: distinguir especies de flores.

**`label_smoothing`, no jurar en falso.** Cuando el sistema acierta una foto, la correccion normal le dice "esta es la respuesta correcta al cien por cien". Suavizar las etiquetas con `0.1` rebaja esa certeza a un 90 por ciento y reparte el resto. Asi se le ensena a no ser tan rotundo, lo que mejora su comportamiento con fotos nuevas y raras.

**`warmup_epochs` y `freeze_epochs`, empezar con cuidado.** Las dos primeras epocas el ritmo de aprendizaje sube poco a poco en lugar de arrancar de golpe, para no desbaratar los numeros heredados de ImageNet en el primer lote. Y las tres primeras epocas la red base queda **congelada**, es decir, sus numeros no se tocan y solo aprende la parte final nueva. Es prudente: primero se deja que la parte nueva se coloque, y despues se afina todo junto.

**`early_stop_patience`, saber parar.** Si pasan seis epocas seguidas sin que el resultado sobre las fotos de validacion mejore, el entrenamiento se detiene solo y se queda con la mejor version conseguida. La palabra *patience* significa paciencia, y aqui esta bien elegida: es el numero de oportunidades que se le dan antes de rendirse.

**`amp`, calcular con menos decimales.** La abreviatura viene de *automatic mixed precision*. Consiste en hacer parte de las cuentas con numeros de menos decimales, lo que ocupa menos memoria y va mas rapido, sin perder calidad apreciable. Se ignora automaticamente cuando no hay tarjeta grafica.

**`topk` y `min_conf`, la respuesta que ve la persona.** `topk = 5` significa que el sistema propone sus cinco mejores candidatas en lugar de una sola. `min_conf` es el minimo de seguridad exigido para contestar con una especie: si la mejor candidata no llega al 25 por ciento de confianza, el sistema responde "no reconocida", que es mucho mas honesto que inventarse un nombre.

**`seed`, el numero que fija el azar.** El sistema usa azar en varios sitios: al desordenar las fotos, al recortarlas y al apagar conexiones. `seed = 42` es la semilla que hace que ese azar sea siempre el mismo, y se explica con calma mas abajo, junto a la herramienta `set_seed`.

**`num_workers`, los ayudantes.** Preparar cada foto es trabajo pesado: hay que abrir el archivo, recortarlo y convertirlo. Con `num_workers = 2`, dos procesos ayudantes van preparando lotes mientras el principal calcula. En Windows esos ayudantes pueden dar problemas, y por eso existe la posibilidad de bajarlo a cero.

**`device`, donde se calcula.** El valor `"auto"` significa "usa la tarjeta grafica si la hay, y si no, el procesador". La explicacion completa esta en la herramienta `resolve_device`.

### El modo debug y por que cambia cosas automaticamente

`debug` es el parametro con el nombre mas informal y, sin embargo, el mas util del dia a dia. La palabra viene del mundo de la informatica y significa "modo de depuracion": una version reducida del trabajo, pensada para comprobar que todo funciona sin esperar a que termine de verdad.

En modo debug se usan 408 fotos en lugar de las 8189 completas, como se detalla en el capitulo de los datos. Pero ademas, al fabricar la libreta, el propio archivo cambia otros ajustes por su cuenta. Esto ocurre en un metodo especial de las clases de datos, llamado `__post_init__`, cuyos cuatro guiones bajos a cada lado son la convencion de Python para los metodos internos.

```python
    def __post_init__(self) -> None:
        if self.debug:
            self.epochs = min(self.epochs, 2)
            self.freeze_epochs = 0
            self.warmup_epochs = 0
            self.amp = False
            self.num_workers = 0
        self.data_root = str(self.data_root)
        self.out_dir = str(self.out_dir)
```

- `def __post_init__(self)` es el metodo que la clase ejecuta automaticamente **justo despues** de rellenar los apartados. Es el sitio natural para retocar la libreta segun lo que se acaba de escribir en ella.
- `if self.debug:` significa "solo si se ha pedido el modo de prueba". En una corrida normal, todo lo que viene debajo se salta y nada cambia.
- `self.epochs = min(self.epochs, 2)` deja los repasos en dos como mucho. `min` elige el menor de los dos valores, asi que si alguien pide 25 repasos en modo debug, se queda con 2; si pide 1, se queda con 1. La idea es que la prueba dure minutos, no horas.
- `self.freeze_epochs = 0` y `self.warmup_epochs = 0` desactivan el cuidado inicial. Tiene sentido: con solo dos epocas, dedicar tres a congelar la red base o dos a calentar el ritmo no dejaria tiempo para aprender nada. Se quiere comprobar que el mecanismo gira, no obtener un buen resultado.
- `self.amp = False` desactiva el calculo de menos decimales. Es una simplificacion para que la prueba no dependa de capacidades especiales del motor de calculo.
- `self.num_workers = 0` deja de usar procesos ayudantes. Esto es puro pragmatismo: en una prueba de un minuto, arrancar y coordinar dos procesos auxiliares cuesta mas de lo que aporta, y ademas es la causa tipica de errores en Windows.
- Las dos ultimas lineas, `self.data_root = str(self.data_root)` y `self.out_dir = str(self.out_dir)`, envuelven esos valores en texto. Sirve para que la libreta funcione igual si alguien escribe la ruta como texto `"data"` o como objeto de tipo ruta. Convertirlo siempre a texto evita una familia entera de errores tontos mas adelante.

> Nota: que el modo debug cambie los ajustes por su cuenta es comodo, pero tambien es la clase de magia que conviene conocer. Si algun dia una corrida se comporta de forma rara y el resultado no cuadra con lo pedido, lo primero que hay que mirar es si el modo debug quedo encendido.

### Las propiedades de rutas

El final de `src/config.py` resuelve una molestia cotidiana: decidir donde va cada archivo de salida. En lugar de repartir rutas escritas a mano por todos los programas, la libreta las construye a partir de `out_dir`.

```python
    # ------------------------------------------------------------- derivados
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.out_dir) / "best.pt"

    @property
    def history_path(self) -> Path:
        return Path(self.out_dir) / "history.json"

    @property
    def metrics_path(self) -> Path:
        return Path(self.out_dir) / "metrics.json"

    @property
    def config_path(self) -> Path:
        return Path(self.out_dir) / "config_usada.json"

    def figures_dir(self) -> Path:
        return Path(self.out_dir) / "figuras"

    def predictions_dir(self) -> Path:
        return Path(self.out_dir) / "predicciones"

    def ensure_dirs(self) -> None:
        """Crea los directorios de salida. Idempotente."""
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.figures_dir().mkdir(parents=True, exist_ok=True)
        self.predictions_dir().mkdir(parents=True, exist_ok=True)
        Path(self.data_root).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)
```

- Una **propiedad** es un apartado que no guarda un valor fijo, sino que lo calcula en el momento de pedirlo. Se reconoce por el decorador `@property` encima. La diferencia practica es esta: `cfg.out_dir` es un valor guardado, mientras que `cfg.checkpoint_path` se calcula al vuelo. Por eso las propiedades se usan sin parentesis, como si fueran un dato mas, y los metodos se usan con parentesis.
- `Path(self.out_dir) / "best.pt"` construye la ruta del archivo completando la carpeta de salida con el nombre del archivo. La barra inclinada entre las dos partes no es una division: en el modulo `Path` significa "y dentro de esta carpeta". El resultado es algo como `resultados\best.pt`. Se usa `Path` en lugar de pegar textos a mano porque `Path` sabe adaptarse al sistema operativo y coloca la barra correcta en Windows y en Linux.
- `checkpoint_path` apunta a `best.pt`, el archivo que guarda la mejor version del modelo conseguida durante el entrenamiento. La palabra inglesa *checkpoint* se usa mucho en este terreno y significa "punto de control": una copia del estado del trabajo que permite retomarlo o usarlo despues. Es, con diferencia, el archivo mas importante de todos, porque es lo que convierte el esfuerzo de entrenar en algo que luego se puede usar para reconocer flores.
- `history_path` apunta a `history.json`, el cuaderno de bitacora del entrenamiento. Guarda, epoca por epoca, como iba el sistema: cuanto se equivocaba, cuanto acertaba y con que ritmo estaba aprendiendo. Sirve para dibujar las curvas de aprendizaje y para justificar los resultados.
- `metrics_path` apunta a `metrics.json`, el resumen de la evaluacion final: porcentaje de acierto exacto, acierto entre las cinco primeras propuestas, y el detalle por especie. Es de donde salen las cifras que aparecen en los informes.
- `config_path` apunta a `config_usada.json`, la copia de la libreta tal como estaba en esa corrida. Este archivo es el que evita la pregunta incomoda "y esto, con que ajustes se hizo?". Abriendolo se sabe todo.
- `figures_dir()` y `predictions_dir()` son metodos, no propiedades, y por eso se llaman con parentesis. Devuelven dos carpetas: una para los graficos y otra para las predicciones volcadas a archivo. Son metodos y no propiedades por coherencia con `ensure_dirs`, ya que las tres cosas devuelven o crean carpetas y se leen mejor juntas.
- `ensure_dirs()` crea de una vez todas las carpetas de salida. El comentario final de su documentacion dice `Idempotente`, una palabra que conviene conocer: una operacion idempotente se puede repetir tantas veces como se quiera sin que el resultado empeore ni de error. Aqui, pedir "crea las carpetas" cuando ya existen no rompe nada.
- `Path(self.out_dir).mkdir(parents=True, exist_ok=True)` es la receta que hace posible esa repeticion tranquila. `parents=True` significa "crea tambien las carpetas intermedias que falten", y `exist_ok=True` significa "si ya existe, no te quejes y sigue".
- `to_dict()` cierra el archivo devolviendo la libreta entera convertida en diccionario, lista para guardarse en `config_usada.json`. Es la unica razon por la que se importaba `asdict` al principio.

### set_seed: el dado de los numeros aleatorios

`src/utils.py` empieza con una funcion corta que se llama al arrancar cualquier trabajo y que existe por una razon muy concreta: que un experimento se pueda repetir.

```python
def set_seed(seed: int = 42) -> None:
    """Fija la semilla en random, numpy y torch (CPU y CUDA) para reproducibilidad."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - entorno sin torch
        pass
```

- La palabra **semilla** traduce el ingles *seed* y describe muy bien lo que hace. En un generador de numeros al azar, la semilla es el punto de partida: si le das siempre la misma, obtienes siempre la misma secuencia de numeros "al azar". Como los dados cargados, pero a proposito y de forma util.
- `def set_seed(seed: int = 42) -> None:` declara la funcion. Recibe un numero entero y, si no se le da ninguno, usa 42. El `-> None` avisa de que la funcion no devuelve nada: su tarea es dejar el entorno preparado, no calcular un valor.
- `random.seed(seed)` fija el azar del modulo `random`, el que usa Python de serie. Con esto queda fijado, por ejemplo, el desorden de las fotos.
- `np.random.seed(seed)` fija el azar de `numpy`, la libreria de calculo con listas de numeros. Hace falta repetirlo porque es un generador distinto del anterior: fijar uno no fija el otro.
- `os.environ["PYTHONHASHSEED"] = str(seed)` fija el azar interno que Python usa para ordenar ciertas colecciones. Aunque parezca un detalle minucioso, deja la ejecucion completamente determinista.
- `torch.manual_seed(seed)` y `torch.cuda.manual_seed_all(seed)` fijan el azar de PyTorch, la libreria de inteligencia artificial. Se fijan los dos porque los calculos hechos en el procesador y los hechos en la tarjeta grafica pueden usar generadores separados.
- El bloque `try` y `except ImportError` rodea a PyTorch y termina con `pass`, que significa "no hagas nada". Es una precaucion: si PyTorch no estuviera instalado en ese ordenador, la funcion fijaria el azar de lo demas y seguiria sin romperse.
- `str(seed)` convierte el numero en texto, porque las variables de entorno son siempre texto.

**Por que importa la reproducibilidad.** Dos razones, y las dos son practicas. La primera es la **confianza**: si dos ejecuciones con los mismos ajustes dan resultados distintos, es imposible saber si una mejora se debe a un cambio que hiciste o simplemente a la casualidad. Fijando la semilla, la unica diferencia entre dos corridas es lo que tu cambiaste a proposito. La segunda es la **comparacion honesta**: cuando en el capitulo de resultados se dice "esto mejoro", el lector puede repetir la corrida y obtener las mismas cifras. Sin semilla, cada cual obtendria las suyas y la afirmacion quedaria en el aire.

### resolve_device: elegir el motor sin escribirlo a mano

La segunda herramienta decide donde se hacen las cuentas. Es la que evita que alguien escriba "cuda" en mitad del codigo y rompa el programa en cualquier ordenador que no tenga la tarjeta adecuada.

```python
def resolve_device(spec: str = "auto"):
    """Devuelve un torch.device a partir de 'auto' | 'cuda' | 'cpu'.

    'auto' usa CUDA si esta disponible y si no cae a CPU (Colab sin GPU incluido).
    Si se pide 'cuda' y no hay GPU, lanza RuntimeError con un mensaje claro.
    """
    import torch

    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Se pidio device='cuda' pero torch.cuda.is_available() es False. "
            "En Colab: Entorno de ejecucion > Cambiar tipo de entorno > GPU T4."
        )
    return torch.device(spec)
```

- Aqui aparece el termino central: el **dispositivo**, en ingles *device*. Es el motor de calculo que hace las multiplicaciones. Hay dos de uso comun: la **CPU**, el procesador normal del ordenador, que sirve para todo pero es lento en este tipo de cuentas, y la **GPU**, la tarjeta grafica, que hace muchisimas operaciones parecidas a la vez y por eso acelera enormemente el entrenamiento.
- La funcion recibe un texto, que puede ser `"auto"`, `"cuda"` o `"cpu"`. `cuda` es el nombre de la tecnologia de tarjetas graficas de NVIDIA. Devuelve un objeto del tipo `torch.device`, que es la etiqueta que PyTorch entiende para saber donde colocar cada calculo.
- `spec = (spec or "auto").lower()` limpia lo recibido antes de decidir. `spec or "auto"` significa "usa lo que te dieron, y si te dieron algo vacio, usa auto"; es una red de seguridad contra valores en blanco. El `.lower()` pasa todo a minusculas para que `"CUDA"` y `"cuda"` se traten igual.
- `torch.cuda.is_available()` es la pregunta clave: "tienes una tarjeta grafica utilizable?". Devuelve verdadero o falso. Con `"auto"`, la linea `torch.device("cuda" if ... else "cpu")` elige la tarjeta si la hay y el procesador si no. Ese `if` metido en medio de la linea es una forma compacta de escribir una eleccion entre dos valores.
- El bloque siguiente es el interesante: si alguien pidio `"cuda"` a proposito y no hay tarjeta, la funcion se niega a continuar y lanza un error con un mensaje que explica como conseguir una GPU en Colab. Negarse es mejor que la alternativa: si continuara en silencio, la corrida tardaria horas sin que nadie entendiera por que.
- La ultima linea, `return torch.device(spec)`, cubre el caso de `"cpu"` explicito. Aqui no hay comprobacion ninguna, y es correcto: cualquier ordenador tiene procesador.

**Por que nunca se escribe "cuda" a mano.** Porque quien escribe `cuda` a mano esta dando por hecho que hay tarjeta grafica, y ese supuesto solo se cumple en algunas maquinas. El proyecto tiene que funcionar en tres sitios muy distintos: un portatil normal sin tarjeta dedicada, un ordenador con tarjeta, y la nube gratuita de Google Colab, donde la tarjeta puede estar disponible o no segun la sesion. Escribir `cuda` fijo haria que el mismo codigo funcionara en un sitio y fallara en los otros. Pidiendo `"auto"` a esta funcion, el programa se adapta solo. Ademas, el mensaje de error esta escrito para un humano: no dice "CUDA no disponible" y se queda tan ancho, sino que indica el menu exacto donde activar la tarjeta.

### describe_device: ponerle nombre al motor

```python
def describe_device(device) -> str:
    """Texto legible para logs: 'cuda (NVIDIA T4)' o 'cpu'."""
    import torch

    if getattr(device, "type", str(device)) == "cuda":
        idx = getattr(device, "index", None) or 0
        return f"cuda ({torch.cuda.get_device_name(idx)})"
    return "cpu"
```

- Esta funcion recibe un dispositivo y devuelve un texto corto para escribir en el registro de la corrida, por ejemplo `cuda (NVIDIA T4)` o simplemente `cpu`. Un **registro** o *log* es el listado de mensajes que un programa va imprimiendo mientras trabaja, para que luego se pueda reconstruir lo que paso.
- `getattr(device, "type", str(device))` intenta leer el apartado `type` del dispositivo. `getattr` es una forma defensiva de preguntar "tienes este apartado?"; si no lo tiene, devuelve el tercer valor en lugar de fallar. Es una muestra del estilo del proyecto: preguntar antes de dar por hecho.
- `idx = getattr(device, "index", None) or 0` hace lo mismo con el numero de tarjeta, por si un ordenador tuviera varias. El `or 0` cubre el caso de que no venga ningun numero y asume la primera.
- `torch.cuda.get_device_name(idx)` pregunta el nombre comercial de la tarjeta, lo que permite que el registro diga si la corrida fue en una T4 de Google o en la tarjeta de casa.
- La ultima linea devuelve `"cpu"` para cualquier otra cosa. Esta funcion nunca acompana a un error: solo informa, y por eso su via de escape es tan simple.

Su utilidad es pequena pero real: cuando dentro de un mes alguien lea el historial de una corrida, sabra si aquel resultado se obtuvo con tarjeta grafica o con procesador. Es informacion que, en una comparacion de tiempos, cambia por completo la interpretacion.

### save_json y load_json: guardar y leer resultados

```python
def save_json(path: str | Path, obj: Any) -> None:
    """Guarda JSON serializable (convierte Path y tipos numpy/torch basicos)."""
    p = Path(path)
    ensure_dir(p.parent)

    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "item"):  # escalares numpy / torch
            return o.item()
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(f"Objeto no serializable: {type(o)!r}")

    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)
```

```python
def load_json(path: str | Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)
```

- **JSON** es un formato de texto para guardar datos ordenados. Se lee a simple vista y lo entienden practicamente todos los programas, lo que lo hace ideal para los resultados: el historial, las metricas y los ajustes de una corrida quedan guardados en archivos de texto que se pueden abrir con el bloc de notas.
- `save_json` recibe una ruta y un objeto, y lo escribe en disco. `load_json` hace lo contrario: lee el archivo y devuelve el objeto reconstruido.
- `p = Path(path)` y `ensure_dir(p.parent)` preparan el terreno antes de escribir. `p.parent` es la carpeta que contiene al archivo, y `ensure_dir` se encarga de que exista. Asi es imposible el error clasico de intentar escribir en una carpeta que aun no se ha creado.
- La anotacion `str | Path` significa "acepta texto o un objeto de tipo ruta", y `Any` significa "cualquier objeto". La barra vertical se lee "o".
- La funcion interna `_default` es la que resuelve los problemas de traduccion. El formato JSON solo admite numeros sencillos, texto, listas y diccionarios; cualquier otra cosa hay que traducirla antes de escribirla, y esa es exactamente su tarea.
- El primer caso de esa traduccion, `if isinstance(o, Path): return str(o)`, convierte una ruta en texto. `isinstance` pregunta "es este objeto del tipo tal?".
- El segundo, `if hasattr(o, "item")`, cubre los numeros que vienen de PyTorch o de numpy. Estos numeros son objetos con minusculas diferencias respecto a un numero corriente de Python, y el metodo `item()` los devuelve como numero normal. Sin esta linea, guardar metricas fallaria con un error incomprensible, porque el formato JSON no sabe que hacer con un numero de PyTorch.
- El tercero, `if isinstance(o, set): return sorted(o)`, convierte un conjunto en una lista ordenada. Un **conjunto** es una coleccion sin orden y sin repetidos; como el formato JSON no tiene ese concepto, se guarda como lista. Ordenarlo hace que el archivo sea estable: dos corridas identicas producen exactamente el mismo texto.
- La ultima linea, `raise TypeError(...)`, es el caso de "no se como traducir esto". `raise` significa lanzar un error a proposito, y el mensaje incluye el tipo de objeto para que quien lo vea sepa que pasa. Fallar aqui y ahora es mucho mejor que escribir un archivo silenciosamente incompleto.
- `open(p, "w", encoding="utf-8")` abre el archivo para escritura, con la `w` de *write*. El detalle `encoding="utf-8"` merece mencion: es la tabla de caracteres que permite guardar acentos y enes correctamente. Sin ella, en Windows la palabra "validacion" podria guardarse con caracteres rotos. `fh` es simplemente el nombre que se le da al archivo abierto, abreviatura de *file handle*.
- `json.dump(..., indent=2)` escribe con dos espacios de sangrado en cada nivel, pensando en la persona que abra el archivo: asi se ve la estructura en lugar de una linea larguisima. `ensure_ascii=False` pide que los acentos se guarden tal cual y no como codigos numericos.
- `load_json` no necesita nada de todo eso porque leer es mas facil: abre el archivo en modo lectura, con la `r` de *read*, y devuelve su contenido. El `with` que aparece en las dos funciones garantiza que el archivo se cierre solo al terminar, incluso si algo falla por el camino.

### count_params: contar lo que el modelo aprendio

```python
def count_params(model) -> int:
    """Numero de parametros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- Un **parametro** de un modelo es cada uno de los numeros internos que se ajustan durante el entrenamiento. Una red como la que usa este proyecto tiene millones, y ese numero es la mejor medida de su tamano.
- `model.parameters()` entrega la lista de todos los parametros del modelo. `sum(...)` los recorre y suma una cantidad por cada uno.
- `p.numel()` cuenta cuantos numeros hay dentro de ese parametro concreto. El nombre viene de *number of elements*, numero de elementos. Un parametro no tiene por que ser un solo numero: casi siempre es una tabla de muchos, y `numel` cuenta cuantos.
- El filtro `if p.requires_grad` es la parte importante: deja fuera los parametros **congelados**. `requires_grad` significa "necesita ajuste"; cuando vale falso, ese parametro esta bloqueado y no se toca. Contar solo los ajustables responde a la pregunta que de verdad interesa, que es cuanto tiene que aprender el sistema, no cuanto ocupa.
- La sintaxis `sum(... for p in ... if ...)` se llama comprension o generador: es una forma compacta de decir "suma tal cosa para cada elemento que cumpla tal condicion", sin escribir un bucle de tres lineas.

El resultado de esta funcion acaba en el historial de `train.py` y en los informes, donde sirve para explicar por que unos modelos tardan mas que otros.

### ensure_dir: crear carpetas sin sobresaltos

```python
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

- Convierte lo recibido en una ruta, crea la carpeta y devuelve la ruta ya construida. Es la version corta y reutilizable de lo que `ensure_dirs` hace dentro de la libreta de configuracion.
- `mkdir(parents=True, exist_ok=True)` es la llamada que lo hace todo: `mkdir` viene de *make directory*, crear carpeta; `parents=True` crea tambien las carpetas intermedias que falten, y `exist_ok=True` permite repetir la llamada sin que se queje si ya existe.
- Devuelve la ruta, y ese detalle tan pequeno es comodo: permite escribir `p = ensure_dir("resultados/figuras")` y seguir usando `p` sabiendo que la carpeta existe. La funcion se apoya en `save_json`, que la llama para asegurarse de que la carpeta del archivo esta lista antes de escribir.

## Por qué se hizo asi

Este capitulo recoge las decisiones de diseno de los dos archivos y el motivo de cada una.

**Por que centralizar los ajustes en lugar de repartirlos.** Todos los numeros importantes viven en un solo archivo. La ventaja se nota al comparar: cambiar los repasos de 25 a 10 es editar una linea, y la libreta entera sigue siendo coherente. La segunda ventaja, menos evidente, es que `config_usada.json` guarda la libreta completa de cada corrida. Como consecuencia, un resultado siempre se puede reproducir, porque los ajustes con los que se obtuvo quedaron archivados junto a el.

**Por que `config.py` no importa PyTorch.** La cabecera del archivo lo declara como contrato: el modulo de configuracion no debe importar PyTorch y tiene que poder cargarse en cualquier entorno. El motivo es la comodidad y la robustez: asi una herramienta que solo quiera leer los ajustes no arrastra una libreria enorme, y un ordenador donde PyTorch este mal instalado todavia puede consultar la configuracion. Es la misma idea que aparece en `set_seed`, donde PyTorch se importa dentro de un `try`.

**Por que los ajustes ambiguos tienen nombres explicitos.** `lr` y `backbone_lr` en lugar de un solo ritmo; `debug_per_class` en lugar de un numero suelto de imagenes. Cada nombre dice a que parte del sistema afecta. Cuando dentro de seis meses alguien lea una linea de configuracion suelta, no tendra que adivinar que significa.

**Por que el modo debug modifica ajustes por su cuenta.** Porque la alternativa, pedirle a quien escribe el codigo que recuerde cambiar cinco parametros cada vez que quiere una prueba rapida, se olvida siempre. Un modo de prueba que no fuera coherente consigo mismo daria resultados enganosos: con dos epocas y tres de congelacion, por ejemplo, la red base no llegaria a descongelarse nunca y la prueba no comprobaria lo que debe. Que el propio archivo ajuste esos valores garantiza que cualquier prueba rapida sea internamente coherente. El precio es esa pizca de magia que ya avisamos en su seccion.

**Por que las rutas de salida son propiedades calculadas y no valores guardados.** Si `checkpoint_path` fuera un valor guardado, alguien podria cambiar `out_dir` y olvidarse de actualizar la ruta del archivo, con el resultado de escribir el modelo bueno en la carpeta antigua. Calculandola al vuelo a partir de `out_dir`, ese desajuste es imposible: hay una sola fuente de verdad y todo lo demas se deriva de ella.

**Por que `set_seed` fija el azar en tres librerias distintas.** Porque cada una tiene su propio generador de numeros al azar y fijar uno no fija los otros. Si solo se fijara el de PyTorch, quedaria suelto el desorden de las fotos, que lo decide otra libreria; y el resultado cambiaria de una corrida a otra. Para que la reproducibilidad sea real hay que cubrir todos los sitios donde se usa el azar.

**Por que nunca se escribe "cuda" a mano.** Porque el mismo codigo tiene que funcionar en un portatil sin tarjeta, en un ordenador con tarjeta y en Colab, donde la tarjeta aparece o no segun la sesion. Concentrar la decision en `resolve_device` consigue que ningun otro archivo tenga que saber nada al respecto, y que un cambio de entorno no obligue a tocar veinte sitios. Esta regla esta escrita en el contrato del proyecto y se cumple en todos los modulos.

**Por que `save_json` traduce objetos en lugar de exigir numeros corrientes.** Porque las cifras que produce PyTorch y numpy no son numeros corrientes, y obligar a cada programa a convertirlos antes de guardarlos repartiria la misma conversion por todo el proyecto. Haciendola una sola vez en `save_json`, se elimina una familia entera de errores dificiles de diagnosticar. Y `load_json` existe para que cualquier programa, incluidas las pruebas automaticas, pueda leer un resultado y comprobarlo.

**Por que se distinguen parametros entrenables y congelados en `count_params`.** Porque el numero que interesa no es el tamano total del archivo, sino cuanto tiene que aprender el sistema. En este proyecto las primeras epocas transcurren con la red base congelada, y ese cambio se refleja de inmediato en el recuento. Contar solo los ajustables es la medida honesta del esfuerzo de entrenamiento.

**Por que las funciones de utilidad son deliberadamente cortas.** Hay siete herramientas pequenas y cada una hace una sola cosa. Son las piezas que usan todos los demas modulos, y precisamente por eso debian ser faciles de revisar y de probar por separado. Ademas, sus nombres y su forma de uso quedaron congelados el primer dia, de manera que el resto de programas puede apoyarse en ellas sin miedo a que cambien.
