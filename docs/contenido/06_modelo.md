# El modelo: el cerebro que aprende a distinguir flores

## Para qué sirve este archivo (en lenguaje llano)

El archivo **src/model.py** es la fábrica del cerebro del sistema. Aquí todavía no se aprende nada: lo que se hace es **construir** la estructura que después aprenderá, decidir qué partes de esa estructura se pueden modificar y guardar en el disco lo aprendido para poder reutilizarlo.

Tres palabras que se repiten en todo el documento:

- **Variable**: una caja con nombre donde se guarda un valor. Por ejemplo, la variable `dropout` guarda el número 0,3.
- **Función**: un bloque de instrucciones con nombre que recibe datos, hace un trabajo y entrega un resultado. Se usa escribiendo su nombre y un paréntesis: `build_model(...)`.
- **Clase**: un molde para fabricar objetos. Del molde llamado `FlowerClassifier` se pueden fabricar muchos cerebros distintos, cada uno con sus propios números internos.

Además de la clase, el archivo ofrece cinco funciones públicas —`build_model`, `set_backbone_trainable`, `build_optimizer`, `save_checkpoint` y `load_checkpoint`— que el resto del proyecto utiliza.

### Qué es torchvision y por qué no se entrena una red desde cero

```python
import torch
import torch.nn as nn
from torchvision import models as tv_models
```

- **torch** es la librería que hace los cálculos pesados: millones de multiplicaciones por segundo, en el procesador normal (CPU) o en una tarjeta gráfica (GPU).
- **torch.nn** trae las piezas con las que se arma una red: capas, funciones de error y demás.
- **torchvision** es la librería compañera especializada en **imágenes**. Incluye recetas de redes ya montadas y, sobre todo, redes que **ya fueron entrenadas** con millones de fotografías de la base de datos ImageNet.
- La parte `models as tv_models` significa "trae el apartado de modelos de torchvision y llámalo aquí `tv_models`": es solo un apodo para escribir menos.

Entrenar desde cero significa empezar con números al azar y enseñarle a ver desde el principio. Eso exige millones de fotos y días de cómputo. Este proyecto tiene **1020 fotos de entrenamiento** repartidas entre 102 especies de flores: es poquísimo para ese camino. Por eso se parte de una red que ya sabe reconocer bordes, texturas, pétalos y formas, y solo se le enseña la decisión final: "esta combinación de rasgos es una rosa, esta otra es un girasol".

> Nota: a esta técnica se le llama **aprendizaje por transferencia**. Se transfiere el conocimiento de una tarea enorme (reconocer mil objetos distintos en ImageNet) a una tarea pequeña (102 flores).

### La clase FlowerClassifier: el molde del cerebro

```python
class FlowerClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_classes: int = FLOWERS102_CLASSES,
        dropout: float = 0.3,
        backbone_name: str | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(in_features), int(num_classes)),
        )
```

- `class FlowerClassifier(nn.Module)` define el molde. El paréntesis del final significa **heredar de nn.Module**: el molde nuevo es un caso particular de un molde que ya trae PyTorch y recibe gratis toda su maquinaria interna (registrar los números que se pueden aprender, moverlos a la tarjeta gráfica, guardarlos y cargarlos en disco, y alternar entre modo entrenamiento y modo examen).
- `def __init__(...)` es el **constructor**: las instrucciones que se ejecutan cada vez que se fabrica un cerebro nuevo. La palabra `self` significa "este objeto que estoy construyendo".
- `super().__init__()` llama al constructor del molde padre. Sin esta línea, la maquinaria heredada de `nn.Module` no queda preparada y el modelo no funciona.
- Los nombres entre paréntesis son los **parámetros** de la función: los datos que hay que darle para que trabaje. `in_features` es cuántos números entrega el extractor por cada foto; `num_classes` es cuántas especies distintas tiene que distinguir.
- Los valores escritos después del signo igual (`num_classes: int = FLOWERS102_CLASSES`) son **valores por defecto**: si quien llama a la función no indica nada, se usan esos.
- `self.backbone = backbone` y `self.head = ...` guardan las dos mitades del cerebro como **atributos**, es decir, como variables que viven dentro del objeto. Esos dos nombres son obligatorios: el resto del proyecto busca exactamente esos dos atributos para congelar, descongelar e inspeccionar el modelo.

### El extractor de rasgos (backbone) y la cabeza (head)

El **backbone** es el ojo y la corteza visual: recibe la foto y devuelve una lista de números que describen lo que hay en ella. La **cabeza** es la parte que pone nombre: recibe esa lista y reparte puntos entre las 102 especies.

| Backbone | Números que entrega por foto | Comentario |
|---|---|---|
| `efficientnet_b0` | 1280 | El elegido por el proyecto, es el valor por defecto |
| `resnet50` | 2048 | Alternativa clásica y mucho más pesada |
| `mobilenet_v3_large` | 960 | Alternativa ligera, pensada para teléfonos |

- El proyecto eligió **EfficientNet-B0** porque da muy buen resultado con un coste bajo: una corrida completa cabe en 15 a 25 minutos de una GPU T4, y con 1020 fotos conviene no gastar en redes enormes, que se limitan a memorizar las imágenes.
- Las otras dos quedan disponibles por si se quiere comparar, pero piden más memoria y más tiempo.

```python
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(in_features), int(num_classes)),
        )
```

- `nn.AdaptiveAvgPool2d(1)` convierte el mapa de rasgos (una rejilla de números) en un solo número por canal, promediando. Así el resultado tiene siempre el mismo tamaño, sin importar el tamaño de la entrada.
- `nn.Sequential(...)` es una cadena: lo que entra pasa por la primera pieza y lo que sale de ella entra en la segunda.
- `nn.Linear(in_features, num_classes)` es la capa final de decisión. Tiene un peso por cada pareja formada por un rasgo y una especie, y produce los 102 números de salida, llamados **logits**, que son puntuaciones sin normalizar (no son porcentajes).

### El Dropout de la cabeza

```python
            nn.Dropout(p=float(dropout)),
```

- **Dropout** apaga al azar una fracción de los números durante el entrenamiento. Con `p=0.3` se apaga el 30 por ciento en cada pasada, y en la siguiente pasada se apagan otros distintos.
- Sirve para que la red no se apoye siempre en la misma pista. Si memorizara "pétalo rojo con mancha central", el dropout la obliga a buscar señales alternativas, y así aprende de verdad en lugar de aprenderse las fotos de memoria.
- Solo actúa mientras se entrena. Cuando el modelo ya está en modo examen, el dropout se desactiva solo.

### forward: el recorrido de la foto por el cerebro

```python
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Vector de caracteristicas (B, in_features) antes de la cabeza lineal."""
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = self.avgpool(feats)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, num_classes) logits (sin softmax)."""
        return self.head(self.forward_features(x))
```

- **forward** define el camino que recorren los datos. Es la función que PyTorch ejecuta automáticamente cuando en el entrenamiento se escribe `model(imagenes)`.
- `x` es un **lote**: un bloque de imágenes que se procesa de una sola vez. Su forma es `(B, 3, H, W)`: B imágenes, 3 canales de color (rojo, verde y azul), H de alto y W de ancho. En este proyecto cada foto mide **224 por 224** píxeles y los lotes son de **32 imágenes**.
- La salida tiene la forma `(B, num_classes)`: para cada imagen del lote, 102 puntuaciones.
- La nota "sin softmax" aclara que esas puntuaciones todavía no son porcentajes; convertirlas en probabilidades es tarea de otra parte del sistema.
- `feats.dim() > 2` comprueba si el backbone devolvió una rejilla en lugar de una lista. Si es así, `avgpool` la reduce a un número por canal y `torch.flatten(feats, 1)` aplana todo en una lista larga por imagen. Gracias a esto, el mismo cerebro funciona igual sin importar cuál de los tres backbones se haya elegido.

### build_model: la función que fabrica el cerebro

```python
SUPPORTED_BACKBONES: tuple[str, ...] = (
    "efficientnet_b0",
    "resnet50",
    "mobilenet_v3_large",
)
```

```python
def build_model(
    num_classes: int = FLOWERS102_CLASSES,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> FlowerClassifier:
```

- Es la única puerta de entrada: nadie fabrica un `FlowerClassifier` a mano. Así se garantiza que todos los cerebros del proyecto se construyan igual.
- Antes de construir nada, se comprueba que el backbone pedido exista de verdad:

```python
    if not isinstance(backbone, str) or backbone not in _BACKBONE_BUILDERS:
        soportados = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(
            f"Backbone no soportado: {backbone!r}. Soportados: {soportados}."
        )
```

- `raise` lanza un error que detiene el programa y muestra un mensaje. Es mejor parar de inmediato, con la lista de nombres válidos, que entrenar 25 épocas con la red equivocada y descubrirlo al final.
- Un backbone **preentrenado** significa que sus números no empiezan al azar: ya traen los valores que aprendió en ImageNet. Se piden con `pretrained=True`.
- El respaldo automático si la descarga falla:

```python
    try:
        features, in_features = builder(used_pretrained)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de descarga/carga
        if not pretrained:
            # El error es real (no de red): no lo enmascaramos.
            raise
        warnings.warn(
            f"No se pudieron cargar los pesos preentrenados de '{backbone}' "
            f"({type(exc).__name__}: {exc}). Se continua con pretrained=False "
            "(pesos aleatorios); el entrenamiento sigue siendo valido.",
            RuntimeWarning,
            stacklevel=2,
        )
        used_pretrained = False
        features, in_features = builder(False)
```

- `try` y `except` significan "intenta hacer esto; si falla, ejecuta aquello otro en lugar de morir".
- La primera vez que se usa una red preentrenada hay que **descargarla de internet**. Si no hay conexión, o si el servidor de descargas está bloqueado (algunas plataformas lo hacen), la descarga falla.
- En ese caso el programa no se detiene. Avisa con un mensaje visible, mediante `warnings.warn` con categoría `RuntimeWarning`, y continúa con **pesos aleatorios**: el extractor queda sin conocimiento previo.
- Ese aviso existe porque, aunque el entrenamiento sigue siendo válido, el resultado será peor y harán falta más épocas. Quien lea el informe debe saber que ocurrió, en vez de creer que el modelo rindió por debajo de lo esperado sin motivo.
- Detalle importante: si `pretrained` ya era falso y aun así hubo un error, el error **sí** se propaga con `raise`. Ese fallo no es de red, es un problema real y hay que verlo.

### set_backbone_trainable: congelar y descongelar

```python
def set_backbone_trainable(model: FlowerClassifier | nn.Module, trainable: bool) -> None:
    """Activa o desactiva ``requires_grad`` en **todos** los parametros del backbone.

    Se usa en las primeras epocas (``cfg.freeze_epochs``) para entrenar solo la
    cabeza. No cambia el modo train/eval del backbone.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError(
            "set_backbone_trainable espera un modelo con atributo .backbone "
            f"(recibido {type(model).__name__})."
        )
    for param in backbone.parameters():
        param.requires_grad = bool(trainable)
```

- Un **parámetro**, o peso, es cada uno de los números que la red puede aprender.
- `requires_grad` es una etiqueta que lleva cada parámetro y que significa "tengo permiso para que me modifiquen durante el aprendizaje".
- **Congelar** el backbone es poner esa etiqueta en falso en todos sus parámetros: la red los sigue usando para mirar, pero no los cambia. **Descongelar** es ponerla en verdadero y volver a permitir cambios.
- `for param in backbone.parameters()` recorre la lista completa de parámetros, de modo que la operación alcanza al backbone **entero**, no solo a sus últimos bloques. Algunos proyectos congelan únicamente el tramo final de la red; aquí se decidió descongelar todo (el motivo está en la última sección).
- `getattr(model, "backbone", None)` intenta leer el atributo y devuelve `None` si el objeto no lo tiene. Si falta, se lanza un `AttributeError` con un mensaje explicativo: es una protección contra usar la función con el objeto equivocado.
- La **analogía** del traductor: el extractor es como un traductor que ya domina un idioma. Al principio no se le cambia el idioma ni su manera de trabajar, solo se le enseña el vocabulario nuevo de las flores. Cuando ya maneja ese vocabulario, se le permite además ajustar su forma de traducir, poco a poco y con pasos pequeños.

### build_optimizer: el corrector con dos velocidades

```python
    head_params = [p for p in head.parameters()]
    backbone_params = [p for p in backbone.parameters()]

    param_groups: list[dict[str, Any]] = []
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(cfg.lr),
                "weight_decay": float(cfg.weight_decay),
                "name": "head",
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(cfg.backbone_lr),
                "weight_decay": float(cfg.weight_decay),
                "name": "backbone",
            }
        )
    if not param_groups:
        raise ValueError("El modelo no tiene parametros: no se puede crear el optimizador.")

    return torch.optim.AdamW(param_groups)
```

- Un **optimizador** es el mecanismo que decide cuánto y en qué dirección se corrige cada número después de ver el error. La corrección concreta la ordena el entrenamiento con `optimizer.step()`.
- **AdamW** es uno de esos mecanismos, muy usado porque se adapta solo a la escala de cada número. La letra W viene de **weight decay** (decaimiento de pesos): una fuerza suave que empuja los números hacia valores pequeños y evita que la red memorice las fotos en lugar de aprender. En este proyecto vale 1e-4, es decir, 0,0001.
- El **learning rate**, o velocidad de aprendizaje, es el tamaño del paso de cada corrección. Un paso grande aprende rápido pero puede destruir lo que ya se sabía; un paso pequeño es seguro pero lento.
- El **learning rate diferencial** consiste en usar dos velocidades a la vez dentro de la misma corrida, y para eso se crean dos **grupos de parámetros**. El grupo llamado `head` (la cabeza) usa 3e-4, es decir, 0,0003, porque es nueva y no sabe nada y necesita pasos grandes. El grupo llamado `backbone` (el extractor) usa 3e-5, es decir, 0,00003, diez veces más pequeño, porque ya sabe ver y solo hay que afinarlo con cuidado.
- Los dos grupos comparten el mismo `weight_decay` de 1e-4.
- Detalle de diseño: el grupo del backbone se crea **aunque el backbone esté congelado** en ese momento. El motivo está escrito en el propio código: AdamW ignora los parámetros que no tienen corrección pendiente, así que el mismo optimizador sigue sirviendo cuando más adelante se descongela el backbone, sin necesidad de fabricarlo otra vez.

### save_checkpoint: guardar la partida

```python
CHECKPOINT_KEYS: tuple[str, ...] = (
    "model_state",
    "class_names",
    "backbone",
    "num_classes",
    "img_size",
    "config",
    "epoch",
    "val_acc",
)
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

- Un **checkpoint** es como guardar la partida de un videojuego: se escribe en el disco el estado del modelo para poder retomarlo después sin volver a entrenar.
- `model.state_dict()` entrega el diccionario con todos los números aprendidos.
- Las ocho claves obligatorias del contrato son estas:

| Clave | Qué guarda |
|---|---|
| `model_state` | Todos los números aprendidos, es decir, los pesos |
| `class_names` | La lista de las 102 especies, en el mismo orden que las salidas |
| `backbone` | Qué red se usó como extractor |
| `num_classes` | Cuántas especies distingue el modelo |
| `img_size` | De qué tamaño eran las fotos, 224 píxeles |
| `config` | La receta completa con la que se entrenó |
| `epoch` | En qué época se obtuvo este resultado |
| `val_acc` | Qué exactitud tenía en ese momento |

- Se guardan los **nombres de las clases** porque el modelo no maneja nombres: maneja números del 0 al 101. La casilla 37 de la salida no significa nada por sí sola; el nombre es lo que le da sentido. Si los nombres se guardaran en otro archivo y alguien los reordenara por descuido, el sistema respondería con total seguridad la flor equivocada y no daría ningún error. Al viajar dentro del mismo archivo que los pesos, ese desastre es imposible.
- `torch.save(payload, target)` escribe el diccionario en el archivo. La línea anterior, `target.parent.mkdir(parents=True, exist_ok=True)`, crea la carpeta de destino si todavía no existe, para que el guardado no falle la primera vez.

### load_checkpoint: retomar la partida

```python
    faltantes = [k for k in CHECKPOINT_KEYS if k not in checkpoint]
    if faltantes:
        raise KeyError(
            f"Al checkpoint '{target}' le faltan claves obligatorias: {faltantes}."
        )
```

```python
    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,  # el state_dict ya trae los pesos: nunca hay descarga
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
```

- Lo primero es comprobar que el archivo exista. Si no está, el mensaje dice qué hacer: entrenar antes con `python -m src.train`.
- Después se verifica que estén las ocho claves. Si falta alguna, se lanza un `KeyError` que indica exactamente cuál. Un error inmediato y claro es preferible a un resultado mal calculado en silencio.
- El modelo se vuelve a fabricar con la misma receta (mismo backbone, mismo número de clases, mismo dropout) y se pide con `pretrained=False`, porque los números buenos ya vienen dentro del archivo: no hay que descargar nada.
- `load_state_dict(..., strict=True)` copia los números guardados dentro del modelo recién fabricado y exige que **todos** encajen en su sitio. Si la estructura no coincidiera, se detiene con un error en lugar de cargar a medias.
- `model.eval()` pone el modelo en modo examen: desactiva el dropout para que las respuestas sean estables y repetibles.
- La función devuelve tres cosas: el modelo listo para usar, la lista con los nombres de las clases y un diccionario llamado `meta` con el resto del contenido del archivo (la configuración, el tamaño de imagen, la época, la exactitud y cualquier dato adicional).

## Por qué se hizo así

- **Se parte de una red ya entrenada y no de cero** porque hay 1020 fotos de entrenamiento. Con tan pocos ejemplos, una red que empieza al azar se limita a memorizar las fotos y falla con cualquier imagen nueva. Una red que ya sabe ver flores solo necesita aprender los nombres.
- **Los dos atributos `.backbone` y `.head` son obligatorios**, no un detalle estético. Todo el resto del proyecto depende de poder separar las dos mitades: para congelar y descongelar, para armar los dos grupos de parámetros y para inspeccionar qué mira el modelo cuando acierta.
- **Se congelan primero y se descongelan después las dos mitades** en lugar de entrenar todo a la vez. Si desde el primer día se corrigiera el extractor con las señales de una cabeza que todavía responde al azar, esas correcciones borrarían el conocimiento previo. Congelar al principio protege lo que ya se sabía.
- **El backbone se descongela entero, no solo sus últimos bloques.** Es cierto que cuesta más tiempo y memoria, pero las 102 especies de flores se parecen mucho entre sí y los rasgos que sirven para perros o coches no bastan. Permitir que todo el extractor se especialice en flores es lo que explica el salto de precisión de las últimas épocas, y se controla con una velocidad de aprendizaje diez veces menor que la de la cabeza.
- **El optimizador se construye una sola vez, al principio**, con los dos grupos ya dentro. Si se fabricara de nuevo al descongelar, se perdería el estado interno de AdamW y el entrenamiento sufriría un reinicio invisible.
- **El checkpoint se guarda con toda la información necesaria para reconstruir el modelo**, y no solo con los pesos. Un archivo con únicamente los números es inservible: no dice de qué tamaño eran las fotos, ni cuántas especies había, ni en qué orden. Guardando la receta completa, el archivo se explica a sí mismo y funciona meses después, en otro ordenador y sin recordar nada.
- **Los errores de verdad no se ocultan.** Solo se perdona el fallo de descarga de los pesos preentrenados, y aun así avisando. Cualquier otro problema detiene el programa con un mensaje claro, porque un modelo entrenado con la configuración equivocada produce resultados que parecen válidos y no lo son.
- **No hay ninguna orden que obligue a usar la tarjeta gráfica** (`cuda`). El dispositivo se recibe como argumento. Gracias a eso el mismo archivo funciona en un ordenador sin GPU, en una GPU de la nube y en el portátil de quien revisa el proyecto, sin tocar una sola línea.
