# El entrenamiento: las 25 pasadas que convierten el modelo en experto

## Para qué sirve este archivo (en lenguaje llano)

El archivo **src/train.py** es el que enseña de verdad. Toma el cerebro construido en el archivo del modelo, las fotos preparadas en el archivo de datos y ejecuta la receta que convierte una red genérica en una especialista en flores.

La corrida completa del proyecto hace **25 pasadas** por las 1020 fotos de entrenamiento, con lotes de **32 imágenes** de **224 por 224** píxeles. En una GPU T4 eso tarda entre **15 y 25 minutos**. No hace falta tanto para ver que el sistema funciona: con solo **4 épocas** ya se alcanza un **37,8 por ciento** de acierto en la primera respuesta y un **67,4 por ciento** en las cinco primeras, medido sobre el conjunto oficial de prueba.

### La función train: la receta completa

```python
def train(cfg: Config) -> dict:
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    cfg.ensure_dirs()
```

- `cfg` es el objeto de **configuración**: una caja con todos los ajustes de la corrida (cuántas épocas, de qué tamaño son los lotes, qué velocidades de aprendizaje, dónde se guardan los resultados).
- `set_seed(int(cfg.seed))` fija el punto de partida de todos los números al azar. En este proyecto la semilla es 42 y siempre la misma.
- `resolve_device(cfg.device)` decide dónde se calcula: en la tarjeta gráfica si está disponible, y si no en el procesador.
- `cfg.ensure_dirs()` crea las carpetas de salida antes de empezar, para que nada falle a mitad de camino.

Antes del bucle se arma todo lo necesario:

```python
    model = build_model(
        num_classes=int(cfg.num_classes),
        backbone=str(cfg.backbone),
        pretrained=bool(cfg.pretrained),
        dropout=float(cfg.dropout),
    ).to(device)

    freeze_epochs = max(0, int(cfg.freeze_epochs))
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)  # primeras epocas: solo la cabeza
    else:
        set_backbone_trainable(model, True)
```

- `.to(device)` mueve el modelo al dispositivo elegido.
- `freeze_epochs` vale 3 en la corrida completa: durante las **3 primeras épocas** el extractor de rasgos queda congelado y solo aprende la cabeza.
- El optimizador se construye una sola vez y se reutiliza durante todo el entrenamiento, incluso cuando el backbone se descongela.

### El bucle de entrenamiento

```python
    for epoch_index in range(epochs):
        epoch = epoch_index + 1  # 1-based (ver docstring del modulo)
        congelado = epoch <= freeze_epochs
        set_backbone_trainable(model, not congelado)

        train_loss, train_acc = _etapa_con_fallback("train", epoch)
        val_loss, val_acc, val_top5 = _etapa_con_fallback("val", epoch)
```

- El bucle se repite una vez por época.
- Las **épocas se cuentan desde 1**, no desde 0, para que los informes digan "época 3" de forma natural.
- En cada vuelta se decide si el backbone sigue congelado y se entrena, primero con las fotos de estudio y después con las de examen.
- Al final de cada época se guarda una línea en el registro con la pérdida, el acierto y la velocidad de aprendizaje usada.

### Qué es una época y qué es una iteración

- Una **época** es una pasada completa por todas las fotos de entrenamiento. La corrida completa hace 25.
- Una **iteración**, o paso, es el trabajo con un solo lote. Con 1020 fotos y lotes de 32, cada época se recorre en unas 32 iteraciones.
- La **pérdida** (en inglés, **loss**) es un único número que mide cuánto se equivocó el modelo en ese lote: cuanto más pequeño, mejor.
- El **acierto**, o **accuracy**, es la fracción de imágenes bien clasificadas. Se mide en top-1 (la primera respuesta es la correcta) y en top-5 (la especie correcta está entre las cinco respuestas más probables del modelo).

### El profesor corrigiendo exámenes: zero_grad, backward y step

```python
    for batch in bar:
        images, targets = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
```

```python
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
```

- `images` es el lote: 32 fotos convertidas en números. `targets` son las respuestas correctas de esas 32 fotos. `logits` son las 102 puntuaciones que el modelo da a cada imagen.
- `optimizer.zero_grad(set_to_none=True)` borra las notas de corrección del lote anterior. Sin este borrado, los errores se irían acumulando y las correcciones se volverían disparatadas. Es como limpiar la pizarra antes de empezar el examen siguiente.
- `loss.backward()` es el trabajo del profesor que busca culpables: calcula un **gradiente** para cada número del modelo. Un gradiente es una indicación de cuánto contribuyó ese número al error y hacia dónde hay que moverlo para reducirlo.
- `optimizer.step()` aplica las correcciones: mueve cada número un poquito en la dirección que reduce el error. El tamaño de ese movimiento es la velocidad de aprendizaje.
- La **analogía completa** es la del profesor corrigiendo exámenes: se reparte un examen (el lote), se calcula la nota del error (la pérdida), se averigua en qué se falló y cuánto pesa cada parte (el gradiente) y se anota la corrección que hay que estudiar (el paso del optimizador). Después, examen nuevo y pizarra limpia.
- La evaluación funciona distinto:

```python
@torch.no_grad()
def _run_eval_epoch(
```

- `@torch.no_grad()` significa "aquí no hace falta calcular gradientes". En modo examen solo se quiere saber si acierta, no corregir nada, así que se ahorra muchísimo tiempo y memoria. Además se llama a `model.eval()`, que desactiva el dropout.

### label_smoothing: no jurar que la respuesta es el cien por cien

```python
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
```

- La **función de pérdida** es la fórmula que convierte "cuánto te equivocaste" en un solo número. La elegida aquí es la entropía cruzada, la habitual cuando hay que elegir una sola clase entre muchas.
- Sin suavizado, la respuesta correcta vale 1 (el cien por cien) y todas las demás valen 0. Con `label_smoothing=0.1`, la respuesta correcta vale 0,9 y el 0,1 restante se reparte entre todas las demás.
- **Por qué se suaviza**: obligar al modelo a estar absolutamente seguro lo empuja a memorizar los casos raros, porque la única forma de no fallar nunca es aprenderse las fotos de memoria. Con el suavizado, el modelo aprende a ser prudente, duda un poco y se equivoca mejor cuando aparece una flor que nunca vio. En este proyecto el valor es 0,1.

### El planificador de la velocidad de aprendizaje

```python
def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> Any:
    warmup = max(0, int(cfg.warmup_epochs))
    epochs = max(1, int(cfg.epochs))
    cosine_epochs = max(1, epochs - warmup)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=0.0
    )
    if warmup <= 0:
        return cosine

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
    )
```

- Un **planificador** es el encargado de cambiar la velocidad de aprendizaje a lo largo del entrenamiento. No conviene usar siempre la misma: al principio estorba y al final sobra.
- La primera parte es el **calentamiento** (**warmup**), con `LinearLR`: durante las **2 primeras épocas** la velocidad sube desde el 10 por ciento de su valor hasta el valor completo. Se hace así porque al principio el modelo responde al azar y un paso grande con información tan mala podría destrozar los pesos preentrenados.
- La segunda parte es el **enfriamiento**, con `CosineAnnealingLR`: la velocidad baja de forma suave, siguiendo una curva de coseno, hasta casi cero en la última época. Así el modelo da pasos largos al principio, cuando está buscando la zona buena, y pasos diminutos al final, cuando solo queda afinar.
- `SequentialLR` encadena las dos partes: primero el calentamiento y luego el enfriamiento.
- El planificador actúa sobre **los dos grupos de parámetros a la vez**, y como aplica el mismo factor de escala a cada uno, la cabeza y el backbone conservan siempre su proporción de diez a uno (3e-4 frente a 3e-5).
- El valor que se anota en el historial es el de la cabeza, y se lee justo **antes** de aplicar el paso del planificador, para que la curva del informe muestre la velocidad que se usó de verdad en esa época y no la de la época siguiente:

```python
        head_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
```

### Precisión mixta (AMP)

```python
    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    scaler = _make_grad_scaler() if amp_enabled else None
    if bool(cfg.amp) and not amp_enabled:
        print("  AMP solicitado pero desactivado: el device no es CUDA (CPU).")
```

- Los números de la red se guardan normalmente con muchos decimales. La **precisión mixta** usa un formato más corto para la mayoría de los cálculos: ocupa la mitad de memoria y va aproximadamente al doble de velocidad.
- El inconveniente es que un formato corto pierde decimales y puede producir resultados inservibles. Para evitarlo entra en juego el **escalador** (`GradScaler`), que multiplica el error por un número grande antes de repartirlo y lo devuelve a su sitio después, de modo que los decimales importantes no se pierdan.
- **Solo se activa si hay GPU**, y esa es la razón de la condición `device.type == "cuda"`. Esa aceleración solo existe en las tarjetas gráficas de NVIDIA; en el procesador no aporta nada y solo añadiría riesgo. Por eso, cuando se pide en un ordenador sin tarjeta, el sistema lo desactiva y avisa con un mensaje, sin detenerse. Es la diferencia entre los 15 o 25 minutos de una T4 y varias horas de procesador.

### El guardado del mejor modelo y la parada temprana

```python
        mejora = val_acc > best_val_acc
        marca = ""
        if mejora:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            epochs_sin_mejora = 0
            save_checkpoint(
                cfg.checkpoint_path,
                model,
                class_names,
                cfg,
                epoch=epoch,
                val_acc=val_acc,
            )
            marca = " *mejor"
        else:
            epochs_sin_mejora += 1
```

```python
        if epochs_sin_mejora >= max(1, int(cfg.early_stop_patience)):
            print(
                f"Early stopping: {epochs_sin_mejora} epocas sin mejorar val_acc "
                f"(paciencia {cfg.early_stop_patience}); mejor = {best_val_acc:.4f} "
                f"en la epoca {best_epoch}."
            )
            break
```

- La **exactitud de validación** (`val_acc`) se mide sobre fotos que el modelo nunca usa para aprender. Es el equivalente a un examen sorpresa: la nota de estudio se puede inflar memorizando, la del examen sorpresa no.
- El modelo solo se guarda en el disco cuando mejora esa marca. El archivo resultante es el mejor de toda la corrida, no el último. La línea del registro marca esas épocas con la etiqueta `*mejor`.
- La **parada temprana** vigila las épocas sin mejora. La paciencia es de **6 épocas**: si pasan seis seguidas sin superar el mejor resultado, el entrenamiento se detiene solo y se queda con lo mejor que había conseguido. Ahorra tiempo de GPU y, sobre todo, evita seguir entrenando un modelo que ya empezó a memorizar.

### El historial: las 13 claves

```python
HISTORY_LIST_KEYS: tuple[str, ...] = (
    "epochs",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_top5",
    "lr",
)
HISTORY_SCALAR_KEYS: tuple[str, ...] = (
    "best_val_acc",
    "best_epoch",
    "epochs_run",
    "device",
    "num_params",
    "segundos",
)
```

- Un **historial** es el acta de la corrida. Se guarda como archivo `history.json`, que es un formato de texto que cualquier programa puede leer, y es la materia prima de las gráficas y de las tablas del informe.
- Las siete primeras claves son listas con un valor por época. Las seis últimas son valores sueltos que resumen toda la corrida.
- La clave `val_top5` mide si la respuesta correcta aparece entre las cinco primeras del modelo. Es una medida más indulgente que el top-1 y resulta útil para saber si el modelo se acerca aunque falle por poco.

| Clave | Qué guarda |
|---|---|
| `epochs` | El número de cada época ejecutada |
| `train_loss` | El error medio sobre las fotos de entrenamiento |
| `train_acc` | La fracción de aciertos en entrenamiento |
| `val_loss` | El error medio sobre las fotos de validación |
| `val_acc` | La fracción de aciertos en validación, la nota honesta |
| `val_top5` | La fracción de veces que la respuesta correcta está entre las cinco primeras |
| `lr` | La velocidad de aprendizaje de la cabeza en esa época |
| `best_val_acc` | La mejor exactitud de validación de toda la corrida |
| `best_epoch` | En qué época se logró ese mejor resultado |
| `epochs_run` | Cuántas épocas se ejecutaron de verdad |
| `device` | En qué dispositivo se calculó |
| `num_params` | Cuántos números aprendibles tiene el modelo |
| `segundos` | Cuánto duró la corrida |

### La red de seguridad de num_workers

```python
def _es_error_de_workers(exc: BaseException) -> bool:
    """True si la excepcion huele a fallo de los procesos hijo del DataLoader.

    En el sandbox de Windows de este entorno los workers no pueden abrir *named
    pipes* y revientan con ``PermissionError [WinError 5]``; en Linux/Colab
    ``num_workers > 0`` funciona bien. En vez de abortar un entrenamiento ya
    empezado, `train` reintenta UNA vez la etapa con ``num_workers=0``.
    """
    if not isinstance(exc, (OSError, RuntimeError, EOFError, MemoryError)):
        return False
    if isinstance(exc, PermissionError):  # el caso observado: WinError 5
        return True
    texto = f"{type(exc).__name__}: {exc}".lower()
    return any(pista in texto for pista in _PISTAS_WORKERS)
```

```python
        try:
            return _etapa(etapa, epoch)
        except Exception as exc:  # noqa: BLE001 - se re-lanza si no es de workers
            if num_workers_efectivo <= 0 or not _es_error_de_workers(exc):
                raise
            mensaje = (
                f"La etapa '{etapa}' fallo con num_workers={num_workers_efectivo} "
                f"({type(exc).__name__}: {exc}). Se reintenta una sola vez con "
                "num_workers=0 (en Colab/Linux los workers funcionan; en el sandbox "
                "de Windows no pueden abrir named pipes)."
            )
            print("Aviso: " + mensaje)
            warnings.warn(mensaje, RuntimeWarning, stacklevel=2)

            data_cfg = replace(cfg, num_workers=0)
            train_dl, val_dl, _test_dl, class_names = build_dataloaders(data_cfg)
            num_workers_efectivo = 0
            return _etapa(etapa, epoch)  # unico reintento: si falla, propaga
```

- `num_workers` es el número de **procesos auxiliares** que van preparando las fotos (leerlas del disco, recortarlas, girarlas) mientras la tarjeta gráfica calcula con el lote anterior. Con ayudantes, la GPU no espera y todo va más rápido.
- El problema es que esos ayudantes necesitan un mecanismo del sistema operativo llamado **named pipe**. En algunos entornos de Windows restringidos no pueden abrirlo y revientan con un error de "acceso denegado". Es un fallo del entorno, no del código.
- En vez de abortar el entrenamiento a mitad, el sistema hace lo siguiente: detecta que el fallo huele a los procesos auxiliares, avisa con un mensaje visible, reconstruye los datos con `num_workers=0` (es decir, sin ayudantes: el programa principal prepara las fotos una a una) y repite **una sola vez** la etapa que falló. Si vuelve a fallar, entonces sí se detiene y muestra el error.
- **Por qué se añadió**: cuando la avería ocurría en la época 12 de 25, se perdía todo el trabajo hecho, que son minutos de GPU pagados o racionados. Esta red de seguridad convierte un desastre en una simple advertencia. En Linux y en la nube los ayudantes funcionan bien y la red de seguridad nunca se activa, así que allí no se pierde velocidad.

### La interfaz de línea de comandos

Una **interfaz de línea de comandos** (en inglés, **command line interface**) es la forma de dar órdenes escribiendo texto en lugar de pulsar botones. Es útil porque deja escrito exactamente con qué ajustes se hizo cada corrida, y ese texto se puede copiar y repetir.

```python
    parser.add_argument("--epochs", type=int, default=None, help="numero de epocas")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size", help="tamano de lote")
    parser.add_argument("--backbone", type=str, default=None, help="efficientnet_b0 | resnet50 | mobilenet_v3_large")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="dispositivo")
    parser.add_argument("--debug", action="store_true", help="subconjunto pequeno (smoke test)")
```

- La orden se escribe así: `python -m src.train --epochs 5 --debug`. La parte `python -m src.train` significa "ejecuta el archivo de entrenamiento como un programa", y lo que sigue son las opciones.
- Cada opción tiene un valor por defecto, así que escribir `python -m src.train` a secas ya entrena con la receta completa del proyecto.
- `--debug` es el **modo de prueba rápida**: usa muy pocas fotos por especie y ajusta la configuración a dos épocas sin calentamiento ni precisión mixta, para comprobar en un minuto que todo funciona. Además, cambia la carpeta de salida a `resultados/debug` para no pisar los resultados definitivos que ya están guardados.

| Opción | Qué controla |
|---|---|
| `--epochs N` | Cuántas pasadas completas por las fotos |
| `--batch-size N` | Cuántas imágenes por lote |
| `--backbone B` | Qué extractor de rasgos se usa |
| `--data-root D` | En qué carpeta están las fotos |
| `--out-dir O` | Dónde se guardan los resultados |
| `--device auto\|cuda\|cpu` | Dónde se calcula |
| `--num-workers N` | Cuántos procesos auxiliares preparan las fotos |
| `--debug` | Prueba rápida con pocas fotos |
| `--seed N` | El punto de partida de los números al azar |

### Los archivos que deja la corrida

- `best.pt`: el modelo ganador, en el formato de checkpoint del proyecto.
- `history.json`: el acta con las 13 claves de la corrida.
- `config_usada.json`: la receta exacta que se empleó, para poder repetirla.
- `figuras/curvas.png`: tres gráficas con la evolución del error, del acierto y de la velocidad de aprendizaje.

## Por qué se hizo así

- **Primero se congela el backbone y después se descongela.** Con 3 épocas congeladas, la cabeza (que empieza respondiendo al azar) aprende a leer los rasgos del extractor sin estropearlos. Solo entonces se permite que el extractor también se adapte a las flores. Si se hubiera descongelado desde el primer minuto, las correcciones llegarían con información sin sentido y borrarían el conocimiento previo, que es justo el tesoro que se quiere aprovechar.
- **El mejor modelo se guarda por exactitud de validación, no por el error de entrenamiento.** El rendimiento sobre las fotos de estudio siempre parece mejor de lo que es, porque el modelo puede memorizarlas. La exactitud de validación se mide sobre fotos que nunca vio, así que es la única nota que dice la verdad. Guardar por entrenamiento sería guardar al mejor memorizador, no al mejor clasificador.
- **La semilla está fija en 42.** Los números al azar intervienen en varias partes (el orden en que se reparten las fotos, los valores iniciales de la cabeza, el apagado del dropout). Si cambiaran en cada corrida, comparar dos experimentos sería imposible: no se sabría si la mejora vino del cambio que se hizo o de la casualidad. Con la semilla fija, cualquiera que repita la misma orden sobre los mismos datos obtiene el mismo resultado, y así el informe se puede comprobar.
- **Se guarda un acta de la corrida y no solo el modelo.** El historial es lo que permite dibujar las curvas, detectar que algo va mal (por ejemplo, que el error de validación empieza a subir mientras el de entrenamiento baja) y explicar en el informe de dónde sale cada número. Un modelo sin su acta no se puede defender.
- **El planificador calienta y luego enfría en lugar de mantener una velocidad fija.** Los primeros pasos son los más peligrosos, porque el modelo todavía no tiene ningún criterio; y los últimos son los más delicados, porque solo queda afinar. Subir al principio y bajar al final es la forma de aprovechar las dos situaciones.
- **La parada temprana y el suavizado de etiquetas combaten el mismo enemigo por dos caminos distintos**: que el modelo se aprenda las 1020 fotos de memoria en lugar de aprender a reconocer flores. El suavizado le quita la tentación de estar seguro al cien por cien, y la parada temprana corta el entrenamiento en cuanto deja de mejorar en el examen sorpresa.
- **La precisión mixta solo se activa en tarjeta gráfica porque solo ahí tiene sentido.** Aceptar un riesgo de precisión a cambio de velocidad es un buen negocio cuando el cálculo está en la GPU; en el procesador no hay ninguna ganancia que justifique ese riesgo, así que se apaga y se avisa.
- **El modo de prueba rápida escribe en su propia carpeta.** Un modo pensado para comprobar que todo arranca no puede sobrescribir el resultado definitivo que costó 25 minutos de GPU: si lo hiciera, una simple prueba de un minuto borraría el trabajo bueno.
