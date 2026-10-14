# Verificacion adversarial del proyecto "IA flores" (SA-10)

> Documento propiedad de **SA-10 (verificador adversarial)**. Su funcion no es escribir
> funcionalidad, sino **intentar refutar** que el proyecto funciona y dejar constancia escrita
> de lo que aguanta y de lo que no.
>
> **Aviso de concurrencia.** Durante esta auditoria varios archivos cambiaron (SA-05, SA-06,
> SA-08, SA-09 y el orquestador seguian trabajando). Cada veredicto se refiere a la revision
> indicada en la seccion 0.2, y donde un hallazgo quedo resuelto por otro subagente **mientras**
> yo auditaba, se dice explicitamente. Al final del documento (seccion 7) hay un registro de
> todo lo que cambio bajo mis pies.

---

## 0. Entorno, revision auditada y metodo

### 0.1 Entorno de verificacion (peculiaridades de ESTA maquina, no del proyecto)

Todo se ejecuto desde la raiz del proyecto con el venv del proyecto y las variables de entorno
de `contrato interno` §1:

```powershell
$env:TMP = (Resolve-Path .pytmp).Path; $env:TEMP = $env:TMP
$env:MPLCONFIGDIR = (Resolve-Path .pytmp).Path
.\.venv\Scripts\python.exe <comando>
```

Tres peculiaridades del entorno de verificacion (no del codigo) que condicionaron el metodo:

1. **`TORCH_HOME` no es escribible por defecto.** El cache por defecto de torch (una carpeta
   bajo el perfil del usuario, `<unidad_del_autor>:\...\.cache\torch`, redactada aqui a proposito)
   lo bloquea el sandbox, asi que `build_model(pretrained=True)`
   cae a su fallback documentado (`warnings.warn` + pesos aleatorios). Para las corridas de esta
   auditoria se exporto `TORCH_HOME=<proyecto>\.piptmp\torchhome` **antes de importar torch**
   (contiene los pesos de `efficientnet_b0` y `mobilenet_v3_large`; `resnet50` se descargo
   durante la auditoria: 97,8 MB, sin errores). En Colab `~/.cache/torch` es escribible y esto no
   aplica.
   **Prueba de que es del entorno y no del codigo:** con `TORCH_HOME` en un directorio
   escribible ninguna corrida emitio el aviso de `pretrained` y la perdida bajo de
   `ln(102) = 4.625`; sin el, el aviso aparece. Ver seccion 2.8.e.
2. **`nbclient`/`jupyter_client` no pueden arrancar un kernel aqui.** Evidencia literal
   (`.pytmp/sa10_probe.py`):

   ```
   == 2. nbclient puede arrancar un kernel ==
      FALLO nbclient: error (5, 'SetFileSecurity', 'Acceso denegado.')
   ```

   Por eso el notebook se ejecuto con un emulador propio celda a celda (seccion 2.4). Es una
   limitacion del entorno de verificacion, **no** del proyecto.
3. **Fuera de Jupyter, `plt.show()` bloquea.** El primer intento de ejecutar el notebook quedo
   colgado en la celda 12 (`plt.show()` con backend interactivo abre una ventana y espera). En
   Colab el backend es *inline* y no bloquea: es un artefacto de mi emulador. La ejecucion
   valida se hizo con `MPLBACKEND=Agg` y la celda 12 solo emitio
   `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown`.

### 0.2 Revision auditada (hashes SHA-256, 12 primeros digitos)

Hashes tomados **al cerrar la auditoria** (revision final):

| Archivo | SHA-256 | Bytes |
|---|---|---|
| `src/config.py` | `7E091D7CAA8D` | 3487 |
| `src/utils.py` | `F14831515CD9` | 3082 |
| `src/data.py` | `239072B818F5` | 17029 |
| `src/model.py` | `42FF890C0092` | 15348 |
| `src/train.py` | `4C1AA61E7C4C` | 26031 |
| `src/evaluate.py` | `22F5FAC257B7` | 28388 |
| `src/predict.py` | `F7CE8AACF653` | 13465 |
| `src/app.py` | `DB9D831F58A0` | 15241 |
| `src/__init__.py` | `B67ABE3EC8C9` | 93 |
| `tests/conftest.py` | `243354390676` | 1884 |
| `tests/test_data.py` | `F4D39692CFAF` | 22288 |
| `tests/test_model.py` | `162552495937` | 10867 |
| `tests/test_train.py` | `77FFCE08F370` | 23547 |
| `tests/test_eval_predict.py` | `CEE3BEDF8093` | 22613 |
| `tests/test_smoke.py` | `420CAFDE521C` | 4889 |
| `notebooks/flores_colab.ipynb` | `4B9E0D8C75EA` | 56447 |
| contrato interno (retirado del repositorio) | `ADD6776FA45F` | 15895 (parcheado por mi: §1 y regla 5 de §0) |
| `README.md` | `3CDC58164B7E` | 15522 (parcheado por mi) |
| `INFORME.md` | `CF043365F164` | 52853 |
| `PLAN_TRABAJO.md` | `B4704BCB4252` | 11421 |
| `requirements.txt` | `A5BD90A0BE7A` | 696 |
| `pytest.ini` | `F5B476564D65` | 337 |
| `empaquetar_colab.py` | `3D5F4A8F761C` | 3368 |
| `.gitignore` | `F22056EAA62D` | 221 |

*(`_sa09_build.py`, un script de 43 KB que habia en la raiz al empezar la auditoria y que
regeneraba el notebook, fue eliminado por otro subagente durante la auditoria; ver seccion 7.
`docs/VERIFICACION.md` es este mismo documento: su hash cambia con cada edicion, asi que no se
lista aqui. Los hashes de `src/`, `tests/` y `notebooks/` de esta tabla son los de la revision
exacta sobre la que se ejecuto el `pytest` final: 78 passed.)*

### 0.3 Que se hizo

1. Lectura completa de `contrato interno`, `PLAN_TRABAJO.md`, `README.md`, `INFORME.md`,
   `empaquetar_colab.py`, `notebooks/flores_colab.ipynb`, los 8 modulos de `src/` y los 5 (luego
   6) archivos de `tests/`, ademas de `pytest.ini`, `.gitignore` y `assets/cat_to_name.json`.
2. Suite completa 3 veces (con 73, 77 y 78 casos, segun iban anadiendose tests) y, la ultima,
   con `IA_FLORES_SLOW=1`.
3. **Corrida end-to-end propia desde cero** con otra configuracion (resnet50 + semilla 7) y una
   corrida con el **dataset completo** (mobilenet, 2 epocas, `num_workers=2`), mas la evaluacion
   independiente de un checkpoint entrenado sobre las 6149 imagenes oficiales de test.
4. Re-verificacion **independiente** de fuga de datos (ficheros oficiales `setid.mat` +
   `imagelabels.mat`, MD5 de las 8189 imagenes, subconjuntos de debug, determinismo de los
   transforms).
5. Comprobacion independiente de si **cada checkpoint del repo** uso de verdad pesos ImageNet.
6. Ejecucion del notebook completo en modo debug a una copia temporal
   (`.pytmp/verif_notebook.ipynb`; `notebooks/` intacto) y comprobacion de que el notebook no
   cambiaba durante la ejecucion.
7. Auditoria de rutas absolutas, `.cuda()`/GPU, `requirements.txt`, CLIs (`--help` y sin
   checkpoint), coherencia documento <-> codigo, literatura citada, versiones instaladas y del
   ZIP de entrega (regenerado y auditado).

---

## 1. Veredicto de los 10 puntos

| # | Afirmacion a refutar | Veredicto |
|---|---|---|
| 1 | `pytest tests -q` pasa entero (dicen "66 passed + 1 skipped") y ningun test pasa por casualidad | **PARCIALMENTE REFUTADO.** La suite pasa entera, pero el recuento literal **no es ese**: `78 passed` (con `IA_FLORES_SLOW=1`) o `76 passed, 1 skipped` sin la variable. Los tests comprueban cosas reales, pero **`src/app.py` no tiene ningun test** y hay 3 puntos donde un fallo real acabaria en `skip` |
| 2 | El pipeline corre de punta a punta | **CONFIRMADO.** Corrida propia resnet50+semilla 7 (train -> evaluate -> predict -> build_app) y corrida de 2 epocas con el dataset completo; ademas, evaluacion de 6149 imagenes: `top1=37,8 %`, `top5=67,4 %` |
| 3 | No hay fuga de datos: splits disjuntos y val/test sin augmentacion | **PARCIALMENTE REFUTADO.** Splits disjuntos por indice y por ruta (0 intersecciones), val/test deterministas (sin augmentacion): confirmado. Pero hay **4 pares de imagenes byte-identicas (MD5) y 2 cruzan splits** (train<->test y val<->test): refuta el "sin duplicados globales" |
| 4 | El notebook se ejecuta de punta a punta en modo debug | **CONFIRMADO** (13/13 celdas de codigo, 102,1 s, sin errores, hash del notebook igual antes y despues). No verificable como notebook real de Jupyter/Colab en este sandbox |
| 5 | No hay rutas absolutas de Windows en lo entregado | **REFUTADO** en un punto: `contrato interno` linea 34 (que viajaba en el ZIP) mas la propia norma que citaba el patron. **Ambos parcheados**: hoy 0 coincidencias en todo lo entregado, dentro y fuera del ZIP. `src/`, `tests/`, `notebooks/`, el resto de `.md`, los `.py` de la raiz y `requirements.txt`: limpios |
| 6 | No hay `.cuda()` duro, todo corre en CPU y la rama CUDA/AMP esta bien escrita | **CONFIRMADO** en lo verificable en CPU; la rama CUDA/AMP solo es auditable por lectura (no hay GPU utilizable) |
| 7 | `requirements.txt` correcto y suficiente; Colab no reinstala torch; `scipy` necesario | **PARCIALMENTE REFUTADO** cuando empece (faltaba `nbformat` y `gradio>=4.44` no reflejaba la API de Gradio 6); **ambos puntos fueron corregidos por el orquestador durante la auditoria** (`gradio>=6.0`, `nbformat>=5.9`). Colab no reinstala torch y `scipy` esta: correcto |
| 8 | Los numeros del INFORME/README son honestos y coinciden con `src/config.py` | **CONFIRMADO en lo esencial**: ninguna metrica inventada, cifras de literatura rotuladas y verificadas contra la fuente, los 21 hiperparametros coinciden literalmente. Incoherencias: "ultimos bloques" en `README.md` (parcheado), una descarga de `cat_to_name.json` que da 404, y **ningun artefacto registra si los pesos preentrenados se cargaron de verdad** (hallazgo propio, seccion 2.8.e) |
| 9 | La demo no revienta sin checkpoint y las 4 CLIs responden a `--help` | **CONFIRMADO** |
| 10 | El ZIP contiene todo lo necesario y nada peligroso | **CONFIRMADO** tras regenerarlo (seccion 2.10): 25 archivos, sin `data/`, `resultados/`, `__pycache__`, `.pt` ni rutas absolutas; `docs/`, `tests/test_smoke.py` y el notebook al dia |

---

## 2. Evidencia por punto

### 2.1 Punto 1 - La suite de tests

**Comando final (revision con `tests/test_smoke.py` ya creado y con los tests lentos activados):**

```powershell
$env:IA_FLORES_SLOW = "1"
.\.venv\Scripts\python.exe -m pytest tests -q
```

**Salida literal:**

```
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 83.93s (0:01:23)
```

**Ejecuciones anteriores, para dejar constancia de que el recuento se movio durante la
auditoria (los tests se estaban anadiendo en paralelo):**

```
# revision con 4 tests menos (test_train.py y test_eval_predict.py sin los casos de num_workers)
72 passed, 1 skipped in 36.57s
SKIPPED [1] tests\test_train.py:452: lento (dataset real): exporta IA_FLORES_SLOW=1 para ejecutarlo

# misma suite ya con esos 4 tests nuevos, sin IA_FLORES_SLOW
76 passed, 1 skipped in 112.12s (0:01:52)

# recuento por archivo en ese momento
   17 tests/test_data.py
   22 tests/test_eval_predict.py
   22 tests/test_model.py
   16 tests/test_train.py
77 tests collected in 3.96s
```

Conclusion: **la suite pasa entera** (78/78 con los tests lentos activados), pero la cifra
"66 passed + 1 skipped" que circula como afirmacion es **falsa en todas las revisiones que he
ejecutado** (era 73, luego 77, ahora 78 casos). No es una mentira: es un dato que caduco porque
los tests seguian creciendo.

**Tests debiles encontrados (revision uno a uno de los 78):**

| Ubicacion | Debilidad | Severidad |
|---|---|---|
| `tests/` (ningun archivo) | **`src/app.py` no tiene ni un solo test**: buscar `app`, `build_app` o `gradio` en `tests/` no devuelve ninguna referencia. El criterio de aceptacion de `contrato interno` §4 para `app.py` no esta cubierto por la suite (lo he verificado yo a mano: seccion 2.9). `tests/test_smoke.py`, en cambio, **ya existe** (creado durante la auditoria) y si ejercita `data -> model -> train -> evaluate -> predict` | media |
| `tests/test_data.py:509` | `_build_or_skip` captura `except Exception` y hace `pytest.skip(...)`: **cualquier fallo real** de `build_dataloaders` (firma rota, dataset corrupto, error de torchvision) se convierte en "skipped" en lugar de "failed". En mi corrida los dos tests lentos se ejecutaron y pasaron, pero el mecanismo oculta fallos | media |
| `tests/test_data.py:38-43,493` | `pytest.importorskip("torch"/"torchvision"/"scipy")`: si falta una dependencia, los tests se saltan en silencio y la suite "pasa" sin probar nada | baja |
| `tests/test_eval_predict.py:108-191`, `tests/test_smoke.py:99-107` | Las metricas se comprueban **estructuralmente** (claves, rangos, `top5 >= top1`, numero de filas del CSV), pero **nada** verifica que `top1` coincida con el recuento de `correcto=True` del CSV ni que la matriz de confusion cuadre con las predicciones: una aritmetica de metricas equivocada pasaria los tests. Yo lo he cruzado a mano en dos corridas (seccion 2.2) y coincide al ultimo digito | baja |
| `tests/test_data.py:229-237` | `test_build_transforms_train_is_random` depende de aleatoriedad real (`assert not torch.equal(dataset[0][0], dataset[0][0])`): assert probabilistico (riesgo despreciable) | baja |
| `tests/test_data.py:250` | Los tests offline sustituyen `_flowers102_class` por `_FakeFlowers102`, que imita atributos **privados** de torchvision (`_image_files`, `_labels`): si torchvision los renombrase, el falso seguiria pasando y solo lo detectarian los dos tests lentos con el dataset real | baja |
| `tests/` (ningun archivo) | Ningun test ejecuta el CLI como modulo (`python -m src.train ...`): se llama a `main([...])` directamente, asi que el `raise SystemExit(main())` y la resolucion de paquetes no estan cubiertos. Lo he verificado por subprocess (seccion 2.9) | baja |

**Lo que SI es solido** (importante para no exagerar el veredicto): `test_train.py` verifica la
secuencia exacta de congelado/descongelado (`assert llamadas == [False, False, False, True, True, True]`),
la forma del warmup+coseno con valores numericos, el early stopping con una serie determinista, que
**no se construye `GradScaler` en CPU** (con un espia), la reproducibilidad con la misma semilla y
el reintento con `num_workers=0`. Los tests de checkpoint comparan salidas con `max diff == 0.0` y
verifican las 8 claves del contrato **en el archivo guardado**. Los de `predict` cubren monotonia,
recorte de `topk`, umbral `min_conf`, la cache `lru_cache` (contando llamadas) y 9 formatos de
entrada. `test_smoke.py` monta un dataset sintetico en disco y recorre `data -> model -> train ->
evaluate -> predict` con artefactos reales y comprobacion del umbral.

### 2.2 Punto 2 - Corrida end-to-end propia (otra configuracion)

Configuracion elegida **a proposito distinta** a la que ya se habia usado
(`efficientnet_b0`/`mobilenet_v3_large`, semilla 42):

* `backbone="resnet50"`, `seed=7`, `debug=True` (subconjunto real: 4 imagenes por clase),
  `epochs=2`, `batch_size=32`, `device="cpu"`, `pretrained=True`,
  `out_dir="resultados/verif_sa10"`; script `.pytmp/sa10_e2e.py`.

**Salida literal (recortada a lo util):**

```
Entrenamiento Flowers-102
  dispositivo : cpu
  backbone    : resnet50 (pretrained=True, dropout=0.3)
  epocas      : 2 | batch: 32 | img: 224 | clases: 102
  lr cabeza/backbone: 0.0003/3e-05 | warmup: 0 | freeze: 0 | amp: False
  debug       : True | salida: resultados\verif_sa10
  lotes: train=13 val=13 | imagenes: train=408 val=408 | nombres de clase: 102 | num_workers: 0
[  1/2] backbone entrenando | train loss 4.6196 acc 0.0098 | val loss 4.5352 acc 0.0490 top5 0.1740 | lr 3.000e-04 *mejor
[  2/2] backbone entrenando | train loss 4.4840 acc 0.1201 | val loss 4.4712 acc 0.1324 top5 0.3162 | lr 1.500e-04 *mejor
Listo en 436.8 s (7.3 min) | epocas ejecutadas: 2/2
Mejor val_acc: 0.1324 (epoca 2) |  Parametros: 23,717,030
[SA-10] avisos durante el entrenamiento: ninguno

[evaluate] split=test imagenes=408 top1=0.1348 top5=0.3137 macro_f1=0.1042 (73.46s)

top1 en metrics.json      : 0.13480392156862744
top1 recalculado del CSV  : 0.13480392156862744     -> coinciden: True
num_images == filas CSV   : True
top5 >= top1              : True
imagenes mal clasificadas por el CSV : 353
etiquetas reales distintas            : 102

imagen: image_00001.jpg   (top-5)
  1. bougainvillea 0.0144 | 2. yellow iris 0.0144 | 3. great masterwort 0.0134 | ...
ruido aleatorio (min_conf=0.25):
  no reconocida / fuera del catálogo  0.0178

build_app(checkpoint real)    : Blocks | launch: True
build_app(checkpoint ausente) : Blocks | launch: True
predecir(imagen real) -> ['no reconocida / fuera del catálogo']  (la confianza no llega a 0,25)
predecir(ckpt ausente) -> {} | contiene 'Entrena primero': True
[SA-10] FIN
```

Lectura: con 4 imagenes por clase y 2 epocas el modelo **ya aprende** (top-1 13,5 % frente al
0,98 % del azar; top-5 31,4 % frente al 4,9 %), las probabilidades son bajas y por eso el umbral
`min_conf` responde "no reconocida". Sin avisos de pesos preentrenados y con la aritmetica de
metricas cuadrando con el CSV.

**Corrida adicional con el dataset COMPLETO y `num_workers=2`** (mobilenet_v3_large, semilla 42,
2 epocas, backbone congelado; `.pytmp/sa10_nw2.py`), que ademas prueba la degradacion automatica
anadida a `src/train.py`:

```
  lotes: train=32 val=32 | imagenes: train=1020 val=1020 | nombres de clase: 102 | num_workers: 2
  AMP solicitado pero desactivado: el device no es CUDA (CPU).
Aviso: La etapa 'train' fallo con num_workers=2 (PermissionError: [WinError 5] Acceso denegado).
       Se reintenta una sola vez con num_workers=0 (en Colab/Linux los workers funcionan; en el
       sandbox de Windows no pueden abrir named pipes).
[  1/2] backbone congelado | train loss 4.6378 acc 0.0118 | val loss 4.6283 acc 0.0078 top5 0.0539 | lr 3.000e-05 *mejor
[  2/2] backbone congelado | train loss 4.5585 acc 0.0176 | val loss 4.4907 acc 0.0510 top5 0.1892 | lr 1.650e-04 *mejor
Listo en 347.4 s (5.8 min) | epocas ejecutadas: 2/2
[SA-10] avisos: ["RuntimeWarning: La etapa 'train' fallo con num_workers=2 (PermissionError: ..."]
```

Dos comprobaciones colaterales valiosas:

* **Warmup + coseno correctos**: con `lr=3e-4` y `warmup_epochs=2`, la epoca 1 usa `3.0e-05`
  (10 %) y la epoca 2 `1.65e-04` (55 %), exactamente como describe `INFORME.md` §7.1, y con
  `freeze_epochs=3` el backbone permanece congelado.
* La degradacion `num_workers=2 -> 0` **funciona y no pierde el entrenamiento**: avisa una vez,
  reconstruye los loaders y termina. Las instrucciones de `README.md` §3 funcionan tal cual en
  Windows pese al problema de los workers.

**Metricas REALES sobre el split oficial completo de test (6149 imagenes)** evaluando un
checkpoint entrenado con el dataset completo (4 epocas, mobilenet, `resultados/learn/best.pt`),
con `.pytmp/sa10_eval_learn.py` y `num_workers=0`:

```
[evaluate] split=test imagenes=6149 top1=0.3778 top5=0.6741 macro_f1=0.3536 (378.31s)
[SA-10] top1 recalculado del CSV: 0.377785 (2323/6149) | coincide: True
[SA-10] num_images == filas: True
[SA-10] top_confusions[:5]: [['petunia','wallflower',58], ['cyclamen','magnolia',47],
                             ['rose','camellia',45], ['watercress','wallflower',38],
                             ['water lily','silverbush',37]]
[SA-10] peores F1: [('sweet pea', 0.0), ('giant white arum lily', 0.0),
                    ('prince of wales feathers', 0.0), ('artichoke', 0.0), ('mexican aster', 0.0)]
[SA-10] clases con support>0: 102 | suma de supports: 6149
```

Estas cifras (37,8 % top-1 / 67,4 % top-5 / macro-F1 0,354 con solo 4 epocas, de las cuales 3
con el backbone congelado) **no estan en el informe** y no deben confundirse con el resultado
final del proyecto: sirven como evidencia de que el pipeline mide de verdad y como suelo de
partida. Dos hipotesis del `INFORME.md` §9.3 quedan corroboradas por estos datos reales:
macro-F1 (0,354) < top-1 (0,378) —desempeno desigual entre clases, con 5 especies en F1 = 0— y
las confusiones se concentran en pares visualmente parecidos (petunia->wallflower,
rose->camellia), no en pares aleatorios.

### 2.3 Punto 3 - Fuga de datos (verificacion independiente)

Metodo propio (`.pytmp/sa10_leak.py` y `.pytmp/sa10_dups.py`): leer `setid.mat`/`imagelabels.mat`
con `scipy.io`, recomponer los splits por indice, contrastarlos con lo que entrega torchvision,
calcular **MD5 de las 8189 imagenes**, revisar los subconjuntos de debug y la aleatoriedad de los
transforms.

**Salida literal (recortada):**

```
claves setid.mat: ['trnid', 'tstid', 'valid']
  trnid :  1020 imagenes | ids 28..8177
  valid :  1020 imagenes | ids 17..8187
  tstid :  6149 imagenes | ids 1..8189
  imagenes totales en imagelabels.mat: 8189 | clases: 102
 |trnid & valid| = 0      |trnid & tstid| = 0      |valid & tstid| = 0
 |union| = 8189 == 8189 : True
  ids duplicados dentro de cada split: {'trnid': 0, 'valid': 0, 'tstid': 0}
train: ficheros==setid:True etiquetas==imagelabels-1:True
val  : ficheros==setid:True etiquetas==imagelabels-1:True
test : ficheros==setid:True etiquetas==imagelabels-1:True
 |train & val|  (rutas) = 0      |train & test| (rutas) = 0      |val & test|   (rutas) = 0
== 4. MD5 de las 8189 imagenes ==
  ficheros hasheados: 8189 | hashes duplicados (global): 4
== 5. Subconjuntos de debug ==
  tamanos: 408 408 408 | nombres: 102 | clases cubiertas en debug: 102
 |train & val| = 0 | |train & test| = 0 | |val & test| = 0
== 6. Transforms ==
  transforms train=True : identicos en dos aplicaciones -> False
  transforms train=False: identicos en dos aplicaciones -> True
```

**Detalle de los duplicados exactos:**

```
  md5=454bab361c -> image_07307.jpg [train] clase 94 | image_07309.jpg [test] clase 94  => splits=['train','test']
  md5=28dc3ced5b -> image_07323.jpg [test]  clase 94 | image_07328.jpg [test] clase 94  => splits=['test']
  md5=8a645f3ded -> image_07885.jpg [test]  clase 99 | image_07886.jpg [test] clase 99  => splits=['test']
  md5=d5de43837c -> image_08067.jpg [test]  clase 31 | image_08077.jpg [val]  clase 31  => splits=['test','val']
grupos de duplicados exactos: 4 | grupos que cruzan splits (fuga exacta): 2
```

Conclusiones:

* **CONFIRMADO**: splits oficiales reconstruidos y disjuntos por indice y por ruta, union
  completa de 8189, sin indices duplicados, sin rutas compartidas en los subconjuntos de debug
  (que cubren las 102 clases), y val/test **sin augmentacion aleatoria** (dos aplicaciones del
  transform dan el mismo tensor exacto) mientras train si es aleatorio. Las etiquetas de
  torchvision son exactamente `imagelabels - 1` en el orden de `setid.mat`, asi que el mapeo
  `class_names[i] <-> etiqueta i` del contrato es correcto (comprobado ademas contra los nombres:
  `torchvision.datasets.Flowers102.classes` y `assets/cat_to_name.json` coinciden al 100 %, 102
  nombres, claves 0-based).
* **REFUTADO**: "sin duplicados globales" y la afirmacion del `INFORME.md` §5.3 (revision
  inicial) de que las particiones oficiales separan las imagenes "por planta fotografiada". Hay
  **4 pares de imagenes byte-identicas y 2 cruzan splits**. Es un defecto del **dataset oficial**
  (la pagina de Oxford solo publica listas de indices en `setid.mat`), no del codigo, y su
  impacto numerico es de 2 imagenes sobre 8189 (~0,02 %), irrelevante para las metricas
  agregadas.
  **Estado**: el `INFORME.md` §5.3 fue reescrito durante la auditoria (hash `CF043365F164`) y ya
  documenta literalmente "cuatro pares de imágenes byte a byte idénticas, de los cuales dos
  cruzan particiones (train↔test y val↔test)" y que eso "invalida cualquier afirmación de
  separación estricta 'por planta fotografiada'". Hallazgo **cerrado por otro subagente**.

### 2.4 Punto 4 - Ejecucion del notebook

`notebooks/flores_colab.ipynb` **no se toco** (se verifico el hash antes y despues: `4B9E0D8C75EA`
en ambos casos). Se copio a `.pytmp/verif_notebook.ipynb` y se ejecuto con
`.pytmp/sa10_nbexec.py`. Unica modificacion a la copia: `out_dir="resultados/debug"` ->
`out_dir="resultados/verif_sa10_nb"` en el modo debug de la celda 10, para no pisar el
`resultados/debug/` que otro subagente estaba generando en paralelo. `MODO_DEBUG = True` (su
valor por defecto).

**Salida literal (resumen por celda; el log completo esta en `.pytmp/sa10_nbexec_log2.txt`):**

```
sha256(notebooks/flores_colab.ipynb) = 4b9e0d8c75ea3fc2f8e8610dfb0132e6096f00558b18eb4d9184504f3ca35113
celdas con out_dir cambiado (para no pisar resultados/debug de otros): 1

### CELDA 2 (entorno) OK en 7.1s
torch y torchvision ya estan disponibles: no se tocan (regla de Colab).
Todo lo necesario ya estaba instalado: no se llama a pip.
torch: 2.14.1+cpu | torchvision: 0.29.1+cpu | CUDA disponible: False
### CELDA 4 (Drive) OK
Fuera de Colab (o sin google.colab): se omite el montaje de Drive.
### CELDA 6 (codigo del proyecto) OK
MODO_CODIGO='auto': proyecto detectado en ...\IA flores  |    src/config.py : OK
### CELDA 8 (imports) OK
### CELDA 10 (config) OK
Modo: DEBUG (subconjunto) | Dispositivo: cpu | Epocas: 2 | batch: 32 | img: 224 | num_workers: 0
### CELDA 12 (entrenamiento) OK en 76.9s
[  1/2] backbone entrenando | train loss 4.6504 acc 0.0245 | val loss 4.5279 acc 0.0490 top5 0.1520 | lr 3.000e-04 *mejor
[  2/2] backbone entrenando | train loss 4.4869 acc 0.0588 | val loss 4.4505 acc 0.1005 top5 0.2721 | lr 1.500e-04 *mejor
Listo en 76.1 s (1.3 min) | Parametros: 4,138,210
Entrenamiento terminado en 1.3 min | Mejor val_acc: 0.10049019607843138 (epoca 2)
Sin Drive montado: se omite el respaldo (modo local).
<celda 12>:68: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
### CELDA 14 (evaluacion sobre test) OK en 9.3s
[evaluate] split=test imagenes=408 top1=0.0760 top5=0.2377 macro_f1=0.0563 (9.28s)
### CELDA 15 (metrics.json + matriz + peores clases) OK
Top-1: 7.60% | Top-5: 23.77% | macro-F1: 0.0563 | imagenes: 408 | 9.3 s
Figura: resultados\verif_sa10_nb\figuras\confusion_matrix.png
### CELDA 17 (resumen para el informe) OK
(imprime el bloque Markdown entre ---8<--- y --->8--- y lo guarda en resumen_informe.md)
### CELDA 19 (prediccion suelta) OK
Imagen de ejemplo: data\flowers-102\jpg\image_00001.jpg
 1. no reconocida / fuera del catálogo           1.72%
### CELDA 20 (demo Gradio) OK en 0.7s
Interfaz construida: Blocks | tiene launch: True
* Running on local URL:  http://127.0.0.1:7860
Demo lanzada.  URL local: http://127.0.0.1:7860/
### CELDA 22 (parar la demo) OK
Closing server running on port: 7860 | Demo detenida.
### CELDA 24 (ZIP de resultados + cache en Drive) OK en 7.4s
ZIP creado: resultados_flores.zip (170.5 MB)
Cache del dataset: sin Drive, se omite (el dataset se queda en la sesion).
=== Artefactos de la corrida ===
  checkpoint OK | historial OK | config usada OK | metricas OK | resumen informe OK
  curvas OK | matriz de confusion OK | top confusiones OK | predicciones OK | ZIP OK
### CELDA 25 (markdown) -

RESULTADO: notebook ejecutado de punta a punta sin errores en 102.1 s
sha256(notebooks/flores_colab.ipynb) DESPUES = 4b9e0d8c75ea3fc2f8e8610dfb0132e6096f00558b18eb4d9184504f3ca35113
```

**Reproducibilidad comprobada**: la ejecucion se hizo **dos veces** con el mismo notebook, la
misma semilla y el mismo proceso (la primera quedo bloqueada en `plt.show()` despues de entrenar
y evaluar; la segunda se completo). Los valores de la celda 12 son **identicos bit a bit** entre
ambas corridas:

```
  train_loss identico=True [4.650416093714097, 4.486891849368226]
  train_acc  identico=True [0.024509803921568627, 0.058823529411764705]
  val_loss   identico=True [4.52793300853056, 4.450500890320423]
  val_acc    identico=True [0.049019607843137254, 0.10049019607843138]
  val_top5   identico=True [0.15196078431372548, 0.27205882341176474]   # (valores exactos de run 2)
  segundos   run1=336.3 (con contention)   run2=76.1 (sin contention)
```

**Nota metodologica honesta (que es y que no es esta ejecucion).** Es una ejecucion **celda a
celda en un solo proceso Python**, no una ejecucion de Jupyter/Colab con `nbclient` (imposible
aqui: `SetFileSecurity: Acceso denegado`). Ejecuta exactamente el mismo codigo, en el mismo
orden, en el mismo espacio de nombres, y se detiene en el primer error. **No** cubre magics,
`%%capture`, reinicios de kernel, ni las celdas especificas de Colab (`files.upload`,
`files.download`, `drive.mount`, que aqui se omiten por diseno y no fallan). Ademas se forzo
`MPLBACKEND=Agg`: fuera de Jupyter `plt.show()` abre una ventana con el backend interactivo y
**bloquea**, lo que produjo el unico `UserWarning` de toda la ejecucion. En Colab el backend es
*inline* y no bloquea.

### 2.5 Punto 5 - Rutas absolutas de Windows

Busqueda de rutas absolutas de Windows (patrones del tipo letra-de-unidad + dos puntos + barra,
y menciones de la carpeta de usuarios) en **todo** el arbol entregable: una sola coincidencia
real (la linea 18 de `contrato interno` era la propia norma, no una ruta):

```
el contrato interno
Line 34: | Python local de pruebas | C:\...\Python312\python.exe (3.12) |      <- ruta del autor, redactada aqui
```

Esa linea viajaba dentro de `IA_flores_colab.zip`. **Parcheada** (seccion 3). Ademas se
reformulo la propia norma de la linea 18, que citaba el patron literal, para que ni siquiera
aparezca la mencion: hoy un grep de rutas absolutas de Windows (y de la carpeta de usuarios) en
**todo** lo entregado, incluido el interior del ZIP, devuelve **0 coincidencias**. El resto esta
limpio: `src/*.py` (solo la URL de Oxford en `data.py`), `tests/*.py`, `notebooks/*.ipynb`
(comprobado tambien dentro del ZIP: ninguna ruta absoluta), los `.md`
(solo URLs), `requirements.txt`, `pytest.ini`, `.gitignore` y `empaquetar_colab.py`.

### 2.6 Punto 6 - `cuda()` duro, CPU y rama AMP

Todas las apariciones de `cuda` en `src/` son legitimas y ninguna es una llamada dura en el
camino de ejecucion: `utils.py:40` (`manual_seed_all`), `utils.py:55-56` (`is_available`),
`utils.py:70` (`get_device_name`, solo si el device ya es CUDA), `data.py` (`pin_memory`
condicionado a CUDA), `train.py:270` (fallback `torch.cuda.amp.GradScaler` para torch antiguo,
solo alcanzable con CUDA+AMP). **CONFIRMADO: no hay `.cuda()` duro.**

CPU: **toda** la auditoria corrio en CPU (`torch.cuda.is_available() == False`): entrenamiento,
evaluacion de 6149 imagenes, prediccion, demo, tests y ejecucion del notebook.

Rama CUDA/AMP (revision de codigo, no ejecutable aqui):

* `amp_enabled = bool(cfg.amp) and device.type == "cuda"`; en CPU queda `False` y esta
  **verificado con espia** (`tests/test_train.py::test_amp_desactivado_en_cpu_sin_avisos`) que no
  se construye `GradScaler` y que `autocast` se invoca con `enabled=False` sin generar avisos.
* Con AMP activo: `torch.amp.autocast("cuda", enabled=True)` + `torch.amp.GradScaler("cuda")` y
  el ciclo `scaler.scale(loss).backward()` / `scaler.step(optimizer)` / `scaler.update()`. Es el
  patron correcto.
* Los tensores se mueven con `.to(device, non_blocking=True)`; `pin_memory` solo con CUDA.
* **NO VERIFICABLE** aqui: que en FP16 sobre una T4 la perdida no se vaya a `inf/nan` y que el
  `GradScaler` escale como debe.

### 2.7 Punto 7 - `requirements.txt`

**CONFIRMADO:**

* El archivo avisa de que en Colab **no** se ejecute tal cual y de que alli solo hace falta
  `gradio scikit-learn matplotlib` (+`scipy`, que Colab ya trae). El notebook (celda 2)
  implementa exactamente eso: **comprueba** `torch`/`torchvision` y **nunca** los instala
  (`"torch y torchvision ya estan disponibles: no se tocan (regla de Colab)."`), e instala por
  `pip` solo lo que falte de `gradio`, `sklearn`, `matplotlib` y `scipy`.
* `scipy>=1.11` esta presente y es necesaria (`torchvision.datasets.Flowers102` usa
  `scipy.io.loadmat`); `src/data.py` convierte el `ImportError` en un mensaje util.
* Todas las dependencias usadas estan declaradas y las versiones instaladas coinciden con las
  que `contrato interno` §1 declara: torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.2,
  scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, gradio 6.30.0,
  pytest 9.1.1, nbformat 5.11.1, tqdm 4.70.1, Pillow 12.3.0 (verificado modulo a modulo).

**REFUTADO en la revision inicial, ya corregido durante la auditoria:**

* Faltaba `nbformat`, que `contrato interno` §1 lista como instalado y §5 afirma incluido en
  `requirements.txt`. **Ya esta** (`nbformat>=5.9`).
* `gradio>=4.44` permitia instalar versiones contra las que no esta escrito `src/app.py`, que
  documenta haberse escrito contra Gradio 6.30.0 y usa `gr.Blocks(fill_width=True)`,
  `sources=[..., "clipboard"]` y `gr.Label(num_top_classes=...)`. **Ahora es `gradio>=6.0`.**
  Verificado que la version instalada (6.30.0) admite `Blocks(fill_width=...)` y
  `launch(debug=..., prevent_thread_lock=...)`. **NO VERIFICABLE** sin instalar Gradio 4.44/5.x
  (no se hizo para no degradar el venv compartido): a partir de que version exacta existen
  `fill_width` y `sources=["clipboard"]`.

### 2.8 Punto 8 - Honestidad de los numeros y coherencia documento <-> codigo

**a) No hay metricas inventadas (CONFIRMADO).** `INFORME.md` §8 y `README.md` §1 mantienen sus
tablas como pendientes (`_pendiente de la corrida en Colab_` / `_pendiente de tu ejecución_`;
15 marcadores en el informe) y el Apendice A lo explicita: "Este informe no contiene ningun
resultado experimental inventado, estimado ni extrapolado". Los unicos numeros medidos que hay
en `resultados/` **no se citan** como resultados del informe; los que existian al empezar eran,
ademas, los de un modelo sin entrenar:

```
resultados/full/metrics.json  : split=test num_images=6149 top1=0.01008 top5=0.04456 macro_f1=0.000196
                                backbone=mobilenet_v3_large epoch=1 val_acc=0.00980
resultados/smoke/history.json : train_loss=[4.6627, 4.6250] val_acc=[0.009804, 0.009804]
```

`top1 = 0,98 % = 1/102` y `top5 = 4,90 % = 5/102`: un modelo aleatorio puntua en el azar **y no
por debajo**, lo que de paso valida la aritmetica de las metricas (etiquetas bien alineadas).

**b) Cifras de literatura rotuladas y correctas (CONFIRMADO).** `INFORME.md` §8.2 presenta el
rango "~92 % a ~97 %" como referencia de la literatura y **no** como medicion propia, con una
advertencia explicita. El dato del trabajo original es correcto: la pagina de la publicacion de
Nilsback y Zisserman dice literalmente *"from 55.1% for the best single feature to 72.8% for the
combination of all features"*, que es exactamente lo que cita el informe. La pagina oficial del
dataset confirma tambien las cifras del §5.1 ("Each class consists of between 40 and 258
images", "large scale, pose and light variations", "several very similar categories"). La linea
base aleatoria (0,98 % / 4,9 %) es aritmetica correcta.

**c) Hiperparametros: coinciden literalmente con `src/config.py` (CONFIRMADO).** Contraste campo
a campo de la tabla de `INFORME.md` §6.3 con `Config`: `backbone="efficientnet_b0"`,
`pretrained=True`, `dropout=0.3`, `num_classes=102`, `img_size=224`, `epochs=25`,
`batch_size=32`, `lr=3e-4`, `backbone_lr=3e-5`, `weight_decay=1e-4`, `label_smoothing=0.1`,
`warmup_epochs=2`, `freeze_epochs=3`, `early_stop_patience=6`, `amp=True`, `topk=5`,
`min_conf=0.25`, `seed=42`, `num_workers=2`, `device="auto"`, `debug=False`,
`debug_per_class=4`. **Los 21 campos coinciden sin excepcion.** El optimizador es AdamW
(`src/model.py::build_optimizer`) y el numero de parametros de la cabeza (1280x102+102 = 130 662)
es correcto. "~15-25 min en T4" esta rotulado como estimacion, no como medicion.

**d) Incoherencias encontradas (y su estado).**

| # | Incoherencia | Donde | Severidad | Estado |
|---|---|---|---|---|
| d1 | "despues **se descongelan los ultimos bloques**" frente a un codigo que descongela el backbone **entero** (`set_backbone_trainable(model, not congelado)` pone `requires_grad=True` en todos los parametros). `INFORME.md` §4.2/§6.1 lo describe bien ("se descongela el *backbone*") | `README.md` §5.2 y `PLAN_TRABAJO.md` §4 | media | **parcheado** en `README.md` (seccion 3); `PLAN_TRABAJO.md` queda como documento de plan |
| d2 | Afirmacion de que las particiones oficiales separan las imagenes "por planta fotografiada" | `INFORME.md` §5.3 | media | **corregido por otro subagente** durante la auditoria (seccion 2.3) |
| d3 | Se describe la descarga de `cat_to_name.json` "de la pagina de Oxford": esa URL **devuelve HTTP 404** (verificado en vivo; la pagina oficial solo publica imagenes, segmentaciones, matrices de distancia, `imagelabels.mat`, `setid.mat` y README). El paso de descarga (prioridad 4 de `get_class_names`) es codigo muerto; no afecta al resultado porque la prioridad 1 (`Flowers102.classes`, atributo de clase) y la 3 (`assets/cat_to_name.json`) siempre funcionan | `INFORME.md` §5.5, `contrato interno` §3, `src/data.py:45` | baja | abierto |
| d4 | Ningun artefacto registra si los pesos preentrenados se cargaron realmente (ver e) | `src/train.py`, `src/model.py::save_checkpoint` | media | abierto (parche propuesto) |
| d5 | `tests/test_smoke.py` anunciado en `README.md` §4 y `contrato interno` §2 y no existente | documentacion | media | **resuelto**: el archivo fue creado durante la auditoria y ahora esta y pasa |
| d6 | Documentacion del `Estado del proyecto` desactualizada | `README.md` §9 | baja | **ya estaba actualizada** cuando la revise a fondo |

**e) Verificacion de los pesos preentrenados de cada checkpoint (hallazgo propio).**
Metodo (`.pytmp/sa10_pretrained.py`): comparar la primera convolucion del backbone de cada
checkpoint con los pesos ImageNet de torchvision. Si el modelo se entreno desde cero, la
distancia es del orden de la escala de los pesos; si venia de ImageNet y solo se ajusto unas
epocas, es minuscula.

| Checkpoint | Backbone | `config.pretrained` | `max|Δw|` 1ª conv | Escala ImageNet | Veredicto |
|---|---|---|---|---|---|
| `resultados/smoke/best.pt` | mobilenet_v3_large | `True` | 3.61164 | 3.68407 | **pesos ALEATORIOS** |
| `resultados/smoke2/best.pt` | mobilenet_v3_large | `True` | 3.61164 | 3.68407 | **pesos ALEATORIOS** |
| `resultados/learn/best.pt` | mobilenet_v3_large | `True` | 0.00027 | 3.68407 | preentrenado cargado |
| `resultados/debug/best.pt` | efficientnet_b0 | `True` | 2.53657 | 2.53095 | **pesos ALEATORIOS** |
| `resultados/verif_sa10/best.pt` (mia) | resnet50 | `True` | 0.00033 | 1.97896 | preentrenado cargado |
| `resultados/verif_sa10_nb/best.pt` (mi notebook) | efficientnet_b0 | `True` | 0.00033 | 2.53095 | preentrenado cargado |

El fallback de `pretrained` es real y **silencioso en los artefactos**: los checkpoints con pesos
aleatorios declaran `pretrained=True` y su `history.json` no tiene ninguna clave que lo delate
(`any('pretrain' in k for k in history) == False`). El unico aviso es un `RuntimeWarning` que en
un notebook se pierde entre el ruido de `pip` y `tqdm`. En Colab la descarga de pesos
normalmente funciona y esto no se activa, pero si fallara, la corrida daria ~1 % de accuracy sin
que ningun artefacto lo refleje.

### 2.9 Punto 9 - Demo y CLIs

Comando: `.pytmp/sa10_cli.py` (subprocess real sobre `python -m <modulo>`).

```
### 1. python -m <modulo> --help
  src.train      rc=0 | usage: python -m src.train [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                          (incluye --num-workers)   <-- extension autorizada por el contrato interno §3.b
  src.evaluate   rc=0 | usage: python -m src.evaluate [-h] [--checkpoint CHECKPOINT]
                          (incluye --num-workers)
  src.predict    rc=0 | usage: python -m src.predict [-h] --image IMAGE [--checkpoint CHECKPOINT]
  src.app        rc=0 | usage: python -m src.app [-h] [--checkpoint CHECKPOINT] [--topk TOPK]

### 2. CLI sin checkpoint (ruta inexistente)
  evaluate rc=1
    [error] No existe el checkpoint '...\no_existe_ckpt.pt'. Para generarlo, entrena primero: `python -m src.train`.
    [error] Comando: python -m src.train
  predict rc=1
    [error] No existe el checkpoint '...'. Para generarlo, entrena primero: `python -m src.train` (comando: python -m src.train).
    [error] Entrena primero: python -m src.train

### 3. build_app con checkpoint inexistente (sin lanzar servidor)
  rc=0
tipo: Blocks | tiene launch: True
predecir(None) -> {} | **Sube una imagen primero.** ...
predecir(imagen, ckpt ausente) -> {}
tabla_markdown([]) -> No se pudo obtener ninguna prediccion para esa imagen.
```

En la corrida end-to-end (seccion 2.2): `predecir(ckpt ausente) -> {} | contiene 'Entrena
primero': True`, y en el notebook (celda 20) la demo se levanto de verdad
(`Running on local URL: http://127.0.0.1:7860`) y se cerro en la celda 22.

Conclusion: **CONFIRMADO**. Las 4 CLIs responden a `--help` con `rc=0`; `evaluate` y `predict`
salen con `rc=1` y un mensaje claro que incluye el comando a ejecutar cuando falta el
checkpoint; `build_app` no revienta con checkpoint ausente ni al predecir; `src.app` muestra
ademas el aviso dentro del layout. Observaciones menores: `build_app` devuelve `gr.Blocks` en
lugar de `gr.Interface` (desviacion **documentada** en el docstring del modulo; cumple el
contrato funcional porque tiene `.launch(...)`).

### 2.10 Punto 10 - ZIP de entrega

Se regenero con `empaquetar_colab.py` **despues** de escribir este informe, y se audito el
resultado. Numeros de la version final: **25 archivos, 140 KB, `testzip() == None`** (sin
entradas corruptas). Contenido:

```
25 archivos, 0 entradas corruptas (testzip -> None)
  .gitignore  el contrato interno  INFORME.md  PLAN_TRABAJO.md  README.md  pytest.ini  requirements.txt
  assets/cat_to_name.json      docs/VERIFICACION.md      notebooks/flores_colab.ipynb
  src/__init__.py src/app.py src/config.py src/data.py src/evaluate.py src/model.py
  src/predict.py src/train.py src/utils.py
  tests/conftest.py tests/test_data.py tests/test_eval_predict.py tests/test_model.py
  tests/test_smoke.py tests/test_train.py
```

Comprobaciones:

* Nada peligroso: sin `data/`, sin `resultados/`, sin `__pycache__`, sin `.pyc`, sin `.pt`, sin
  `.venv`, sin `.pytmp`/`.piptmp`, sin `resultados_flores.zip`.
* Sin rutas absolutas de Windows dentro de ningun archivo del ZIP (comprobado leyendo cada
  entrada; tras parchear `contrato interno`, 0 coincidencias).
* `src/config.py` esta en la raiz del ZIP, que es lo que busca el notebook (`raiz_con_config`).
* El notebook del ZIP es **el mismo** que el del disco (mismo SHA-256), y `docs/` y
  `tests/test_smoke.py` estan incluidos.
* `empaquetar_colab.py` **no** viaja dentro del ZIP (no es necesario para ejecutarlo, pero el
  notebook menciona el comando `python empaquetar_colab.py`, que quien solo tenga el ZIP no
  podra ejecutar): hallazgo bajo.

Conclusion: **CONFIRMADO**. La version que yo recibi al empezar la auditoria tenia dos
problemas que ya no existen: (a) la copia del notebook estaba **desactualizada** respecto al
disco (`cfd8237e...` dentro frente a `97ef2b36...`/`4b9e0d8c...` fuera), (b) no incluia `docs/`.
Ambos quedaron resueltos en las regeneraciones posteriores.

---

## 3. Parches aplicados

Todos son **minimos, quirurgicos y de documentacion**; ninguno toca codigo de ejecucion ni los
archivos prohibidos (`notebooks/flores_colab.ipynb`, `src/train.py`, `src/evaluate.py`).

| # | Archivo | Antes | Despues | Motivo |
|---|---|---|---|---|
| 1 | `README.md` §5.2 | "Las primeras épocas entrenan con el backbone congelado; después **se descongelan los últimos bloques** usando *learning rate* diferencial (más bajo para el backbone que para la cabeza)." | "Las primeras épocas entrenan con el backbone congelado; después **se descongela el backbone completo** (el código no selecciona "últimos bloques": `set_backbone_trainable(model, True)` en `src/train.py`) usando *learning rate* diferencial (más bajo para el backbone que para la cabeza)." | Coherencia documento <-> codigo (hallazgo d1) |
| 2 | `contrato interno` §1 | `\| Python local de pruebas \| C:\...\Python312\python.exe (3.12) \|` (ruta del autor, redactada aqui) | `\| Python local de pruebas \| 3.12 (el del sistema; las pruebas del proyecto se ejecutan con el venv `.venv`) \|` | Era la unica ruta absoluta de Windows y viajaba en el ZIP (punto 5) |
| 3 | `contrato interno` §0 regla 5 | La regla prohibia las rutas absolutas **citando el patron literal** (unidad + carpeta de usuarios + puntos suspensivos). | "**Sin rutas absolutas de Windows** en el código (nada de rutas que empiecen por la letra de unidad del disco local del autor)." | La propia norma citaba el patron, de modo que un grep seguia encontrando la cadena en un archivo entregable (falso positivo que se elimina asi) |

**Corregidos por otros subagentes durante la auditoria** (los detecte y verifique, pero no los
parchee yo):

| Archivo | Cambio | Hallazgo |
|---|---|---|
| `requirements.txt` | `gradio>=4.44` -> `gradio>=6.0`; se anadio `nbformat>=5.9` | punto 7 |
| `.gitignore` | se anadio `resultados_flores.zip` | el notebook lo crea en la raiz fuera de Colab |
| `INFORME.md` §5.3 | reescrito para documentar los duplicados exactos entre splits y retirar la afirmacion "por planta" | d2 |
| `tests/test_smoke.py` | creado (93 lineas, end-to-end sintetico) | d5 |
| `notebooks/flores_colab.ipynb` | actualizado a la version final (26 celdas, 13 de codigo) | — |
| `_sa09_build.py` | **eliminado** de la raiz | era un artefacto de desarrollo de 43 KB que, al ejecutarse, sobrescribia el notebook |

**Parches propuestos y NO aplicados** (porque son de archivos en edicion o de otro propietario):

* `src/train.py` (SA-05): registrar el valor real de `pretrained` y mostrarlo.
  ```python
  print(f"  pesos preentrenados cargados: {getattr(model, 'pretrained', None)}")
  ...
  history["pretrained_usado"] = bool(getattr(model, "pretrained", False))
  ```
* `src/data.py:45` (SA-01): la URL `https://www.robots.ox.ac.uk/~vgg/data/flowers/102/cat_to_name.json`
  devuelve 404; corregirla, sustituirla por un origen que exista o documentar que el paso es
  best-effort no funcional.
* `INFORME.md` §5.5 (SA-08): misma matizacion sobre la descarga de `cat_to_name.json`.
* `README.md` §4 / `contrato interno` §2: ahora que `tests/test_smoke.py` existe, estan al dia.

---

## 4. Hallazgos abiertos (severidad y recomendacion)

**ALTA: ninguno.** No he encontrado ningun defecto que rompa el pipeline, corrompa artefactos ni
produzca metricas falsas.

**MEDIA**

1. **Los artefactos no distinguen un modelo preentrenado de uno aleatorio.** Evidencia: tres
   checkpoints del repo (`smoke`, `smoke2`, `debug`) declaran `config.pretrained=True` con
   `max|Δw|` del orden de la inicializacion (3,61 / 3,61 / 2,54 frente a escalas 3,68 / 3,68 /
   2,53), y `history.json` no guarda nada al respecto. En Colab, si la descarga de pesos falla,
   la corrida daria ~1 % de accuracy y **ningun artefacto lo reflejaria**.
   *Recomendacion*: aplicar el parche propuesto (guardar `pretrained_usado` en `history.json` y
   en el checkpoint, imprimirlo en una linea destacada; opcionalmente abortar si
   `pretrained=True` no se pudo cumplir).
2. **`src/app.py` no tiene ningun test** (la demo es el unico entregable sin cobertura
   automatizada; su criterio de aceptacion esta en `contrato interno` §4).
   *Recomendacion*: `tests/test_app.py` con `build_app` (con y sin checkpoint), `predecir(None)`,
   `predecir(imagen, ckpt_ausente)` y `tabla_markdown` (yo lo he verificado a mano y funciona).
3. **`tests/test_data.py::_build_or_skip` convierte cualquier excepcion en `skip`.**
   *Recomendacion*: limitar el `skip` a errores de red, como ya hace
   `tests/test_train.py::_parece_error_de_red`, y dejar que el resto falle.
4. **La aritmetica de `evaluate` no se cruza en los tests.** Nada comprueba que `top1` sea el
   recuento de `correcto=True` del CSV ni que la matriz de confusion cuadre con las predicciones;
   yo lo he verificado en dos corridas y coincide al ultimo digito.
   *Recomendacion*: anadir ese cruce a `test_eval_predict.py` (2 lineas).
5. **`PLAN_TRABAJO.md` §4 sigue diciendo "descongelar los ultimos bloques"** (coherente con el
   plan original, incoherente con la implementacion final).
   *Recomendacion*: anotarlo como decision finalmente no aplicada o ajustarlo.

**BAJA**

6. La descarga de `cat_to_name.json` desde Oxford da **404** (codigo muerto; sin impacto porque
   `Flowers102.classes` y `assets/cat_to_name.json` bastan y coinciden al 100 %).
7. `empaquetar_colab.py` no viaja en el ZIP aunque el notebook lo mencione.
8. Los tests no ejecutan las CLIs como modulo (`python -m ...`), solo `main([...])`.
9. `tests/test_data.py::test_build_transforms_train_is_random` es un assert probabilistico.
10. Hay 4 directorios vacios `pytest-cache-files-*` en la raiz (restos de ejecuciones previas;
    no entran en el ZIP porque `empaquetar_colab.py` usa lista blanca).

---

## 5. Resumen de debilidades de los tests

Los 78 tests comprueban cosas reales con valores esperados concretos; no he encontrado ningun
test "de adorno" (sin asserts o con asserts triviales del tipo `assert x is not None`). Las
debilidades son de **cobertura** y de **robustez ante fallos**: 0 tests para `src/app.py`; 3
puntos donde un fallo real terminaria en `skip`; la aritmetica de metricas solo se comprueba de
forma estructural; no se verifica que los pesos preentrenados se carguen; no se ejercitan los
CLIs como modulo; 1 assert probabilistico.

---

## 6. Lo que NO se ha podido verificar (honestidad de esta auditoria)

1. **La corrida real en Colab con GPU T4.** No hay acceso a Colab y la GPU local (RTX 3060 Ti)
   no es utilizable bajo el sandbox. Todo lo medido es CPU.
2. **La rama CUDA/AMP** (`autocast` + `GradScaler` en FP16): solo revisada por lectura y con los
   espias de los tests en CPU. No se comprobo convergencia, rendimiento ni ausencia de
   `inf/nan`.
3. **La corrida completa de 25 epocas.** El "~15-25 min en T4" y las metricas finales de la
   seccion 8 del informe siguen sin existir. Lo mas largo que he ejecutado han sido 2 epocas del
   dataset completo (5,8 min en CPU) y una evaluacion de 6149 imagenes (6,3 min en CPU) con un
   checkpoint de 4 epocas.
4. **`share=True` (enlace publico de Gradio), `files.upload`, `files.download` y el montaje de
   Google Drive.** Solo verificado que la app se construye y levanta en local
   (`http://127.0.0.1:7860`) y que esas ramas estan envueltas en `try/except` con mensajes
   claros.
5. **La instalacion real por `pip` de la celda 2** (aqui todas las dependencias ya estaban, asi
   que la rama `pip install -q` no se ejecuto) y la compatibilidad con Gradio 4.44/5.x.
6. **La ejecucion del notebook como notebook de Jupyter/Colab**: `nbclient` no puede arrancar
   kernel en este sandbox (`SetFileSecurity: Acceso denegado`). Se ejecuto el codigo celda a
   celda en un proceso Python, lo que no cubre magics, reinicios de kernel ni el estado del
   kernel tras un fallo.
7. **La descarga del dataset desde cero en Colab.** Aqui `data/flowers-102` ya existia; si se
   verifico que torchvision acepta el dataset existente, que los splits son los oficiales y que
   `build_dataloaders` funciona con y sin debug.
8. **La rama `MODO_CODIGO="zip"` del notebook** (exige `files.upload`): se audito el contenido
   del ZIP y se ejecuto la rama `auto`; subir el ZIP a Colab no se pudo probar.
9. **Instalar `requirements.txt` en un entorno limpio**: no se hizo (habria degradado el venv
   compartido); la revision es de contenido y coherencia, no de resolucion real de dependencias.
10. **La ejecucion del notebook con la GPU y con `MODO_DEBUG=False`** (la corrida real de 25
    epocas), que es el unico paso que el proyecto declara pendiente y que corresponde al usuario.

---

## 7. Registro de cambios ocurridos durante la auditoria

Para que nadie compare cifras de revisiones distintas:

| Momento | Que cambio | Impacto en la auditoria |
|---|---|---|
| Durante la 1ª ejecucion de pytest | SA-05/SA-06 anadieron 4 tests (`num_workers` en `train.py`/CLIs y sus casos) | El recuento paso de 73 a 77 casos |
| Al empezar mi corrida e2e | `src/train.py` (22151 -> 26031 B) y `src/evaluate.py` ganaron la degradacion automatica de `num_workers` y el flag `--num-workers` | Tuve que re-verificar que el `README.md` §3 (uso local en Windows) ya funciona; corrida `sa10_nw2` para comprobarlo |
| A mitad de la auditoria | `INFORME.md` §5.3 reescrito con el hallazgo de duplicados MD5 | d2 cerrado |
| A mitad de la auditoria | `requirements.txt`: `gradio>=6.0` y `nbformat>=5.9` | Punto 7 cerrado |
| A mitad de la auditoria | `.gitignore` += `resultados_flores.zip`; `_sa09_build.py` eliminado | Hallazgos 9 y 10 evitados |
| A mitad de la auditoria | `tests/test_smoke.py` creado; notebook actualizado a `4B9E0D8C75EA` | P1 gana un test solido; volvi a ejecutar el notebook con la version nueva |
| Al ejecutar el notebook (1er intento) | Se bloqueo en `plt.show()` (mi emulador, no el proyecto) | 2ª ejecucion con `MPLBACKEND=Agg`, completada en 102,1 s |
| Al cerrar la auditoria | Yo parchee `README.md` §5.2 y `contrato interno` §0/§1 (seccion 3) | Rutas absolutas: 0 coincidencias; fine-tuning descrito como es |
