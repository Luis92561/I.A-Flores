# El cuaderno de Colab, las pruebas automáticas y el empaquetado

Este capítulo explica las tres piezas del proyecto que no son el programa en si, pero que hacen
que el programa se pueda ejecutar y se pueda confiar en el: el **cuaderno de Colab** (el archivo
que pulsas para que todo ocurra), las **pruebas automáticas** (el mecanismo que detecta que algo
se ha roto) y el **empaquetado** (el archivo comprimido que se sube a la nube). También se explica
el archivo `requirements.txt`, que es la lista de programas auxiliares que necesita el proyecto.
No hace falta saber programar para leerlo: cada término técnico se define la primera vez que
aparece.

## Qué es un cuaderno (notebook) y qué es una celda

Un **notebook** (cuaderno) es un documento que mezcla texto explicativo y código ejecutable en la
misma página. El formato se llama **Jupyter**, y **Google Colab** es un servicio gratuito de
Google que abre estos cuadernos en el navegador y ejecuta el código en un ordenador suyo, no en el
tuyo. Eso es lo importante para este proyecto: no hay que instalar nada en tu ordenador, y el
cuaderno puede usar una **GPU**.

Una **GPU** (unidad de procesamiento gráfico) es un componente pensado para dibujar imágenes que,
resulta, también sirve para hacer los cálculos matemáticos de la inteligencia artificial mucho más
deprisa que el procesador normal. **Colab** regala horas de una GPU concreta, la **T4**, y por eso
el proyecto entero está pensado para ejecutarse ahí.

El cuaderno se divide en **celdas** (casillas). Hay dos tipos:

- **Celda de texto.** Contiene explicaciones escritas, títulos y tablas. No se ejecuta: solo se
  lee.
- **Celda de código.** Contiene instrucciones para el ordenador. Se ejecuta al pulsar el botón de
  reproducir que aparece a su izquierda, y la respuesta del programa (los "mensajes que imprime")
  aparece justo debajo de la celda.

La diferencia práctica es está: el texto te cuenta **qué** se va a hacer y **por qué**; la celda
de código lo **hace**. El cuaderno de este proyecto es `notebooks/flores_colab.ipynb`. La
extensión `.ipynb` significa "cuaderno de IPython" y es el formato estándar de Jupyter y de Colab.
El cuaderno tiene 26 celdas, de las cuales 13 son de código y 13 son de texto. Se ejecutan **en
orden**, de arriba abajo, porque cada celda aprovecha lo que hizo la anterior: si te saltas una,
la siguiente puede no encontrar lo que necesita.

Una aclaración útil: **no hay nada que memorizar de las celdas de código**. Están escritas y
probadas; tu trabajo es pulsar "ejecutar" y, si algo se queja, leer el mensaje. Todo el código
lleva comentarios en español explicando cada bloque.

## Recorrido del cuaderno, paso a paso

Las celdas de código están numeradas como pasos del 1 al 10. Esto es lo que hace cada uno, en
lenguaje llano.

### Pasó 1 - Comprobar la GPU e instalar solo lo que falte

Antes de gastar tiempo, el cuaderno mira que el entorno este en condiciones. Lo primero es
comprobar que existe una GPU disponible, con una orden que se llama `torch.cuda.is_available()`.
Si la respuesta es "no", el cuaderno **no se rompe**: avisa y sigue, pero entrenar sin GPU tarda
muchisimo más, así que lo razonable es activarla en el menu de Colab antes de continuar.

Después viene una decisión importante que se explica sola en una tabla del propio cuaderno: los
**programas auxiliares**. Un programa de inteligencia artificial no se escribe desde cero: se
apoya en bibliotecas, que son colecciones de código ya hecho. La celda hace tres cosas:

1. **Comprueba** si ya están las bibliotecas pesadas (`torch` y `torchvision`, que son el motor de
   cálculo) y **no las instala nunca**. Esto es deliberado y se explica en el apartado del
   `requirements.txt`.
2. **Instala solo lo que falta** de cuatro bibliotecas más ligeras: `gradio` (la pantalla
   interactiva), `scikit-learn` (las métricas), `matplotlib` (los gráficos) y `scipy` (para leer
   los ficheros del catálogo de flores). Si ya estaban, no se llama al instalador.
3. Imprime un **resumen de versiones**: que motor tiene, si hay GPU, cuánta memoria tiene.

```text
torch y torchvision ya estan disponibles: no se tocan (regla de Colab).
Todo lo necesario ya estaba instalado: no se llama a pip.
=== Resumen del entorno ===
torch          : 2.14.1+cpu
CUDA disponible: False
```

### Pasó 2 - Montar Google Drive (opcional) y guardar el dataset

**Google Drive** es el disco duro en la nube de Google; **montar** Drive significa "conectar" esa
carpeta para que el programa pueda leer y escribir en ella como si fuera una carpeta normal. Este
pasó es **opcional** y sirve para dos cosas:

- **Que el trabajo no se pierda.** Colab borra su disco cuando te desconectas. Si el resultado del
  entrenamiento (el archivo `best.pt`, que es el **checkpoint**, o sea la "foto" del modelo ya
  entrenado) está copiado en Drive, sobrevive a la desconexion.
- **No volver a descargar los datos.** El catálogo de fotos ocupa unos 330 MB y, sin esta caché,
  Colab lo vuelve a descargar en cada sesión. Aquí se guarda en Drive una vez y se restaura al
  empezar.

Todo el pasó está envuelto en un mecanismo de protección (`try`/`except`, que significa "intenta
esto y, si falla, no te caigas"): si Drive no está disponible, el cuaderno avisa y continua. Hay
un interruptor al principio de la celda para decidir si quieres usar Drive o no.

### Pasó 3 - Obtener el código del proyecto

Aquí el cuaderno se hace con el programa. Hay cuatro maneras y se eligen con una variable llamada
`MODO_CODIGO`:

| `MODO_CODIGO` | Cuándo usarlo | Qué hace |
|---|---|---|
| `"auto"` (el valor por defecto) | Siempre | Si el proyecto ya está en la máquina, lo usa; si no, pide subir el archivo comprimido |
| `"zip"` | Estas en Colab y no tienes el proyecto allí | Pide subir el ZIP, lo descomprime y busca dentro la carpeta del proyecto |
| `"clone"` | Tienes el proyecto publicado en GitHub | Se lo descarga desde internet |
| `"drive"` | Ya tienes la carpeta del proyecto en Drive | La copia a la zona de trabajo |

En la práctica se deja `MODO_CODIGO = "auto"`, que es inteligente: busca una señal concreta (un
archivo llamado `src/config.py`) y, si la encuentra, no te pregunta nada. Solo si no la encuentra
te pide el ZIP. Al final la celda comprueba que el archivo clave existe y, si no, se detiene con un
mensaje claro en lugar de fallar tres pasos más adelante de forma confusa.

```text
MODO_CODIGO='auto': proyecto detectado en /content/IA_flores
Raiz del proyecto: /content
src/config.py    : OK
```

### Pasó 4 - Imports del proyecto

"Importar" significa traer al cuaderno las piezas escritas en otros archivos, para poder usarlas.
Esta celda carga la configuración del proyecto, la lista de las 102 especies, la función que fija
el azar para que los resultados sean reproducibles y la función que decide si se calcula en GPU o
en el procesador. No hay nada que configurar: se ejecuta y dice "Imports del proyecto OK".

### Pasó 5 - Configuración: modo debug y modo completo

Aquí se decide si la corrida es un **ensayo rápido** o la **corrida de verdad**. El cuaderno
arranca siempre en **modo debug** (ensayo), y esa palabra merece una explicación: no significa
"con errores", sino "en pequeño, para comprobar que todo funciona". La tabla del cuaderno lo
resume así:

| | `MODO_DEBUG = True` (por defecto) | `MODO_DEBUG = False` |
|---|---|---|
| Imágenes | unas pocas por especie (102 x 4 x 3) | 1020 de entrenamiento, 1020 de validación y 6149 de prueba |
| Épocas | 2 | 25 |
| Precisión mixta | desactivada | activada si hay GPU |
| Salida | carpeta `resultados/debug` | carpeta `resultados` |
| Tiempo en GPU T4 | 1 o 2 minutos | 15 a 25 minutos |

Una **época** es una pasada completa por todas las fotos de entrenamiento; se dan varias pasadas
porque el modelo mejora poco a poco. La **precisión mixta** es un truco para hacer los cálculos con
menos decimales y así ir más rápido sin perder calidad apreciable.

El detalle importante es que el modo debug escribe en una carpeta distinta, así que **no pisa** el
resultado bueno. La rutina recomendada es: primero debug para verificar que todo funciona, luego
cambiar la variable y repetir para la corrida definitiva.

### Pasó 6 - Entrenamiento

**Entrenar** es el proceso de aprendizaje: se le muestran al modelo las fotos con su nombre
correcto, se mide cuánto se equivoca y se corrige. Esta celda lanza ese proceso y avisa de todo lo
que va generando:

- `best.pt`: el checkpoint, o sea la copia del modelo en su mejor momento.
- `history.json`: el diario de la corrida, con la pérdida (cuánto se equivoca), el acierto y la
  velocidad de aprendizaje, época por época.
- `config_usada.json`: los ajustes exactos que se usaron, por si hay que repetir la corrida.
- `figuras/curvas.png`: un gráfico con la evolución de todo lo anterior.

La primera vez se descarga el catálogo de flores (330 MB) dentro de la carpeta `data/`; después se
reutiliza. Además, si Drive está montado, la celda copia los resultados a Drive **justo después de
entrenar**, de modo que si Colab se cae un minuto más tarde el trabajo está a salvo.

El cuaderno también traduce a lenguaje claro un aviso técnico que puede aparecer: si no consigue
bajar los pesos preentrenados desde internet, lo dice y explica que el entrenamiento sigue siendo
válido, solo que parte de cero. Este aviso tiene su propia historia, contada en el capítulo de
decisiones y errores.

### Pasó 7 - Evaluación sobre el conjunto de prueba oficial

**Evaluar** es poner a prueba al modelo con fotos que **nunca vio** mientras aprendía. Si se
evaluara con las mismas fotos del entrenamiento, el resultado sería engañoso: sería como darle a
alguien el examen con las respuestas ya vistas. El catálogo de flores trae sus propias particiones
oficiales, y aquí se usa la de prueba (6149 fotos).

Esta celda escribe:

- `metrics.json`: el resumen numérico. Los dos datos principales son el **top-1** (porcentaje de
  veces que la especie más probable era la correcta) y el **top-5** (porcentaje de veces que la
  respuesta correcta estaba entre las cinco primeras propuestas). También guarda el **macro-F1**,
  una nota que reparte el merito entre las 102 especies por igual, para que una especie con muchas
  fotos no tape a otra con pocas.
- `figuras/confusion_matrix.png`: la "matriz de confusión", una tabla de 102 x 102 que muestra que
  especies se confunden entre si, y `figuras/top_confusions.png`, un resumen de las confusiones más
  frecuentes.
- `predicciones/test_predicciones.csv`: la lista completa de fotos con lo que se predijo para cada
  una.

La segunda celda de este pasó imprime el resumen y el ranking de las especies peor clasificadas, y
enseña la imagen de la matriz de confusión.

### Pasó 8 - Resumen listo para pegar en el informe

Esta celda no calcula nada nuevo: **recoge** las cifras de los dos archivos anteriores y las
presenta como un bloque de texto ya formateado, entre dos marcas visuales. Ese bloque se copia y se
pega directamente en la sección de resultados del informe. Lo importante es que **no hay ningún
número escrito a mano**: la tabla se rellena sola con lo que mida tu corrida, y si la corrida fue
en modo debug, el propio bloque lleva un aviso recordando que esas cifras son de un ensayo. El
bloque también se guarda en un archivo para no perderlo.

### Pasó 9 - Predicción de una foto suelta y demostración interactiva

Una **predicción** es pedirle al modelo que diga que flor es. Hay dos formas aquí:

- **Foto suelta.** Se elige una foto del catálogo (o de la carpeta de recursos del proyecto) y el
  programa imprime las cinco especies más probables con su porcentaje.
- **Demostración interactiva.** Se construye una pantalla web sencilla (hecha con la biblioteca
  `gradio`) donde puedes arrastrar una foto tuya, o usar la cámara, y ver el resultado. Si todavía
  no existe el modelo entrenado, la pantalla no se cae: muestra el aviso de que hay que entrenar
  primero.

Sobre está demo el cuaderno avisa de dos cosas: que se puede generar un enlace público temporal
(que conviene no dejar abierto con fotos privadas) y que se lanza en segundo plano para no dejar la
celda ocupada. Hay incluso una celda auxiliar para apagarla si quieres liberar el recurso.

### Pasó 10 - Empaquetar los resultados, descargarlos y cachear los datos

El último pasó comprime toda la carpeta `resultados/` en un archivo ZIP llamado
`resultados_flores.zip`, lo descarga a tu ordenador y deja una copia en Drive. Además copia el
catálogo de fotos a Drive para que la próxima sesión no vuelva a descargar los 330 MB. Si no hay
Drive, todo esto se omite sin errores.

Al final imprime un inventario: una lista con lo que existe y lo que falta de todo lo que debía
generarse. Es la comprobación final de que la corrida produjo lo que tenía que producir.

### Checklist final

La última celda del cuaderno es una **lista de comprobación** escrita, con casillas. Reúne en un
solo sitio lo que hay que verificar antes de lanzar la corrida definitiva, donde queda cada
archivo, como reanudar el trabajo si Colab se desconecta, como descargar los resultados y cuánto
tarda cada escenario. Es la página que conviene leer antes de ponerse a ejecutar.

## Las pruebas automáticas

Una **prueba automática** (o test) es un pequeño programa cuyo único trabajo es comprobar que otra
parte del programa hace lo que promete. Se ejecuta solo, sin que nadie mire, y responde "correcto"
o "falló". La idea es sencilla: si cambias algo y rompes otra cosa sin darte cuenta, la prueba lo
detecta en segundos, en lugar de que el error aparezca tres semanas después delante de un tribunal.

Las pruebas de este proyecto viven en la carpeta `tests/`. Son **78 pruebas** repartidas en seis
archivos:

| Archivo | Que comprueba, en una frase |
|---|---|
| `tests/test_data.py` | Que las fotos se cargan y se preparan bien: que el tamaño de los lotes es el correcto, que las etiquetas van del 0 al 101, que el entrenamiento recibe fotos variadas y la validación siempre las mismas, y que la lista de las 102 especies se obtiene incluso sin internet. |
| `tests/test_model.py` | Que el modelo se construye bien, que recibe una imagen y devuelve 102 respuestas, que se puede guardar en un archivo y volver a cargarlo dando exactamente el mismo resultado, y que se puede congelar y descongelar su parte preentrenada. |
| `tests/test_train.py` | Que el entrenamiento termina, guarda sus archivos y sigue las reglas pactadas: la forma de la velocidad de aprendizaje, que se congela y se descongela la parte preentrenada en el momento previsto, que la parada temprana corta a tiempo y que dos corridas con la misma semilla dan el mismo resultado. |
| `tests/test_eval_predict.py` | Que la evaluación escribe las métricas, las figuras y el listado de predicciones, que falla con un mensaje claro si falta el modelo entrenado, y que la predicción devuelve resultados ordenados, recorta bien el número de propuestas y respeta el umbral de confianza. |
| `tests/test_smoke.py` | Que el recorrido completo funciona encadenado (cargar datos, modelo, entrenar, evaluar, predecir) sobre un conjunto de imágenes inventadas, sin internet y en unos segundos. |
| `tests/conftest.py` | No es una prueba, es el preparador común: crea las carpetas temporales que usan las demás pruebas, con un ajuste especial para está máquina concreta. |

Para que sirven, dicho en una frase: **son el detector de humo del proyecto**. Si alguien toca el
código y rompe la carga de datos, o el guardado del modelo, o el cálculo del umbral, lo sabrá en un
minuto sin tener que revisar nada a mano. Y si una prueba se queda en verde después de un cambio,
eso es la evidencia de que el cambió no rompió lo que ya funcionaba.

Se ejecutan con el comando siguiente, desde la carpeta del proyecto:

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

La respuesta termina con una línea parecida a `78 passed`, que significa "78 pruebas superadas".
Las pruebas que necesitan el catálogo real de 330 MB (las más lentas) solo se ejecutan si se avisa
antes con una variable de entorno; si no, se saltan para no hacer esperar. Su propio texto lo dice:
si falta un programa auxiliar, algunas pruebas se saltan en lugar de fallar, de modo que un
`78 passed` sin haber preparado el entorno no es exactamente lo mismo que un `78 passed` completo.
Por eso el registró de verificación detalla con que ajustes se obtuvo cada recuento.

## El empaquetado: empaquetar_colab.py

Un **ZIP** es un archivo comprimido: un solo archivo que contiene muchos otros, ocupando menos
espacio, como una caja donde metes el proyecto entero para transportarlo. Es la forma práctica de
llevar el proyecto a Colab, porque subir carpetas sueltas por la pantalla del navegador es lento y
propenso a errores.

El programa `empaquetar_colab.py` construye ese archivo. Se ejecuta con:

```bash
.venv\Scripts\python.exe empaquetar_colab.py
```

Produce `IA_flores_colab.zip` en la carpeta del proyecto. Lo más interesante no es lo que incluye,
sino **lo que deja fuera**:

- **La carpeta `data/` con las 330 MB de fotos.** No viaja en el ZIP: es enorme, es descargable y
  además no es nuestra (las fotos pertenecen a sus autores y se distribuyen para uso académico).
- **La carpeta `resultados/`.** Son resultados generados por una corrida concreta; el ZIP debe
  contener el programa, no sus resultados.
- **Los restos de ejecuciones anteriores**: las carpetas de caché del propio Python y los archivos
  que se regeneran solos.

Lo que si incluye es el código, las pruebas, el cuaderno, la documentación, los recursos del
proyecto y los archivos de configuración.

Para decidir esto se usa una **lista blanca**: en lugar de recorrer todo el proyecto y quitar lo
que no quieres, se escribe **explícitamente** la lista de lo que si quieres. Es una decisión de
seguridad y de robustez, y tiene dos motivos que el propio programa documenta:

1. Con una lista blanca es imposible que se cuelen por descuido los 330 MB de fotos o los
   resultados de una corrida.
2. En el ordenador de desarrolló, algunas carpetas temporales no se pueden ni leer por una
   restricción del sistema; recorrerlo todo fallaría. La lista blanca ni las toca.

El programa no se limita a crear el ZIP: después **lo vuelve a abrir para comprobarlo**. Verifica
que las piezas clave están dentro (el archivo de configuración, los módulos, el cuaderno, la lista
de dependencias) y que ninguna entrada está corrupta. Si falta algo, lo dice y termina con un
aviso de error en lugar de dejarte un paquete incompleto.

```text
ZIP creado: IA_flores_colab.zip  (160 KB)
Archivos incluidos: 28

Comprobacion de la estructura que espera el notebook:
  OK   src/config.py
  OK   src/utils.py
  OK   src/data.py
  OK   src/model.py
  OK   notebooks/flores_colab.ipynb
  OK   requirements.txt

RESULTADO: estructura correcta
```

Un dato curioso que merece mención: el proyecto entero, sin las fotos ni los resultados, cabe en
unos 160 KB, es decir, menos de una décima parte de una sola foto sin comprimir de muchos móviles.
Un informe de verificación anterior registró 25 archivos y 142 KB; la cifra creció después, porque
el propio documento que estas leyendo viaja dentro del paquete. Los archivos de datos son los que
pesan, no el programa.

## El archivo requirements.txt

Un archivo `requirements.txt` es una **lista de la compra de programas auxiliares**. Cada línea
indica un nombre y la versión mínima que hace falta, por ejemplo la primera línea dice que se
necesita una versión de `torch` igual o posterior a la 2.4. Sirve para que cualquiera pueda
preparar un ordenador nuevo con un solo comando y obtener las mismas piezas:

```bash
pip install -r requirements.txt
```

Ese comando se llama `pip` y es el instalador de programas auxiliares de Python. Aquí aparece la
**trampa más importante de todo el proyecto**, y está escrita con mayúsculas en el propio archivo y
repetida en el cuaderno:

> En Colab **no** se debe ejecutar ese comando. Colab ya trae `torch` y `torchvision` preparados
> para funcionar con la GPU. Si se reinstalan desde `pip`, es muy probable que se sustituyan por
> versiones para procesador normal y la GPU deje de servir. La solución es instalar únicamente lo
> que falte, que es exactamente lo que hace el pasó 1 del cuaderno. Si aún así se rompe, hay que
> reiniciar el entorno de Colab.

El archivo cumple entonces dos papeles distintos según donde estés: en un ordenador propio es la
lista de la compra completa; en Colab es una **referencia** de lo que el proyecto usa, pero no se
ejecuta tal cual. Las anotaciones que acompañan a algunas líneas explican para que sirve cada pieza
y por que se exige una versión mínima concreta, por ejemplo que la biblioteca de la pantalla
interactiva necesita una versión moderna porque en las antiguas faltaba una pieza que el proyecto
usa.

## Resumen del capítulo

- El **cuaderno** `notebooks/flores_colab.ipynb` es el guion: 26 celdas, 13 de ellas de código,
  que recorren el proyecto en 10 pasos, empiezan en modo debug para verificar rápido y terminan
  descargando los resultados.
- Las **78 pruebas automáticas** de `tests/` vigilan que nada se rompa en silencio, sin necesidad de
  revisar el programa a mano.
- El **empaquetado** `empaquetar_colab.py` construye un ZIP con lista blanca que deja fuera las
  330 MB de fotos y los resultados, y que se autocomprueba después de crearse.
- El **`requirements.txt`** es la lista de dependencias, con el aviso crítico de no reinstalar
  `torch` dentro de Colab.
