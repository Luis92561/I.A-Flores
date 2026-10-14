# Por qué se hizo así: las decisiones y los errores del camino

Este capítulo no explica **que** hace el proyecto, sino **por que** se decidió cada cosa y **que
salió mal** por el camino. Es el capítulo más útil de todo el documento, por una razón sencilla:
los aciertos se cuentan solos, pero los errores solo enseñan si alguien los cuenta. Al final hay
también una lista honesta de lo que **no** se ha podido verificar.

Cada decisión se presenta con el problema que había encima de la mesa, la alternativa que se
descartó y el motivo.

## 1. Por qué se usaron las flores de Oxford y no fotos propias

**El problema.** Para que el programa aprenda a distinguir especies hace falta un montón de fotos
ya etiquetadas con su nombre correcto. Una persona sola puede hacer, con mucha paciencia, cien o
doscientas fotos de su jardín. Un programa de este tipo necesita miles.

**La alternativa descartada.** Hacer las fotos uno mismo. Se descartó por tres motivos que se
refuerzan entre si:

- **Cantidad.** Con doscientas fotos de veinte especies el programa no aprende: memoriza. Aprender
  a distinguir 102 especies requiere miles de ejemplos.
- **Calidad de las etiquetas.** Cada foto habría que etiquetarla a mano, y equivocarse en la
  etiqueta de una sola foto engaña al programa. Existen flores muy parecidas entre si, y una
  persona sin formación botánica dudaría.
- **Comparabilidad.** Aunque se hicieran las fotos, no habría con que comparar el resultado: nadie
  más tendría ese conjunto de fotos, así que la cifra de acierto no diría nada a nadie.

**La decisión.** Usar **Oxford Flowers-102**, un catálogo público creado por investigadores de la
Universidad de Oxford, con 8189 fotografías de 102 especies de flores comunes en el Reino Unido.
Tiene tres virtudes que resuelven los tres problemas: es grande, viene ya etiquetado y es un
catálogo conocido y usado en publicaciones científicas, así que los resultados se pueden comparar
con los de otros.

**El precio que se paga, y se dice en voz alta.** El catálogo son fotos de catálogo, en su mayoría
hechas en el Reino Unido, con fondos e iluminaciones parecidos. Fotos propias, con fondo cargado,
de noche o de una variedad distinta, pueden salir mucho peor. Que el programa acierte mucho con
este catálogo **no garantiza** que acierte con tus fotos. Validarlo con fotos propias quedó como
trabajo futuro, no se hizo.

## 2. Por qué PyTorch y no TensorFlow

**El problema.** Existen varias "cajas de herramientas" para construir inteligencia artificial.
Las dos grandes son **PyTorch** y **TensorFlow**. Una caja de herramientas, en este contexto, es
un conjunto de piezas ya escritas que te ahorra programar desde cero las cuentas matemáticas.
Había que elegir una.

**La decisión.** Se eligió **PyTorch**. Los motivos, en orden de importancia práctica:

- **Tiene ya preparado el catálogo de flores y muchos más.** La biblioteca hermana de PyTorch,
  TorchVision, incluye una pieza que descarga Flowers-102 y reparte las fotos en sus particiones
  oficiales con una sola orden. Con la otra herramienta ese trabajo habría habido que hacerlo a
  mano.
- **Tiene ya preparado el modelo preentrenado** que se quería usar, con sus pesos ya entrenados
  descargables, y también modelos alternativos para comparar sin cambiar apenas código.
- **La pantalla interactiva de demostración (Gradio) se lleva muy bien con PyTorch**: se puede
  enseñar directamente un modelo en una página web sin escribir una aplicación entera.
- **Está pensado para leerlo.** Está escrito de forma más parecida al lenguaje normal, lo que
  importa mucho en un proyecto cuyo código tiene que entender otra persona, o el propio autor
  meses después.

**Lo que no se hizo.** No se comparó experimentalmente con TensorFlow. La elección se tomó por
razones de utilidades disponibles y de facilidad de lectura, no por una medición de velocidad. Es
una elección razonada, no un resultado experimental, y por eso se presenta como tal.

## 3. Por qué se usó una red ya entrenada y no una red construida desde cero

**El problema.** El conjunto de entrenamiento tiene 1020 fotos para 102 especies: **diez fotos por
especie**. Una red de este tipo tiene que aprender a reconocer pétalos, texturas y formas, y diez
ejemplos no alcanzan para aprenderlo todo desde el principio.

**La alternativa descartada.** Construir la red desde cero, con pesos iniciales al azar, y
enseñarle todo con esas diez fotos por especie. Se descartó porque la experiencia dice que, con tan
pocos datos, la red memoriza las diez fotos concretas en lugar de aprender el concepto de la
especie. Aprende a reconocer "está foto", no "está flor". Además necesitaría muchisimo más tiempo
de cálculo.

**La decisión.** Usar **aprendizaje por transferencia**, que se explica con una comparación: en
lugar de enseñar a alguien a mirar flores desde cero, se parte de una persona que ya sabe mirar
imágenes en general, porque ha visto millones de ellas, y solo se le explica la parte de flores. En
términos técnicos, se toma una red ya entrenada para clasificar un millón de imágenes de mil
categorías distintas (un catálogo llamado **ImageNet**) y se aprovecha su partes iniciales, que
detectan bordes, manchas y texturas: eso es útil para cualquier imagen, no solo para las del
catálogo original.

La red elegida se llama **EfficientNet-B0**, una red relativamente pequeña y muy eficiente, adecuada
para la GPU gratuita de Colab y para entrenar en minutos. Sobre ella se pone una **cabeza nueva**:
una última capa, propia de este proyecto, que aprende a repartir lo que la red ve entre las 102
especies. La cabeza se inicializa al azar y es lo que realmente se aprende en este proyecto.

**Traducción honesta.** El proyecto no enseña a una red a ver flores desde cero: aprovecha una que
ya sabe ver imágenes y le enseña la última parte del trabajo. Eso es una técnica estándar, no un
atajo vergonzoso, y se documenta como tal.

## 4. Por qué congelar la parte preentrenada 3 épocas y luego descongelarla con una velocidad menor

**El problema.** Si desde el primer momento se permite ajustar **toda** la red, incluidas las
partes preentrenadas, los primeros cálculos son muy malos (la cabeza nueva empieza al azar y da
respuestas absurdas). Esos errores, aplicados sobre las partes que ya eran buenas, las estropean.
Es como pedirle a un fotógrafo experto que se fije en las correcciones de alguien que acaba de
empezar: acabaría dudando de lo que hacía bien.

**La decisión, en dos etapas.** Se llama **congelar** a impedir temporalmente que una parte de la
red se modifique, y **descongelar** a volver a permitirlo.

1. **Las 3 primeras épocas se congelan** las partes preentrenadas: se usan tal cual, solo como
   "vista", y únicamente se entrena la cabeza nueva. Así la cabeza aprende a ordenar lo que la red
   ya sabe ver, sin estropear nada.
2. **A partir de la cuarta época se descongela todo** (la red completa, no solo sus últimos
   bloques) pero con una **velocidad de aprendizaje menor**. Esa velocidad (el "learning rate" o
   tasa de aprendizaje) es el tamaño del pasó con que la red corrige sus errores. La cabeza lleva
   un pasó de 0,0003 y las partes preentrenadas un pasó diez veces más pequeño, 0,00003: se afinan
   con cuidado, sin deshacer lo aprendido.

El cuaderno y el informe dejan escrito este número de épocas (3) y estas dos velocidades, para que
la corrida se pueda repetir. También hay una prueba automática que comprueba que la secuencia real
es exactamente "congelado, congelado, congelado, libre, libre, libre": es uno de los pactos que
vigilan las pruebas, porque es fácil equivocarse al programar esto.

**Un matiz que se corrigió.** En una versión inicial de la documentación se decía que al
descongelar se liberaban "los últimos bloques". El código hace otra cosa: libera **toda** la parte
preentrenada. Se detectó al revisar, y el texto se corrigió para que dijera la verdad. Es un
ejemplo pequeño pero real de por que conviene leer el código antes de describirlo.

## 5. Por qué se aumentaron los datos y por qué se suavizaron las etiquetas

Aquí hay dos técnicas distintas que atacan el mismo problema: **con diez fotos por especie, la red
tiende a memorizar**. Y dos medidas para evitarlo.

**Aumentar los datos.** Aumentar los datos (aumentó de datos, o "data augmentation") significa
fabricar variantes de cada foto de entrenamiento: recortes, giros, cambios de color y de brillo. La
especie no cambia, pero la red ve muchas más versiones de la misma flor. Sirve para dos cosas: hay
más material con el que practicar y, sobre todo, la red se ve obligada a no depender de detalles
accidentales (el fondo, el encuadre). Una flor vista de otra manera sigue siendo la misma flor, y
eso es precisamente lo que se quiere que aprenda.

El detalle no es casual: estas variantes **solo se aplican al entrenamiento**. Para validar y para
medir el resultado final, las fotos se preparan siempre igual, sin azar. Si se aplicaran variantes
al evaluar, el resultado cambiaría de una vez a otra y no se podría comparar nada.

**Suavizar las etiquetas.** Suavizar las etiquetas ("label smoothing") consiste en no exigirle a la
red que este **completamente segura**. Normalmente, cuando se le muestra una foto se le dice "la
respuesta correcta es la especie 47 y todo lo demás está mal". Con el suavizado se le dice algo más
realista: "es casi seguro la 47, pero deja un margen pequeñísimo a las demás". Eso tiene dos
efectos buenos:

- La red se vuelve menos fanática y no se sobreajusta tanto, porque no puede alcanzar la perfección
  absoluta y sigue corrigiendo.
- Sus probabilidades son **más honestas**. Esto es decisivo para el umbral de confianza que se
  explica más abajo: si la red dijera "99 por ciento" incluso cuando duda, el umbral no serviría de
  nada.

**El resultado de estas dos medidas**, junto con la parada temprana (dejar de entrenar cuando el
resultado de validación deja de mejorar durante 6 épocas seguidas, para no dar vueltas de más), es
el conjunto de defensas contra el sobreajuste. Sobreajustar significa aprenderse de memoria el
material de entrenamiento y fallar con material nuevo: es el error clásico de este tipo de
proyectos.

## 6. Por qué se usaron las particiones oficiales (y la salvedad de las fotos repetidas)

**El problema.** Las fotos hay que repartirlas en dos grupos: uno para entrenar y otro para
evaluar. Si se evalua con las mismas fotos con las que se entreno, el resultado es un engaño: sería
como aprobar un examen que ya tienes corregido.

**La decisión.** Usar las **particiones oficiales** del catálogo, que vienen ya definidas por sus
autores: 1020 fotos para entrenar, 1020 para validar (que es un grupo intermedio para ir vigilando
si mejora) y 6149 para la prueba final. Los motivos:

- **Comparabilidad.** Cualquier persona que use Flowers-102 usa estas mismas particiones, así que
  las cifras de este proyecto se pueden poner al lado de las publicadas por otros.
- **Reproducibilidad.** No hay que decidir ni sortear quien va a cada grupo: cualquiera obtiene
  exactamente el mismo reparto con el mismo comando.

**La salvedad, contada sin adornos.** Se hizo una comprobación por contenido: se cálculo una huella
digital de cada una de las 8189 imágenes (una huella digital, o suma de verificación, es un código
único que resume el contenido de un archivo: dos archivos idénticos dan el mismo código). Resultado:
hay **4 pares de fotos idénticas**, y **2 de esos pares cruzan particiones** (uno pone la misma
foto en entrenamiento y en prueba, y otro en validación y en prueba).

**Qué significa.** Durante la evaluación final, dos de las 6149 fotos de prueba son fotos que el
modelo ya vio en el entrenamiento. Son 2 imágenes sobre 8189, aproximadamente un **0,02 %**, una
cifra irrelevante para el resultado global. Pero la afirmación que se había escrito inicialmente
(que las particiones separaban las fotos "por planta fotografiada", es decir, que ninguna planta
aparecía en dos grupos) **resultó falsa** y se retiró.

**Que se hizo.** Documentarlo. El informe ahora dice literalmente que hay cuatro pares de imágenes
idénticas byte a byte y que dos cruzan particiones, y que eso invalida cualquier afirmación de
separación estricta. El defecto **no es del programa**: es del catálogo oficial, que publica sus
particiones como listas de números sin comprobar repeticiones de contenido. Lo relevante para quien
lea esto es que se encontró, se midió y se cuenta, en lugar de ocultarlo. Una fuga de datos
silenciada sería un problema mucho más grave que dos fotos repetidas.

## 7. Por qué existe un umbral de confianza y por que acabó convertido en un aviso

**El problema.** El catálogo tiene 102 especies y el modelo solo conoce esas 102. Si le enseñas la
foto de un coche, o de una flor que no está en el catálogo, el modelo igualmente elegira una de sus
102 especies, porque es lo único que sabe hacer. Eso es peor que no responder: da una respuesta
segura y equivocada.

**La solución.** Un **umbral de confianza**: si la probabilidad de la especie más probable no llega
a 0,25 (un 25 %), el sistema no responde ninguna especie y dice **"no reconocida / fuera del
catálogo"**. La idea es que, ante algo desconocido, las probabilidades se reparten entre muchas
especies y la más alta queda baja: esa es la señal de duda que se aprovecha.

**El giro inesperado.** Más adelante se descubrió que este umbral, tal como estaba colocado,
**estaba rompiendo la demostración**: respondía "no reconocida" casi siempre. El motivo está
explicado en el apartado de errores; lo importante aquí es como se corrigió el diseño.

**La decisión final.** El umbral dejó de ser una censura para convertirse en un **aviso**. Ahora la
demostración **siempre** enseña las cinco especies más probables con su porcentaje, y **además**
añade un mensaje de advertencia cuando la confianza no llega al umbral: "confianza baja, puede que
está flor no este en el catálogo, o que el modelo este poco entrenado". Es más útil así, porque el
usuario ve toda la información y decide; antes, el sistema decidía por el y le ocultaba el
resultado. El umbral sigue siendo configurable, así que quien quiera el comportamiento estricto
puede activarlo.

## 8. Por qué todo se comprueba con pruebas automáticas y por que hubo una auditoría adversarial

**Las pruebas automáticas.** Cada pieza del programa tiene sus pruebas: se ejecutan solas, en
minutos, y responden "correcto" o "falló". Son **78**. La razón de tenerlas es muy concreta: alguien
que cambia una línea para arreglar una cosa rompe otra sin enterarse, y el falló aparece mucho
después, en el peor momento. Con pruebas, ese falló aparece en segundos. Un detalle honesto: hay
partes que las pruebas **no** cubren, y eso también está documentado. Por ejemplo, la pantalla
interactiva de demostración no tiene ni una prueba automática, aunque se verificó a mano.

**La auditoría adversarial.** Aquí está la decisión más interesante del proyecto. Se creo un papel
específico: una persona (un revisor independiente) cuyo trabajo **no** era escribir ni mejorar nada,
sino **intentar demostrar que lo que afirmabamos era falso**. Es lo contrario de la actitud
habitual, y precisamente por eso funciona.

Sus instrucciones no eran "revisa que esto este bien", sino "intenta refutar estas diez
afirmaciones". Entre otras cosas:

- Reproducir el recorrido completo **desde cero y con otra configuración**, para comprobar que el
  resultado no depende de una casualidad.
- Comprobar de forma independiente que no había **fuga de datos** (fotos compartidas entre el grupo
  de entrenamiento y el de prueba), sin fiarse de nuestro código: leyendo los ficheros oficiales del
  catálogo por su cuenta y calculando las huellas de las 8189 imágenes.
- Comprobar si cada modelo guardado había usado **realmente** los pesos preentrenados, comparando la
  primera capa de cada uno con los pesos originales de ImageNet.
- Ejecutar el cuaderno entero y comprobar que no cambiaba mientras se ejecutaba.
- Buscar rutas absolutas del ordenador del autor en todo lo que se entrega (un descuido clásico que
  rompe el proyecto en otra máquina y además revela información privada).

El resultado fue **valioso por incordioso**. De las diez afirmaciones, seis se confirmaron y cuatro
quedaron refutadas o parcialmente refutadas. Dos de esas refutaciones eran **errores nuestros**, y
están contados en el apartado de errores. También apareció algo que nadie había notado: ningún
archivo guardaba si los pesos preentrenados se habían cargado de verdad.

La lección, dicha sin rodeos: revisar el propio trabajo con buena voluntad encuentra mucho menos
que encargarle a alguien la tarea de destruirlo. La única condición es que después haya que hacerle
caso, y en este proyecto se hizo.

## 9. Por qué el código "degrada" en lugar de fallar

**El problema.** El proyecto tiene que funcionar en sitios muy distintos: en Colab con GPU, en un
ordenador normal sin GPU, sin conexión a internet, o en un ordenador con restricciones de seguridad
como el que se usó para desarrollarlo. Hay cuatro cosas que **pueden** faltar: internet, GPU,
los procesos auxiliares de carga de datos y el modelo entrenado. Si el programa se cayera cada vez
que falta una de ellas, sería inútil la mitad de las veces.

**La decisión.** Se eligió **degradar**: cuando falta algo, el programa **avisa** y sigue
funcionando como puede, en lugar de detenerse. Degradar es una palabra técnica que aquí significa
"seguir funcionando con menos medios y diciéndolo". Los cuatro casos:

- **Sin internet.** No se pueden descargar las 102 versiones de los pesos. En lugar de caerse, el
  programa entrena igualmente partiendo de pesos al azar, **avisando de que lo está haciendo**. No
  es lo ideal, pero es infinitamente mejor que no hacer nada.
- **Sin GPU.** Todo el cálculo se hace en el procesador normal. Es mucho más lento, pero funciona y
  el resultado es el mismo. El código no tiene ninguna orden que exija GPU obligatoriamente.
- **Sin procesos auxiliares de carga de datos.** El programa reintenta una vez de otra manera y
  avisa. Se explica en el apartado de errores.
- **Sin modelo entrenado.** Si le pides predecir y no hay modelo, no se cae: responde con un mensaje
  claro que además dice **que comando ejecutar** para generarlo. Y la pantalla de demostración
  muestra el aviso "entrena primero" en lugar de quedarse en blanco.

**El límite de esta idea, dicho con honestidad.** Degradar en silencio es peligroso, porque uno
puede creer que obtuvo el resultado bueno cuando en realidad obtuvo el resultado malo. Este proyecto
intentó poner el aviso lo más visible posible, pero el apartado de errores cuenta un caso en el que
el aviso **no se vio** y provocó un malentendido serio. La lección que se saca de ahí es que un
aviso escondido entre el ruido de la pantalla no es un aviso.

## 10. Los errores reales que aparecieron y cómo se resolvieron

Estos son los errores de verdad, contados en el orden en que aparecieron, con el síntoma, la causa y
la solución. Ninguno de ellos es un error de concepto: son los típicos tropiezos de llevarlo a la
práctica.

### 10.a La instalación de librerías fallaba y se resolvió con otra herramienta

**El síntoma.** Al preparar el entorno del ordenador de desarrolló, la instalación de las
bibliotecas fallaba con errores de permisos: el sistema decía "acceso denegado" y no dejaba
continuar. Al principio parecía un problema del instalador.

**La causa.** Se investigó hasta el fondo y resultó ser una limitación del entorno de seguridad de
esa máquina concreta. El instalador habitual (`pip`) crea una carpeta temporal con unos permisos
que, en ese entorno, la dejan sin acceso después, y entonces falla al usarla. Curiosamente, otras
herramientas crean la carpeta de forma distinta y no tropiezan con eso.

**La solución.** Instalar con **otra herramienta** que no usa ese tipo de carpeta temporal, y dejar
el comando anotado en la documentación del proyecto para que nadie pierda una tarde con el mismo
problema. **Esto no afecta a Colab**, donde el problema no existe. Es un ejemplo de un detalle que
depende de la máquina, no del programa, y por eso está documentado aparte: quien lea el error en su
ordenador sabrá que no ha hecho nada mal.

### 10.b Los procesos auxiliares de carga de datos fallaban en Windows

**El síntoma.** El entrenamiento se paraba de golpe, a mitad, con un mensaje de "acceso denegado"
justo cuando empezaba a leer las fotos. En cambió, con las mismas fotos, a veces funcionaba.

**La causa.** Para leer las imágenes rápido, el programa puede usar **varios lectores a la vez**
(fotogramas de trabajo paralelos, o "workers"). Esos lectores se comunican con el programa
principal por un canal interno del sistema operativo, y en Windows, bajó determinadas condiciones de
seguridad, ese canal está bloqueado. Al leer las fotos el programa moría.

**La solución.** En lugar de intentar adivinar de antemano si se puede, el programa **reintenta una
sola vez con un único lector**. Es decir: intenta la vía rápida, y si falla, cambia sola a la vía
lenta y sigue, avisando de que lo hizo. Lo importante es que **no se pierde el entrenamiento** que
ya estaba en marcha. Además se programaron dos pruebas automáticas para esto: una comprueba que el
reintento ocurre cuando fallan los lectores, y otra que **no** se reintenta cuando el error es de
otro tipo (para no ocultar problemas reales tapándolos con un reintento). En Colab y en Linux los
lectores funcionan bien y esta protección ni se activa.

### 10.c Los pesos preentrenados no se descargaban y el modelo entrenaba "en blanco"

Este fue el error más engañoso de todos.

**El síntoma.** Las corridas de prueba daban un acierto bajísimo, del orden del 1 %, que es
exactamente lo que acertaría alguien que respondiera al azar (con 102 opciones, una entre 102 es
aproximadamente 1 %). Parecía que el modelo **no aprendía nada**, y llevó a dudar del diseño
entero: de si el aprendizaje por transferencia servía, de si el número de épocas era el adecuado, de
si el problema era demasiado difícil.

**La causa.** En el ordenador de pruebas, la carpeta donde la biblioteca guarda los pesos
descargados no se podía escribir. Por eso la descarga fallaba, y el programa, fiel a su política de
degradar en lugar de caerse, **entrenaba partiendo de pesos al azar** ("en blanco"): exactamente la
red desde cero que se había decidido evitar. Y aquí está lo grave: **el aviso existía, pero se
perdía** entre los cientos de líneas de mensajes normales del entrenamiento. Además, los archivos
que guardaba el programa declaraban "use pesos preentrenados: si", porque registraban lo que se
había **pedido**, no lo que realmente había ocurrido. O sea: parecía que todo iba bien y el
resultado era malo.

**Como se descubrió.** Lo encontró la auditoría adversarial, y de una forma muy elegante: comparó la
primera capa de cada modelo guardado con los pesos originales de ImageNet. Si el modelo venía de
ImageNet y solo se había ajustado unas épocas, la diferencia debía ser minúscula; si había empezado
al azar, la diferencia tenía que ser del tamaño de los propios pesos. Los números fueron
concluyentes: tres de los modelos guardados tenían **pesos aleatorios** mientras declaraban lo
contrario.

**La solución, en tres partes.**

1. **Entender el alcance real.** El problema era del **entorno de pruebas**, no del programa, y en
   Colab la descarga de pesos funciona normalmente. Eso evito una falsa alarma mayor.
2. **Hacer el aviso imposible de perder.** El cuaderno ya no deja el mensaje técnico crudo: lo
   detecta, lo traduce a lenguaje claro y lo imprime destacado, explicando que el entrenamiento
   sigue siendo válido pero que ha partido de cero.
3. **Registrar la verdad, no la intención.** Se anoto como mejora pendiente guardar en los archivos
   de resultado el valor **real** de si los pesos se cargaron, no el que se pidió, y mostrarlo en
   pantalla. Es la corrección de fondo: el error no fue que fallara la descarga, fue que el
   programa informaba de algo que no había comprobado.

**La lección.** Un aviso que nadie ve no es un aviso, y un dato que dice lo que pediste en lugar de
lo que pasó es peor que no tener dato.

### 10.d El cuaderno no arrancaba por un permiso del sistema

**El síntoma.** Al intentar ejecutar el cuaderno **como cuaderno de verdad** (es decir, con el
programa que abre un cuaderno y va ejecutando sus celdas una a una), el intentó fallaba antes de
empezar, con un error de permisos: el sistema no dejaba crear el proceso interno que ejecuta el
código.

**La causa.** Otra vez una restricción del entorno de seguridad del ordenador de desarrolló, no del
proyecto. Ese proceso interno necesita un canal del sistema operativo que ahí está bloqueado.

**La solución.** Se construyó un **sustituto**: un programa propio que lee el cuaderno, extrae sus
celdas de código y las ejecuta en el mismo orden, en el mismo espacio de trabajo, deteniendose en el
primer error. Con el se comprobó que las 13 celdas de código se ejecutan de principio a fin, en
poco más de minuto y medio (en modo debug), sin errores y sin modificar el cuaderno. La honestidad
obliga a decir lo que ese sustituto **no** cubre: atajos propios de los cuadernos, reinicios del
entorno interno ni las funciones exclusivas de Colab (subir archivos, descargar, conectar Drive).
Por eso esto aparece en la lista de cosas no verificadas: el cuaderno se verificó celda a celda, no
como cuaderno real.

### 10.e La demostración decía "no reconocida" casi siempre

**El síntoma.** La pantalla de demostración respondía "no reconocida / fuera del catálogo" una y
otra vez, incluso con fotos del propio catálogo de flores, y aunque la especie correcta estuviera
entre las propuestas. Parecía que el modelo no servía para nada.

**La causa.** El umbral de confianza (el 25 % que se explicó antes) se aplicaba **antes** de
mostrar el resultado. Y con un modelo poco entrenado, o entrenado sin los pesos preentrenados, las
probabilidades son muy planas: el modelo reparte su duda entre muchas especies y su respuesta más
segura ronda el 2 %, muy por debajo del 25 %. Resultado: el umbral borraba la predicción y lo único
que se veía era el mensaje de rechazo. La causa de fondo era doble: el umbral estaba mal colocado
(borraba información en lugar de advertir) y, en el entorno de pruebas, se sumaba el problema de los
pesos preentrenados del apartado anterior.

**La solución.** Se separaron las dos cosas que el umbral estaba mezclando:

- **Predecir** y **juzgar la confianza** son decisiones distintas. Ahora la demostración pide
  siempre las cinco especies más probables sin filtrar y, **después**, añade un aviso visible si la
  confianza no llega al umbral. El usuario ve siempre el resultado y, además, la advertencia.
- En la celda de predicción del cuaderno se explica el mismo detalle en un comentario, para que
  nadie vuelva a caer en la trampa.

**El detalle revelador.** Este error se detectó precisamente porque alguien miró la pantalla y
preguntó por que respondía siempre lo mismo. Los datos que el programa imprimía eran correctos: el
error estaba en como se presentaba.

### 10.f Dos afirmaciones nuestras resultaron falsas y se corrigieron

La auditoría adversarial refuto dos afirmaciones que el proyecto daba por buenas. Ninguna de las dos
rompía el programa, y las dos importaban, porque en un trabajo académico una afirmación sin
comprobar es un defecto.

**Falsa número 1: el recuento de pruebas.** Circulaba una cifra de pruebas que no correspondía con
la realidad. Los recuentos que se fueron anotando a lo largo del trabajo fueron cambiando (una
revisión, otra, otra), porque las pruebas se estaban escribiendo **en paralelo** mientras se
auditaba; el número final, medido con el comando y con todas las pruebas activadas, es **78**. La
conclusión del auditor fue justa y merece citarse: no era una mentira, era un dato **caducado**.
La corrección no fue solo poner el número bueno, sino dejar por escrito que el número se movió y con
que ajustes se obtuvo cada recuento, para que nadie compare cifras de revisiones distintas y crea
que alguien miente.

**Falsa número 2: la ausencia de duplicados.** El informe afirmaba que las particiones oficiales
separaban las fotos por planta fotografiada, es decir, que ninguna planta aparecía en dos grupos a
la vez. La comprobación por contenido (huellas digitales de las 8189 imágenes) demostró que hay
**cuatro pares de fotos idénticas** y que **dos cruzan particiones**. Es lo que se ha explicado en
el apartado 6: un 0,02 % del total, irrelevante para las cifras, pero la afirmación tal como estaba
escrita era falsa y se retiró, sustituyendola por el dato medido.

**La lección común a las dos.** Ambas eran afirmaciones hechas de buena fe y sin mala intención, y
las dos se cayeron por el mismo motivo: se habían dado por buenas sin medirlas. Por eso el proyecto
terminó con la costumbre de acompañar cada cifra importante de su fuente (el archivo y el dato
exacto de donde sale), en lugar de confiar en la memoria o en la documentación anterior.

## 11. Por qué se documenta lo que NO se ha podido verificar

Al final del informe de verificación hay una lista de cosas que **no** se comprobaron. Podría
parecer una debilidad, y en realidad es lo contrario: es la parte que permite confiar en el resto.
El razonamiento es este.

Si un documento dice "todo verificado" y luego resulta que había cosas sin verificar, **todo** el
documento queda en duda, incluidas las partes que si eran ciertas. En cambió, si dice "esto está
comprobado, esto otro no, y este es el motivo", quien lo lee sabe exactamente sobre que puede
apoyarse. La honestidad sobre los límites no resta crédito: lo reparte correctamente.

Las cosas que quedaron sin verificar son, sobre todo:

- **La corrida completa de 25 épocas en una GPU de Colab.** No había acceso a Colab y la GPU del
  ordenador de desarrolló no se podía usar. Es el único pasó que el proyecto declara pendiente, y
  corresponde a quien reciba el trabajo. Los tiempos (15 a 25 minutos) son una **estimación**, no una
  medición.
- **El uso real de la GPU y de la precisión mixta** (los cálculos con menos decimales que hacen el
  entrenamiento más rápido). Todo lo medido en el desarrolló se hizo en el procesador. Se revisó el
  código línea a línea, pero eso no es lo mismo que verlo funcionar.
- **Las funciones exclusivas de Colab:** conectar Drive, subir el archivo comprimido y descargar
  resultados. Aquí no existen, así que solo se pudo comprobar que están bien protegidas para que un
  falló no rompa nada.
- **La descarga real del catálogo desde cero** en Colab, que aquí ya estaba descargado.
- **La compatibilidad con versiones antiguas** de la biblioteca de la pantalla interactiva.

Conviene decir también lo que **si** se midió, porque es mucho: el recorrido completo entrenar,
evaluar y predecir; una evaluación de las 6149 fotos de prueba, con 37,8 % de acierto exacto y
67,4 % entre las cinco primeras propuestas (con una corrida corta de 4 épocas, que sirve como suelo
de partida y **no** como resultado final); el cuaderno entero ejecutado en modo debug; las 78
pruebas; y las comprobaciones independientes de particiones, pesos preentrenados, rutas y contenido
del paquete.

**Una última advertencia práctica, heredada de todo lo anterior.** Los números de 37,8 % y 67,4 %
**no son el resultado del proyecto**: son lo que se obtuvo con una corrida corta de 4 épocas, hecha
para comprobar que la maquinaria media de verdad. El resultado bueno lo produce la corrida completa
de 25 épocas en Colab, y esa corrida la tiene que lanzar quien reciba el trabajo. El propio cuaderno
imprime, al terminar, un bloque de texto con las cifras reales de esa corrida, listo para copiar en
el informe: no hay que escribir ningún número a mano.

## Resumen del capítulo

- Las **decisiones** (catálogo público, PyTorch, red preentrenada, congelar y descongelar, aumentar
  los datos, suavizar las etiquetas, particiones oficiales, umbral convertido en aviso, degradar en
  lugar de fallar) se tomaron para resolver problemas concretos: pocos datos, poco tiempo de
  cálculo, necesidad de comparar con otros trabajos y necesidad de no dar respuestas falsas.
- Los **errores** fueron reales y de todo tipo: permisos del sistema, procesos auxiliares que
  fallaban, un modelo que entrenaba en blanco sin que se notara, un cuaderno que no arrancaba, una
  demostración que decía siempre lo mismo y dos afirmaciones nuestras que resultaron falsas. Todos se
  investigaron hasta la causa, se corrigieron y se dejaron escritos.
- Lo que **no** se pudo verificar también está escrito, porque un documento que solo habla de sus
  éxitos no se puede usar para decidir en que confiar.
