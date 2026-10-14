# Conceptos básicos: de una foto a una respuesta

> Esta sección está escrita para alguien que nunca ha programado y que no ha estudiado
> inteligencia artificial. No hace falta saber nada de computación para leerla: cada idea se
> explica primero con una comparación cotidiana y solo después se conecta con el proyecto.
> Al final deberías poder explicarle a otra persona, con tus palabras, qué hace el sistema
> "IA flores" cuando le entregas la foto de una flor.

## 1. Qué es una imagen para un ordenador

Cuando miras una foto ves una flor. Un ordenador no ve nada: **solo maneja números**. La
forma que tiene de "ver" la foto es guardarla como una enorme tabla de números, y nada más.

Imagina que divides la foto en una cuadrícula finísima, como un mosaico de azulejos
diminutos. Cada cuadradito se llama **píxel** ("píxel" viene de *picture element*, elemento de
imagen). Una foto de 224 por 224 píxeles, que es el tamaño que usa este proyecto, es una
rejilla de 224 columnas por 224 filas: en total 50 176 cuadraditos.

Cada cuadradito tiene un color, y un ordenador representa el color con tres números: cuánto
rojo, cuánto verde y cuánto azul lleva esa mezcla. Cada uno de esos tres números va de 0 a
255, donde 0 es "nada de ese color" y 255 es "todo ese color". Con esa receta:

- (255, 0, 0) es rojo puro.
- (0, 0, 0) es negro, porque no hay nada de ningún color.
- (255, 255, 255) es blanco, porque están los tres al máximo.
- (255, 160, 200) es un rosa claro, mezcla de mucho rojo y bastante azul.

Si juntamos todo, una foto de 224 por 224 en color es una tabla de números con 224 filas, 224
columnas y 3 valores por casilla. Eso son 150 528 números para una sola foto.

En el proyecto, cuando el programa dice que carga una imagen y la convierte en algo llamado
*tensor*, lo único que hace es leer la foto y colocarla en esa tabla de números, ordenada y
lista para operar con ella. Eso es todo el misterio de "la máquina ve la imagen".

```text
Una foto pequeña de 3 x 3 píxeles, vista como tabla de números
(en cada casilla, los tres canales rojo-verde-azul):

  (255,255,255)  (255,  0,  0)  (255,255,255)
  (  0,255,  0)  (255,255,  0)  (  0,  0,255)
  (128,128,128)  (255,255,255)  (255,255,255)
```

## 2. Qué es clasificar una imagen y qué es una etiqueta

**Clasificar** es elegir una opción de una lista cerrada. Cuando ordenas tu ropa por
"camisetas, pantalones, abrigos" estás clasificando: miras la prenda y decides a qué cajón va.
No describes la prenda entera, solo dices a qué grupo pertenece.

Clasificar una imagen es exactamente eso: mirar la foto y decidir a cuál de las opciones de la
lista pertenece. Cada una de esas opciones se llama **etiqueta** o **clase**. Las dos palabras
significan lo mismo en este contexto: el nombre del grupo.

La lista de clases de este proyecto tiene **102 etiquetas**, porque el catálogo que usamos es
el del dataset público *Oxford Flowers-102*, que contiene **102 especies de flores**. La
respuesta del sistema siempre tiene que salir de esa lista: no puede inventarse una especie
nueva ni responder "no sé".

Y aquí aparece un matiz que conviene entender desde el principio, porque no es lo mismo que
clasificar la ropa:

- Ordenar una camiseta es fácil: la etiqueta es "camiseta".
- Distinguir la especie número 47 de la especie número 48 puede ser difícil **incluso para una
  persona**, si las dos flores son casi idénticas salvo por el número de pétalos o el tono.

> Clasificar no es "entender" la foto. Es elegir una etiqueta de una lista cerrada. Un sistema
> puede acertar mucho y, aun así, no tener ni idea de botánica.

En el código, cada etiqueta es simplemente un número del 0 al 101, y al final se traduce al
nombre de la especie para mostrarlo.

## 3. Qué es un modelo de inteligencia artificial

Un **modelo** es la parte del sistema que sabe hacer la tarea. No es un programa con
instrucciones escritas a mano: es una gran cantidad de números ajustables que, combinados,
producen una respuesta.

### La analogía del aprendiz

Piensa en alguien que quiere aprender a reconocer flores y no tiene maestro de botánica, pero
sí tiene un montón de fotos ya etiquetadas con el nombre correcto. El aprendiz hace lo
siguiente:

1. Mira una foto y arriesga un nombre.
2. Compara su respuesta con el nombre correcto que viene en la ficha.
3. Anota en qué se equivocó.
4. Ajusta un poquito su forma de mirar, para acercarse la próxima vez.
5. Repite esto miles de veces.

Ese ciclo es todo el aprendizaje automático. Al principio el aprendiz acierta por casualidad,
una vez de cada cien. Después de ver muchas fichas, empieza a fijarse en detalles útiles
(color del centro de la flor, forma de los pétalos, hojas) y su porcentaje de acierto sube.

Un detalle importante: el aprendiz **no guarda las fotos** que ve. Lo que guarda son los
ajustes de su forma de mirar. Eso es el modelo entrenado: un archivo con esos números
ajustados. En este proyecto ese archivo se llama *checkpoint* y ocupa muy poco (unos pocos
megas), muchísimo menos que las 8189 fotos del catálogo.

### Reglas a mano frente a aprender de ejemplos

Hay dos maneras de conseguir que un ordenador reconozca flores, y la diferencia es la clave de
todo el campo.

**Escribir reglas a mano.** Tú escribes las instrucciones. Por ejemplo: "si la flor tiene más
de 20 pétalos y el centro es amarillo, es la especie A". Este enfoque es transparente y
controlable, pero se rompe enseguida: las flores cambian de forma según el ángulo, la luz, la
estación del año y la cámara. Harían falta millones de reglas y ni así funcionaría, porque
nadie es capaz de escribir en palabras qué hace exactamente que dos flores se parezcan.

**Aprender de ejemplos.** Tú no escribes ninguna regla sobre flores. En su lugar le das al
modelo miles de fotos con su nombre correcto y dejas que él mismo descubra qué detalles
distinguen una especie de otra. Tú solo decides cómo se le presentan los ejemplos y cómo se
mide su error.

| | Reglas a mano | Aprender de ejemplos |
|---|---|---|
| Quién decide las características | La persona que programa | El propio modelo |
| Cuánto trabajo humano | Muchísimo, y nunca termina | Preparar datos y configurar |
| Si aparecen flores nuevas o fotos raras | Suele fallar | Suele aguantar mejor |
| Se puede explicar fácilmente | Sí, es una lista de reglas | No del todo, es un cuadro de números |

En este proyecto no hay ni una sola regla escrita sobre pétalos, colores de flor ni hojas. Todo
lo que el sistema sabe sobre flores lo aprendió de los ejemplos.

## 4. Qué es una red neuronal y qué es una red convolucional

### Una red neuronal, en palabras simples

Una **red neuronal** es una forma concreta de organizar esos números ajustables. La imagen se
descompone en señales pequeñas, esas señales se combinan en señales un poco más complejas, y
esas a su vez en otras más complejas todavía, en capas sucesivas, hasta que al final sale una
lista de 102 puntuaciones: una por especie.

Una comparación útil es una cadena de montaje de una cocina. En la primera estación se cortan
las verduras. En la segunda se mezclan. En la tercera se sazona. En la cuarta se emplata. Nadie
de esas estaciones "sabe" qué plato es; cada una hace su parte y pasa el resultado a la
siguiente. En una red neuronal, las estaciones son las **capas** y el resultado que cada una
pasa a la siguiente es un conjunto de números.

La analogía del aprendiz del apartado anterior sigue valiendo: "ajustar su forma de mirar" es
literalmente retocar los números de todas esas capas, un poquito en cada ejemplo.

### La red convolucional: una lupa que recorre la foto

Una **red neuronal convolucional** (en inglés *convolutional neural network*, de ahí la
abreviatura CNN) es una red neuronal especializada en imágenes. Su idea central es la
**convolución**, y la mejor forma de imaginarla es una lupa pequeña que se desliza por encima
de la foto.

Imagina una lupa del tamaño de 3 por 3 píxeles. La apoyas en la esquina superior izquierda y
miras solo ese trocito diminuto. Anotas una conclusión sencilla, por ejemplo "aquí hay un
borde que va de claro a oscuro y apunta hacia abajo". Luego mueves la lupa un píxel a la
derecha y repites. Y así, deslizándola por toda la imagen. Al terminar tienes un mapa que dice
en qué zonas de la foto aparece ese tipo de borde.

Ahora usas muchas lupas a la vez, cada una especializada en algo distinto: una detecta bordes
horizontales, otra manchas de color intenso, otra curvas suaves. Después pones una segunda
ronda de lupas que ya no mira la foto original, sino esos mapas: "donde hay un borde curvo
rodeado de una mancha amarilla, hay un pétalo". Y una tercera ronda que combina pétalos:
"muchos pétalos amarillos alrededor de un centro oscuro". De lo simple a lo complejo.

Esto explica por qué funciona tan bien en fotos:

- **Reconoce el rasgo sin importar dónde esté.** Si la flor está en el centro o en una esquina,
  la lupa la encuentra igual, porque recorre toda la imagen. Esto ahorra muchísimo trabajo: la
  misma lupa sirve para cualquier posición.
- **Aprende los rasgos por sí sola.** Nadie le dijo "busca bordes" ni "busca pétalos". La forma
  de cada lupa es uno de esos números ajustables, y el aprendiz los va afinando con los
  ejemplos.

El modelo concreto que usa este proyecto se llama **EfficientNet-B0**. Es una red convolucional
conocida y bastante eficiente, pensada para dar buenos resultados gastando pocos recursos. No
hace falta recordar el nombre: basta con saber que es una red de lupa deslizante, con muchas
capas, y que ya viene con experiencia previa (lo veremos en el apartado 7).

## 5. Entrenar: épocas, lotes, función de pérdida y learning rate

**Entrenar** es el proceso de aprendizaje del apartado 3: pasarle al modelo los ejemplos
etiquetados una y otra vez, medir sus errores y ajustar sus números internos. Terminado el
entrenamiento, el modelo queda guardado y ya no necesita aprender más: solo responde.

Vamos con las piezas, una por una.

### Época

Una **época** es una vuelta completa por todos los ejemplos de entrenamiento. Como estudiar un
libro de texto de principio a fin: no basta con leerlo una vez, se repasa varias veces y en
cada repaso se entiende mejor.

Este proyecto entrena **25 épocas**. Con muy pocas se queda corto de aprendizaje; con
demasiadas, empieza a memorizar (el problema del apartado 6). Veinticinco es un punto razonable
para este catálogo.

### Lote

Si le mostraras al modelo las 1020 fotos de entrenamiento de golpe antes de corregir nada,
sería lentísimo y la corrección sería torpe. En su lugar se le dan de a poquito: un **lote**
(en inglés *batch*) es un grupo pequeño de imágenes que se procesan juntas y que provocan un
único ajuste conjunto.

Aquí el lote es de **32 imágenes** de 224 por 224 píxeles. Piensa en un profesor que corrige 32
exámenes, saca la conclusión general de ese montón y ajusta su manera de explicar antes de
pasar al siguiente montón. Cada lote produce una corrección pequeña.

### Función de pérdida

La **función de pérdida** es el marcador que mide qué tan mal lo está haciendo el modelo. Le
das la respuesta correcta y la respuesta que dio el modelo, y te devuelve un número: cuanto más
grande, peor.

Es el marcador de un partido. Sin marcador no puedes saber si vas ganando, y tampoco sabes
hacia dónde moverte. Toda la corrección de la red se guía por este número: se ajustan los
números internos en la dirección que hace bajar la pérdida. Pérdida que baja de forma sostenida
significa que el aprendiz está mejorando de verdad.

### Learning rate

El **learning rate** (ritmo de aprendizaje) es el tamaño del paso que das al corregir. Y aquí
la analogía es la de bajar una cuesta con niebla:

- Si das pasos **demasiado grandes**, pasas de largo el fondo del valle, rebotas de un lado al
  otro y nunca te asientas.
- Si das pasos **demasiado pequeños**, el descenso es seguro pero tardas una eternidad y puede
  que te quedes atascado a mitad de camino.

En este proyecto se usan dos ritmos distintos a la vez, uno más lento para la parte que ya tiene
experiencia (el *backbone*, del apartado 7) y otro más rápido para la parte nueva que acaba de
nacer (la cabeza que decide entre las 102 especies). La idea: lo que ya sabe bien se retoca con
delicadeza, y lo que es nuevo se aprende con decisión.

### Qué significa que el modelo "aprenda"

Decir que el modelo **aprende** significa, ni más ni menos, que sus números internos se van
ajustando de manera que la pérdida baja y el porcentaje de aciertos sube. No hay comprensión ni
conciencia de por medio: hay números que cambian en la dirección correcta, miles de millones de
veces.

Hay una señal de que está aprendiendo bien y es fácil de reconocer: en el historial de
entrenamiento, la pérdida de entrenamiento baja y la de validación también baja. Si la primera
baja y la segunda empieza a subir, lo que está pasando es el problema del apartado siguiente.

> El entrenamiento completo de este proyecto (25 épocas, lotes de 32 imágenes de 224 por 224)
> está pensado para tardar entre 15 y 25 minutos en una GPU T4, la tarjeta gráfica gratuita
> que ofrece Google Colab.

## 6. Sobreajuste y los tres conjuntos de datos

### Sobreajuste: memorizar en vez de aprender

El **sobreajuste** (en inglés *overfitting*) es el riesgo más importante de todo el aprendizaje
automático. La analogía clásica es el estudiante que se aprende las respuestas del examen de
memoria, sin entender la materia.

Ese estudiante saca un 10 en el examen exacto que memorizó. Pero si le cambias los números del
problema, o le pones un ejercicio con otras palabras, se queda en blanco: no aprendió el
método, aprendió la lista. En un modelo pasa exactamente lo mismo. Si memoriza las fotos de
entrenamiento, acierta casi todas esas fotos y falla las que nunca vio.

Es especialmente peligroso en este proyecto porque hay **muy pocos ejemplos por clase**: el
conjunto oficial de entrenamiento de Flowers-102 tiene 1020 fotos para 102 especies, es decir
**10 fotos por especie**. Con tan pocos ejemplos es fácil memorizar en lugar de aprender, y por
eso el proyecto usa varias defensas: aumento de datos (apartado 8), parada temprana y
evaluación sobre conjuntos que el modelo nunca vio.

### Entrenamiento, validación y prueba

Para detectar el sobreajuste hay que examinar al modelo con fotos que no haya visto. Por eso
los datos se reparten en tres grupos con papeles distintos. En este proyecto se usan los
repartos **oficiales** del dataset Oxford Flowers-102, no unos inventados:

| Conjunto | Para qué sirve | Fotos |
|---|---|---|
| Entrenamiento (train) | Son los ejemplos de los que el modelo aprende | 1020 |
| Validación (val) | Es el ensayo general: sirve para tomar decisiones durante el entrenamiento | 1020 |
| Prueba (test) | Es el examen final, se usa una sola vez y nunca se toca antes | 6149 |

En total son **8189 fotos** repartidas en 102 especies.

La lógica de los tres conjuntos:

1. **Entrenamiento** funciona como los ejercicios que se hacen en casa: se pueden consultar las
   soluciones, y de hecho se consultan en cada lote.
2. **Validación** funciona como el simulacro de examen. No se usa para aprender directamente,
   sino para decidir cuándo parar y qué versión guardar. Es el conjunto que revela si el modelo
   está empezando a memorizar: cuando el acierto en entrenamiento sube y el de validación deja
   de subir, esa es la señal de alarma y se detiene el entrenamiento.
3. **Prueba** funciona como el examen final que no se ha visto nunca. Si el modelo rinde bien
   ahí, la confianza es legítima. Si el examen final fuera el mismo que el simulacro, el
   resultado dejaría de ser creíble, porque se habrían tomado decisiones mirándolo.

> Regla de oro: el conjunto de prueba no se toca hasta el final. Usarlo para decidir cosas
> durante el entrenamiento es como hacer trampa en el examen y luego presumir la nota.

## 7. Transfer learning: la clave de este proyecto

El **transfer learning** (aprendizaje por transferencia) es la idea central que hace que este
proyecto funcione con solo 10 fotos por especie. La analogía es directa:

Imagina a alguien que ya sabe distinguir mil objetos distintos: latas, ruedas, plumas, hojas,
monedas, tejidos, cielos, animales. Sabe reconocer bordes, texturas, curvas, brillos y sombras
porque lleva años practicando con miles de objetos. Ahora le pides que aprenda a nombrar 102
especies de flores.

Esa persona no empieza de cero. No necesita volver a aprender qué es un borde ni qué es una
textura. Solo necesita aprender **cómo se llaman** las combinaciones de rasgos que ya sabe ver:
"esta combinación de pétalos y hojas, que ya reconozco, se llama especie 47". Con diez ejemplos
por flor basta, porque lo difícil (ver) ya está resuelto.

Eso es exactamente lo que hace el proyecto: parte de **EfficientNet-B0 preentrenado con
ImageNet**, un banco público de **1,2 millones de fotos de 1000 categorías** ajenas a las flores.
El modelo llega sabiendo ver; lo único que hay que enseñarle es la lista de nombres de flores.

Los pasos, en palabras llanas:

1. **Se toma el experto prestado.** Se usa esa red que ya sabe mirar imágenes, con todos sus
   números ya ajustados por 1,2 millones de fotos.
2. **Se le cambia la última pieza.** La parte final, que antes decía "esto es un perro" o "esto
   es una bicicleta", se sustituye por una pieza nueva que puede decir 102 nombres. En el
   proyecto esa pieza nueva termina en una decisión entre 102 opciones.
3. **Primero se entrena solo la pieza nueva.** Al principio el experto se queda quieto y solo la
   pieza nueva aprende los nombres. Esto es rápido y seguro.
4. **Después se afina todo, con cuidado.** Luego se permite que el experto también se retoque,
   pero con pasos muy pequeños, para afinar su manera de mirar hacia las flores sin destrozar lo
   que ya sabía. Esto es lo que en el código se llama *descongelar* el backbone.

Beneficios concretos de este enfoque:

- **Menos datos.** Con 10 fotos por especie sería inviable aprender de cero, pero es suficiente
  para afinar un experto.
- **Menos tiempo.** El entrenamiento completo cabe en 15 a 25 minutos de GPU T4.
- **Mejor resultado.** El modelo aprovecha patrones visuales generales que le costaría muchísimo
  redescubrir solo con flores.

Esa es la razón de que el proyecto sea viable como trabajo de curso: no entrena un sistema desde
cero, sino que aprovecha el trabajo de otros y lo especializa.

## 8. Aumentación de datos

La **aumentación de datos** (en inglés *data augmentation*) consiste en fabricar variantes
artificiales de las fotos de entrenamiento. La analogía: si solo tienes diez fotos de una
especie y quieres que el aprendiz la vea desde muchos ángulos, recortas, giras y cambias la luz
de esas fotos para obtener versiones distintas sin necesidad de salir a tomar más fotos.

Aquí está la clave: son variantes que **no cambian la respuesta correcta**. Si giras la foto de
una rosa, sigue siendo una rosa, y su etiqueta sigue siendo la misma. El aprendiz aprende así
que la especie no depende del encuadre, del lado hacia el que mira ni del brillo, sino de la
forma de la flor. Dicho de forma técnica: se le enseña a ser **invariante** a esas
transformaciones.

Las tres variantes que aplica este proyecto son:

**RandomResizedCrop.** Recorta un trozo de la foto al azar y lo agranda hasta los 224 por 224
píxeles que el modelo espera. En este proyecto el recorte cubre entre el 60 % y el 100 % del
área original, así que a veces se ve la flor casi entera y a veces solo una parte con el centro
desplazado. Es como fotografiar la misma flor desde más cerca o desde otro encuadre. Sirve
porque en la vida real nadie encuadra la flor exactamente igual que en el catálogo.

**Volteo horizontal (*flip*).** Se refleja la foto como en un espejo, de izquierda a derecha.
Una flor no tiene "mano derecha": la versión reflejada sigue siendo la misma especie. Sirve
porque duplica los puntos de vista sin costo alguno.

**RandAugment.** Es un paquete de muchas pequeñas transformaciones de aspecto, aplicadas al azar
y con intensidad aleatoria: cambios de brillo, contraste, saturación, tono, giros leves,
desenfoques, ruido. En lugar de elegir a mano una receta fija, se deja que la receta varíe en
cada lote. Sirve porque las fotos reales tienen luces, cámaras y condiciones muy variadas, y el
modelo debe aprender a no depender de ninguna en particular.

> Detalle importante: la aumentación se aplica **solo durante el entrenamiento**. Cuando el
> sistema evalúa o responde a una foto nueva, la imagen se prepara de forma sencilla y estable
> (se redimensiona y se recorta al centro, sin trucos). Si se aumentara también al evaluar, la
> nota final sería un sinsentido, como dejar que el alumno cambie las preguntas del examen.

Efecto práctico: la aumentación es una de las defensas contra el sobreajuste. Con 10 fotos por
especie, ver "muchas versiones" de cada foto equivale a tener más ejemplos, y además ejemplos
que le enseñan lo que no debe importar.

## 9. Cómo se mide si el modelo es bueno

Un modelo no es bueno porque "parece que funciona". Hay que medirlo, y para eso se usan
métricas claras sobre el conjunto de prueba, que el modelo nunca vio.

### Exactitud top-1

Es la métrica más directa: de cada 100 fotos, en cuántas la especie más votada por el modelo es
la correcta. Si el sistema pone la respuesta correcta en primer lugar 37,8 veces de cada 100,
decimos que su exactitud **top-1** es **37,8 %**.

Es como un examen de opción múltiple donde solo cuenta la primera casilla marcada.

### Exactitud top-5

Aquí la regla es más generosa: se considera acierto si la especie correcta está **entre las
cinco más probables** que propone el modelo, sin importar en qué puesto. Con las mismas fotos,
el resultado medido de este proyecto es **67,4 %** en top-5.

Es como un examen donde puedes dar cinco respuestas y basta con que la buena esté entre ellas.

### Por qué con 102 clases el top-5 tiene sentido

Con 102 especies hay que juzgar el resultado contra el azar, y el azar es durísimo:

| Métrica | Azar puro con 102 clases | Este sistema (4 épocas) |
|---|---|---|
| Top-1 | 1 de 102, es decir 0,98 % | 37,8 % |
| Top-5 | 5 de 102, es decir 4,9 % | 67,4 % |

Es decir, el sistema está muy por encima de tirar una moneda de 102 caras. Además hay una razón
práctica: varias de las 102 especies son casi idénticas a simple vista, y los errores se
concentran justo ahí. Para una persona que quiere saber qué flor tiene en la mano, recibir una
lista corta de cinco candidatos con su probabilidad es mucho más útil que una única respuesta
que puede ser la equivocada. El top-5 convierte una respuesta seca en una sugerencia útil.

### Macro-F1

La exactitud tiene un problema: cuenta todas las fotos por igual, y puede esconder un desastre
en algunas especies. Imagina un modelo que acierta perfecto en 90 especies y falla siempre en
las otras 12; la exactitud global puede verse aceptable mientras hay 12 especies para las que el
sistema es inútil.

**Macro-F1** corrige eso. Se calcula una nota de calidad para cada una de las 102 especies
combinando dos cosas: cuántas veces la especie correcta fue efectivamente detectada, y cuántas
veces el sistema la eligió sin serlo. Después se **promedia dando el mismo peso a cada
especie**, sin importar si tiene muchas o pocas fotos en el conjunto de prueba.

La analogía es el boletín de notas frente al promedio del curso. El promedio del curso puede ser
bueno aunque a ti te vaya mal; el boletín te obliga a mirar asignatura por asignatura. Con
macro-F1, una especie olvidada arrastra la nota hacia abajo y queda a la vista. Esto es
especialmente útil en Flowers-102, donde la cantidad de fotos por especie no es uniforme.

### Matriz de confusión

La **matriz de confusión** es una tabla de 102 filas por 102 columnas. Las filas son la especie
verdadera y las columnas, la especie que el sistema respondió. La casilla donde la fila y la
columna coinciden es un acierto; cualquier otra casilla ocupada es un error, y la tabla dice
exactamente **de qué especie a qué especie** se confundió.

Es como un cuadro que anota, en cada partido, no solo quién ganó, sino a quién le ganó. Visto
entero, revela patrones que un solo número esconde: por ejemplo, que las confusiones se
concentran en tres parejas de especies casi gemelas, o que dos especies se confunden
sistemáticamente entre ellas en las dos direcciones.

Es una herramienta de diagnóstico, no una nota: no se aprueba o se reprueba con ella, se
aprende qué arreglar. Si dos especies se confunden mucho, eso orienta la siguiente mejora
(conseguir más fotos de esas dos, revisar si las etiquetas del catálogo son ambiguas).

### Las cifras de este proyecto

Las cifras reales que se citan aquí provienen de una corrida de prueba del sistema, y conviene
leerlas con honestidad:

| Métrica | Resultado | Contexto |
|---|---|---|
| Exactitud top-1 | 37,8 % | 6149 fotos del conjunto oficial de prueba |
| Exactitud top-5 | 67,4 % | Las mismas fotos |
| Macro-F1 | 0,353 | Promedio por especie, con clases de desempeño desigual |

Esos números se obtuvieron con **solo 4 épocas** de las 25 previstas, es decir con el
entrenamiento interrumpido a propósito para verificar que todo el montaje funciona. En los
artículos científicos que usan este mismo catálogo y este tipo de red, el resultado reportado
por **otros autores** ronda el **92 % de exactitud top-1** con el entrenamiento completo. Ese
92 % no es un resultado de este proyecto ni una promesa: es la referencia externa que sirve
para juzgar si una corrida propia salió razonable. Lo que sí está medido aquí es que el sistema
ya aprende algo real y muy por encima del azar incluso a medio entrenar, después de solo
4 épocas.

> Además de las métricas, el proyecto se comprueba con **78 pruebas automáticas** que verifican
> que cada pieza hace lo que promete (por ejemplo, que el top-5 nunca puede ser peor que el
> top-1, o que el umbral de confianza responde "no reconocida" cuando corresponde). Esas pruebas
> no miden la calidad de las predicciones: miden que la maquinaria está bien armada.

## 10. GPU, Google Colab y Gradio

### GPU: la potencia extra

La **GPU** (unidad de procesamiento gráfico) es un componente que nació para dibujar videojuegos
y que resulta ideal para estos cálculos, porque hace muchísimas operaciones sencillas en
paralelo. Un modelo convolucional es exactamente eso: millones de operaciones simples, todas
parecidas, y todas independientes entre sí.

La analogía: una persona cortando verduras es lenta; veinte personas cortando verduras a la vez
son rapidísimas. Multiplicar números para una red neuronal es cortar verduras. La CPU, el
procesador normal, es el chef que hace todo en orden y con mucha habilidad; la GPU es el equipo
grande de cocina que hace lo repetitivo en paralelo.

Efecto medible: un entrenamiento que en un portátil común puede tardar horas, en una GPU T4
tarda **15 a 25 minutos**.

### Google Colab: la cocina prestada

**Google Colab** es un servicio gratuito de Google que presta un ordenador en la nube, con su
GPU, y te deja ejecutar el proyecto desde el navegador. No hay que instalar nada en tu
computadora ni comprar hardware.

Es la cocina prestada: no tienes el equipo en casa, pero puedes entrar, cocinar y llevarte el
plato. Dos consecuencias prácticas que conviene tener presentes:

- La sesión no es eterna. Si te distraes o se corta la conexión, el ordenador prestado se apaga
  y se pierde lo que estaba solo en memoria. Por eso los resultados se guardan en archivos
  (el *checkpoint* y las métricas) y por eso el notebook recomienda guardarlos también en tu
  Drive, que equivale a llevarte la comida a casa antes de que cierren la cocina.
- Hay que comprobar que la GPU está encendida antes de empezar; si no lo está, todo funciona
  igual pero muchísimo más lento.

### Gradio: el mostrador

**Gradio** es una herramienta que crea, en unas pocas líneas, una página web sencilla con un
botón para subir una foto. Es el mostrador donde el usuario final entrega su imagen y recibe la
respuesta.

La analogía completa queda así: el modelo es la receta, la GPU es el equipo de cocina potente,
Colab es la cocina prestada donde se cocina, y Gradio es el mostrador donde tú entregas la foto
de tu flor y te devuelven una lista de cinco especies posibles con su probabilidad.

Gracias a Gradio, la persona que usa el sistema no necesita saber programar ni abrir ningún
archivo de código: arrastra una foto, pulsa un botón y lee el resultado.

## 11. Qué es el umbral de confianza

El **umbral de confianza** es una línea de corte que el sistema aplica a su propia seguridad
antes de responder. En este proyecto se llama `min_conf` y su valor es 0,25, es decir un 25 %.

Durante una predicción, el modelo reparte su opinión entre las 102 especies en forma de
probabilidades, algo parecido a los porcentajes de una encuesta. Sumando todo da el 100 %. Si la
especie mejor posicionada se lleva el 62 %, el sistema está bastante seguro. Si la mejor se
lleva el 11 %, está prácticamente adivinando entre varias posibilidades.

La regla es: **si la probabilidad de la mejor candidata no llega a 0,25, el sistema no elige
ninguna especie** y responde "no reconocida / fuera del catálogo" en lugar de forzar una
respuesta.

### Por qué existe

La razón es que el catálogo tiene exactamente 102 especies y el mundo tiene muchísimas más
flores. Si le entregas la foto de una flor que no está en la lista, de una planta, de una
persona o de una pared, el modelo **igual dará 102 probabilidades**, porque su salida siempre
tiene 102 casillas y no puede decir "no sé" por sí mismo. Sin umbral, elegiría la menos mala de
esas 102 opciones y te la presentaría con toda tranquilidad. Eso es peor que no responder: es
afirmar algo falso con apariencia de seguridad.

El umbral es, entonces, la única manera de que el sistema **admita que no sabe**. La analogía:
un médico que no está seguro de un diagnóstico no debe elegir una enfermedad al azar; debe decir
"con estos datos no puedo concluir, hace falta otra prueba". El umbral es esa prudencia
incorporada.

El valor 0,25 no es mágico y no está escondido en el código: es un parámetro configurable, y
subirlo o bajarlo cambia el carácter del sistema.

- **Umbral más alto** (por ejemplo 0,60): el sistema responde menos veces, pero cuando responde
  suele estar más seguro. Se pierden respuestas correctas junto con las dudosas.
- **Umbral más bajo** (por ejemplo 0,10): el sistema responde casi siempre, incluidos los casos
  en que no tiene ni idea, y por tanto aumenta el riesgo de afirmar algo falso.

Es un equilibrio entre dos errores: quedarse callado cuando sí sabía, o hablar cuando no sabía.
El valor 0,25 es una elección razonable para una demo, no una verdad universal.

> Como dato útil: la propia interfaz muestra siempre las cinco candidatas con su probabilidad.
> Así, incluso cuando la confianza es baja, la persona que usa el sistema ve que el modelo está
> dudando y puede juzgar por sí misma, en lugar de recibir una respuesta seca y aparentemente
> segura.

## Resumen en una frase

Una foto es una tabla de números; el sistema aprendió, mirando miles de ejemplos ya etiquetados
y aprovechando la experiencia que otro modelo adquirió con 1,2 millones de fotos, a decir a cuál
de las 102 especies del catálogo se parece más; cuando no está seguro, lo admite, y cuando sí lo
está, muestra sus cinco mejores candidatas con su probabilidad para que tú decidas.
