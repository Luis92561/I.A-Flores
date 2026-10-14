# Guía práctica: cómo usarlo sin saber programar

## Para qué sirve esta guía

Esta guia explica como poner en marcha el reconocedor de flores del proyecto **IA flores** usando solamente el navegador. No hace falta saber programar, ni instalar programas, ni tener un ordenador potente: todo el trabajo pesado lo hace un ordenador de Google, a traves de una herramienta gratuita que se llama **Google Colab**.

Al terminar tendras tres cosas: un modelo entrenado que reconoce 102 especies de flores, una demostracion interactiva donde subes una foto y ves el resultado, y un archivo comprimido con todos los resultados para adjuntar al informe.

## Qué necesitas antes de empezar

- **Una cuenta de Google** (la misma de Gmail). Es gratis y sirve tanto para Colab como para Drive.
- **Un navegador actualizado**, por ejemplo Chrome, Edge o Firefox. Nada mas: no hay que descargar ningun programa.
- **El archivo del proyecto**, que se llama IA_flores_colab.zip. Es el paquete que te entrega el equipo que desarrollo el proyecto. Si no lo tienes, pidelo antes de empezar: es lo unico que hay que subir a mano.
- **Un rato de tiempo.** Unos 5 minutos para la prueba rapida y entre 20 y 30 minutos para la corrida completa, contando la primera descarga de imagenes.

> Nota: no hace falta tarjeta de credito ni instalar nada. El acelerador gratuito de Google, la GPU, se activa con un par de clics y no se paga.

## Lo que vas a hacer, en cinco líneas

1. Abrir Google Colab en el navegador.
2. Activar la GPU gratuita.
3. Subir el paquete del proyecto, que se llama IA_flores_colab.zip.
4. Ejecutar todo y esperar.
5. Mirar los resultados, probar la demostracion con una foto y descargar el archivo comprimido con todo.

## Las dos corridas: primero la rápida, después la completa

El cuaderno (asi se llama el documento de trabajo de Colab) puede funcionar en dos velocidades. Conviene hacerlas en este orden y no al reves.

| Corrida | Vueltas sobre los datos | Cuanto tarda | Para que sirve | Fiabilidad de las respuestas |
|---|---|---|---|---|
| Rapida (modo de prueba, 2 epocas) | 2 | Unos minutos | Comprobar que todo funciona: la descarga, el modelo, la evaluacion y la demo | Baja: las respuestas no son fiables |
| Completa (25 epocas) | 25 | 15 a 25 minutos en una GPU T4 | Obtener las cifras buenas y un modelo que reconoce de verdad | Alta |

**Por que en este orden.** La corrida rapida es un ensayo barato. Si falta un archivo o algo esta mal configurado, por ejemplo la GPU sin activar, lo descubres en un par de minutos en lugar de despues de media hora. Ademas, la primera vez hay que descargar las imagenes (330 MB), y esa descarga ocurre igual durante la corrida rapida; asi, cuando lanzas la corrida completa los datos ya estan en su sitio.

**Muy importante.** Lo que devuelve la corrida rapida no sirve para el informe ni para sacar conclusiones. Sirve para comprobar que la maquinaria esta bien montada. Los numeros buenos salen siempre de la corrida completa.

## Paso a paso numerado en Google Colab

1. **Abre Google Colab.** Busca Google Colab en tu buscador habitual y entra en la primera pagina oficial que aparezca, o escribelo directamente en la barra de direcciones si ya conoces la direccion. Inicia sesion con tu cuenta de Google. Veras una ventana con una lista de cuadernos recientes y un boton para crear uno nuevo.
2. **Crea un cuaderno nuevo.** Pulsa el boton de cuaderno nuevo. Se abre una pagina con una caja vacia arriba: esa caja es la primera celda, y a su izquierda hay un boton redondo de reproduccion. Una celda es una caja donde se escribe una orden; en este proyecto no escribiras ninguna, solo las ejecutaras.
3. **Activa la GPU.** En el menu superior entra en Entorno de ejecucion y despues en Cambiar tipo de entorno de ejecucion. En la ventana que aparece, busca Acelerador de hardware, elige T4 GPU y pulsa Guardar. Si Colab avisa de que la sesion se reiniciara, acepta. Este es el paso mas importante de toda la guia: sin GPU, la corrida completa puede tardar horas en lugar de minutos.
4. **Sube el paquete del proyecto.** En la barra lateral izquierda de Colab hay un icono con forma de carpeta. Pulsalo, pulsa despues el boton de subir y elige el archivo IA_flores_colab.zip de tu ordenador. Espera a que la barra de progreso termine: son unos pocos megabytes y sube rapido. Si el paquete no aparece, comprueba que estas en la pestana de archivos y no en la de busqueda.
5. **Abre el cuaderno que va dentro del paquete.** Dentro del paquete hay una carpeta llamada notebooks y, dentro, el cuaderno del proyecto. En el menu Archivo elige Subir cuaderno y selecciona ese archivo: se abrira en Colab con todas las celdas ya escritas. Si no encuentras esa opcion, descomprime el paquete en tu ordenador y sube el cuaderno igual que subiste el paquete.
6. **Ejecuta todo.** En el menu Entorno de ejecucion elige Ejecutar todo. Colab empezara a ejecutar las celdas una detras de otra y a mostrar los resultados debajo de cada una. Junto a la celda que esta trabajando en ese momento vera un circulo girando; cuando termina, aparece una marca de visto bueno.
7. **Espera la descarga la primera vez.** Las primeras celdas comprueban el entorno y preparan el respaldo en Drive. Despues el sistema descarga las fotografias de flores, que ocupan 330 MB. Esta descarga solo ocurre la primera vez de cada sesion; si tienes Drive conectado, las sesiones siguientes las encuentran ya preparadas.
8. **Deja que entrene.** Cuando empieza el entrenamiento van apareciendo lineas de texto, una por cada vuelta completa sobre los datos. En la corrida rapida son 2 lineas; en la completa, hasta 25. Cada linea muestra el castigo por los errores y la proporcion de aciertos. No tienes que hacer nada, solo esperar sin cerrar la pestana.
9. **Lee el resumen y prueba la demo.** Al final el cuaderno imprime un resumen con las cifras y despues abre la demostracion. En la demostracion puedes subir una foto de una flor o hacerla con la camara, y veras las cinco especies mas probables con su porcentaje.
10. **Descarga los resultados.** La ultima celda crea un archivo comprimido llamado resultados_flores.zip y lanza su descarga automaticamente. Dentro esta todo: el modelo entrenado, las cifras, las figuras y el resumen listo para el informe.

> Nota: si Colab te pide permiso para acceder a tu Drive, puedes aceptarlo (es lo recomendado) o rechazarlo. Si lo rechazas, todo funciona igual, pero si la sesion se corta perderas el progreso, porque la sesion gratuita se borra al desconectarse.

## Qué verás en pantalla, mensaje por mensaje

### La comprobacion de la GPU

Lo primero que aparece es un resumen del entorno. Busca estas dos lineas:

```text
CUDA disponible: True
GPU            : Tesla T4
```

Que ponga True, es decir verdadero, significa que la GPU esta activa y que el entrenamiento ira rapido. Si en su lugar aparece False acompanado de un aviso que empieza por AVISO: no hay GPU, has olvidado el paso 3. Vuelve a Entorno de ejecucion, Cambiar tipo de entorno de ejecucion, elige T4 GPU, guarda y ejecuta otra vez desde la primera celda.

Tambien veras un mensaje tranquilizador sobre el programa base:

```text
torch y torchvision ya estan disponibles: no se tocan (regla de Colab).
```

Eso significa que Colab ya trae las piezas grandes y que no hay que instalar nada pesado. Si en su lugar aparece la linea Instalando por pip (solo lo que falta), tambien es normal: son dos o tres piezas pequenas que Colab no trae de serie y que se instalan solas en menos de un minuto.

### La primera vez: se esta descargando el dataset de 330 MB

La primera vez que se ejecuta, el sistema descarga de internet las 8189 fotografias de flores del conjunto de datos. Ocupa 330 MB y tarda entre uno y tres minutos segun la velocidad de la conexion. Veras que la celda se queda trabajando un rato largo sin imprimir casi nada: es normal, esta bajando archivos. No cierres la pestana ni interrumpas la celda. Las veces siguientes reutiliza la copia ya descargada y no vuelve a bajar nada.

Si tienes Drive conectado, veras ademas un mensaje que dice que no hay copia en Drive todavia y que se descargara: significa que es la primera vez de verdad. Al final del cuaderno, esa misma copia se guardara en tu Drive para no repetir la descarga.

### Las lineas del entrenamiento con la perdida y la exactitud

Durante el entrenamiento aparece una linea por cada vuelta completa sobre los datos. Tienen este aspecto:

```text
[  1/  2] backbone congelado | train loss 4.1023 acc 0.1500 | val loss 3.7801 acc 0.2100 top5 0.4900 | lr 3.000e-04 *mejor
```

Como se lee, trozo a trozo:

- El primer numero entre corchetes es la vuelta actual y el segundo el total. Si pone 1 de 2 estas en la corrida rapida; si pone 1 de 25, en la corrida completa.
- backbone congelado significa que en esa vuelta solo se entrena la parte nueva del modelo. Mas adelante la linea dira backbone entrenando, que es cuando se afina tambien la parte ya preentrenada.
- train loss es la perdida, el castigo por los errores, medido sobre las fotos de aprendizaje. Cuanto mas bajo, mejor.
- train acc es la exactitud sobre esas mismas fotos: la proporcion de aciertos entre 0 y 1. Un 0.1500 equivale a un 15 por ciento.
- val loss y val acc son lo mismo, pero medidos sobre imagenes que el modelo no ha usado para aprender. Estas son las cifras honestas: si la exactitud de aprendizaje sube mucho y la de validacion se queda atras, el modelo esta memorizando en lugar de aprender.
- top5 es la proporcion de veces que la especie correcta aparece entre las cinco propuestas.
- lr es la velocidad de aprendizaje. Veras que empieza baja, sube durante las primeras vueltas y despues baja sola poco a poco.
- La marca *mejor aparece en las vueltas que mejoran el resultado de validacion. El modelo que se guarda al final es el de la ultima vuelta marcada asi, no el de la ultima vuelta a secas.

Al terminar el entrenamiento aparece un bloque de cierre con el tiempo total y la mejor vuelta:

```text
Listo en 842.3 s (14.0 min) | epocas ejecutadas: 22/25
Mejor val_acc: 0.9531 (epoca 18)
Checkpoint   : resultados/best.pt (existe: True)
```

Si el entrenamiento se para antes de la ultima vuelta, veras una linea que empieza por Early stopping. No es un error: significa que el modelo dejo de mejorar y el sistema prefirio no perder el tiempo.

### El resumen final con las metricas

Despues del entrenamiento viene la evaluacion, que mide el modelo sobre las 6149 fotos de prueba que nunca ha visto. Aparece una linea como esta:

```text
[evaluate] split=test imagenes=6149 top1=0.9420 top5=0.9880 macro_f1=0.9380 (61.42s)
```

Los tres numeros importantes son:

- top1: la exactitud top-1, es decir, la proporcion de fotos en las que la primera propuesta es la especie correcta.
- top5: la exactitud top-5, la proporcion de fotos en las que la especie correcta esta entre las cinco propuestas.
- macro_f1: una nota de conjunto que reparte el merito por igual entre las 102 especies, de modo que una especie dificil no quede tapada por las faciles.

Justo despues, el cuaderno imprime un bloque de texto delimitado por unas marcas de tijera, con las cifras principales, el tiempo, el numero de parametros y las confusiones mas frecuentes. Ese bloque esta pensado para copiarse tal cual en el informe: no hay que escribir ningun numero a mano. Se guarda ademas en un archivo llamado resumen_informe.md.

### El enlace de la demo

Al final el cuaderno levanta la demostracion y muestra algo parecido a esto:

```text
Demo lanzada.
  URL local    : 127.0.0.1:7860
```

Dentro de Colab veras ademas la propia interfaz incrustada en la pagina, lista para usarse. Si se activa la opcion de compartir, aparece ademas una linea con una direccion publica temporal de internet: ese es el enlace publico de la demo. Funciona desde el movil o desde otro ordenador, pero caduca en unas 72 horas y cualquiera que lo tenga puede usar la demo mientras siga vivo, asi que no lo dejes abierto con imagenes privadas.

### El ZIP de resultados

La ultima celda crea y descarga el archivo comprimido:

```text
ZIP creado: /content/resultados_flores.zip (312.4 MB)
```

El tamano exacto varia de una corrida a otra. Si la descarga automatica no arranca, abre el panel de archivos de la izquierda, busca resultados_flores.zip, pulsa los tres puntos que hay a su derecha y elige Descargar.

## La corrida rápida primero y la corrida completa después

El cuaderno trae preparada una prueba rapida que se enciende y se apaga con un interruptor llamado MODO_DEBUG, que significa modo de prueba. Viene activado de fabrica y hace dos cosas: entrena solo 2 vueltas sobre un trozo pequeno del conjunto de imagenes, y guarda sus resultados en una carpeta aparte llamada resultados/debug, de modo que nunca pisa la corrida buena.

La prueba rapida tarda unos minutos y responde a una sola pregunta: funciona todo. De una vez comprueba la descarga de las imagenes, la carga del modelo, el entrenamiento, la evaluacion y la demostracion. Si algo tiene que fallar, que falle aqui, donde lo arreglas en minutos.

Para la corrida buena hay que apagar ese interruptor: busca en el cuaderno la linea que dice MODO_DEBUG = True, cambiala por MODO_DEBUG = False y vuelve a ejecutar desde esa celda hasta el final. Entonces si: 25 vueltas completas, entre 15 y 25 minutos en una GPU T4, y las cifras que si valen.

La razon de este orden es sencilla: no tiene sentido gastar media hora de GPU si falta un archivo o la GPU no esta activada. Primero se comprueba que la maquinaria gira; despues se fabrica la pieza buena.

## Cómo usar la demo

### Subir una foto o hacerla con la camara

La demostracion es una pagina con dos formas de darle una imagen:

- Subir una foto: arrastra el archivo de imagen desde tu ordenador hasta el recuadro, o pulsa el recuadro y selecciona el archivo. Despues pulsa el boton Enviar.
- Usarla con la camara: activa la pestana de camara, permite el acceso cuando el navegador lo pida y haz la foto. Tambien hay un boton para pegar una imagen que hayas copiado antes.

En unos segundos veras el resultado: las cinco especies mas probables, cada una con su porcentaje de confianza, y una tabla con las mismas cifras.

### Como interpretar los porcentajes

El porcentaje no significa que sea seguro que la flor es esa. Es la confianza que el modelo pone en cada una de las cinco candidatas. Se lee asi:

- Si la primera propuesta tiene un porcentaje alto y muy por encima de las demas, el modelo lo tiene claro.
- Si las cinco propuestas tienen porcentajes parecidos, el modelo duda: probablemente la foto no sea de buena calidad o la especie se parezca mucho a otras.
- La lista de cinco es la parte mas util. Aunque la primera no sea la correcta, la buena suele estar entre las cinco. En las pruebas del proyecto la especie correcta aparece entre las cinco propuestas en la gran mayoria de las fotos.

### El aviso de confianza baja

Si la mejor propuesta no llega al 25 por ciento de confianza, la demostracion no te obliga a elegir: te avisa de que la confianza es baja y te recuerda que la foto podria no ser ninguna de las 102 especies conocidas. El modelo solo conoce esas 102; ante una flor de otro catalogo, lo honesto es avisar en lugar de inventarse un nombre.

> Nota: un porcentaje bajo no siempre significa que la foto sea mala. En la corrida rapida los porcentajes son bajos por definicion, porque el modelo apenas ha entrenado. Compara siempre lo que ves con el resultado de la corrida completa.

## Cómo recoger los resultados y dónde queda el informe

Al terminar la corrida completa, todo queda en dos sitios:

- En el archivo resultados_flores.zip que se descarga a tu ordenador. Dentro estan el modelo entrenado, que se llama best.pt; las cifras, en metrics.json; el historial de cada vuelta, en history.json; las figuras, que son las curvas de entrenamiento, la matriz de confusion y el ranking de confusiones; la lista de predicciones; y el resumen llamado resumen_informe.md.
- En el propio cuaderno, que imprime ese mismo resumen listo para copiar. Ese bloque es el que sustituye los huecos que hoy dicen pendiente de la corrida en la seccion de resultados del informe del proyecto.

Para montar el informe final: descomprime el archivo descargado, abre el resumen llamado resumen_informe.md, copia el bloque de cifras en la seccion de resultados y adjunta las tres figuras. No hay que calcular nada, porque las cifras salen directamente de la corrida.

Si conectaste tu Drive, tendras ademas una copia en la carpeta IA_flores de tu unidad de Drive, y el conjunto de imagenes quedara guardado para no volver a descargar los 330 MB la proxima vez.

## Problemas frecuentes y cómo resolverlos

| Que ves en pantalla | Que significa | Que haces |
|---|---|---|
| Aparece CUDA disponible: False o un aviso de que no hay GPU | La GPU no esta activada | Entra en Entorno de ejecucion, Cambiar tipo de entorno de ejecucion, elige T4 GPU, guarda y vuelve a ejecutar desde la primera celda |
| El entrenamiento va lentisimo y cada linea tarda muchisimo | Estas entrenando sin GPU, solo con el procesador | Lo mismo que en el caso anterior. La corrida completa en procesador no es razonable, porque tarda horas |
| Colab se desconecta y se pierde el progreso | La sesion gratuita tiene limite de tiempo y se cierra tambien por inactividad | Conecta tu Drive al empezar. El cuaderno copia el modelo y las cifras a Drive justo despues de entrenar, asi que al reconectar solo hay que recuperar esa carpeta |
| Un aviso dice que no se pudieron descargar los pesos preentrenados | La red de Colab no pudo bajar los pesos de ImageNet | El sistema sigue funcionando, pero aprende desde cero y los resultados seran peores. Vuelve a ejecutar la celda y, si se repite, empieza una sesion nueva de Colab |
| La demo responde no reconocida | La confianza no llego al 25 por ciento | Prueba con una foto mas clara, centrada y con la flor grande. Si estas en la corrida rapida, es lo esperado: termina la corrida completa |
| Tarda mucho mas de lo previsto | Suele ser la descarga inicial de 330 MB o una sesion sin GPU | Espera la primera vez, que es la mas lenta. Las siguientes usan la copia local. Si sigue lento, comprueba la GPU |
| Error de memoria, con el texto out of memory | El lote de imagenes es demasiado grande para la memoria disponible | Baja el tamano de lote de 32 a 16 en la configuracion del cuaderno y reinicia el entorno para liberar memoria |
| La ultima celda avisa de que no hay resultados que comprimir | Se ejecuto antes de entrenar | Ejecuta primero la celda de entrenamiento y despues la de empaquetado |
| El enlace publico de la demo no aparece | No se activo la opcion de compartir | Es opcional. La demo funciona dentro de Colab sin enlace; el enlace solo sirve para abrirla desde el movil o mostrarla a otra persona |

## Advertencia importante sobre la corrida rápida

**En modo rapido las respuestas no son fiables.** Con solo 2 vueltas sobre un trozo minusculo del conjunto de imagenes, el modelo apenas ha aprendido nada: puede fallar la especie, dar porcentajes muy bajos o responder no reconocida ante una flor que conoce perfectamente. Eso es lo esperado y no significa que algo este roto.

La corrida rapida sirve para una sola cosa: comprobar que todo funciona. Las cifras que se copian en el informe, las que se ensenan y las que se usan para decidir si el sistema es bueno salen siempre de la corrida completa de 25 epocas.

## Chuleta final

- Basta con una cuenta de Google, un navegador y el archivo IA_flores_colab.zip.
- Activa la GPU antes que nada.
- Ejecuta primero en modo rapido para comprobar que todo funciona.
- Apaga el modo rapido y lanza la corrida completa de 25 epocas.
- Sube una foto a la demo y lee las cinco propuestas con su porcentaje.
- Descarga resultados_flores.zip y copia el resumen en el informe.
- Si algo falla, lee el mensaje literal que aparece en pantalla antes de cambiar cualquier cosa: casi siempre dice exactamente que paso.
