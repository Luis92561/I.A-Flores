# Qué es este proyecto y qué encontrarás en este documento

## En una frase

Es un programa que **mira una fotografía de una flor y dice de qué especie se trata**, eligiendo entre **102 especies distintas**, y que funciona en Google Colab, es decir, en el navegador y sin instalar nada en tu ordenador.

## La idea, contada como se la contarías a alguien que no programa

Imagina que quieres construir un experto en flores. No puedes escribirle una lista de reglas del tipo "si los pétalos son rosas y hay cinco, entonces es una rosa", porque las flores son demasiado variadas y las reglas se vuelven infinitas. Lo que haces es otra cosa: le enseñas **miles de fotos ya etiquetadas** con su nombre correcto (esta es una rosa, esta es un tulipán) y dejas que él solo descubra las diferencias. Eso es, en esencia, la inteligencia artificial de este proyecto.

El programa hace tres cosas:

1. **Aprende** mirando 8189 fotos de flores que ya vienen clasificadas por la Universidad de Oxford.
2. **Se evalúa** con fotos que nunca vio durante el aprendizaje, para saber si de verdad ha aprendido o solo ha memorizado.
3. **Se usa**: le subes una foto cualquiera y te devuelve las cinco especies más probables con su porcentaje de confianza.

## Qué hace bien y qué no hace

Es honesto decirlo desde el principio.

**Sí hace:**

- Distinguir entre las 102 especies del catálogo con una precisión que, en la corrida completa, es alta.
- Decirte "no estoy seguro" cuando la foto no se parece a nada de lo que conoce, en lugar de inventarse una respuesta.
- Funcionar sin GPU (más lento) y con GPU (rápido), tanto en un ordenador normal como en la nube.

**No hace:**

- No reconoce flores que no estén entre esas 102 especies. Si le enseñas una flor de tu jardín que no está en el catálogo, lo más probable es que elija la que más se le parezca y te avise de que la confianza es baja.
- No detecta varias flores en una misma foto: asume que la foto es de una sola flor.
- No es infalible. Con 102 clases y flores que se parecen mucho entre sí (por ejemplo, distintas variedades de petunia), se equivoca.

## Cómo leer este documento

No hace falta leerlo entero ni en orden. Está pensado para que puedas elegir:

| Si eres... | Lee... |
|---|---|
| Alguien que solo quiere entender qué es y cómo se usa | Las partes I y IV, y el diccionario |
| Alguien que quiere entender cómo funciona por dentro | Las partes I y II completas |
| Alguien que va a leer o modificar el código | Todo, y usa el anexo A como referencia |
| Alguien que solo quiere ejecutarlo | La guía práctica (sección 3) |

El documento está dividido en cuatro partes:

- **Parte I. Para empezar.** Qué es el proyecto, los conceptos básicos explicados desde cero (qué es una imagen para un ordenador, qué es un modelo, qué es entrenar) y la guía práctica paso a paso.
- **Parte II. El código explicado.** Un capítulo por archivo. En cada uno se explica qué hace, se muestra el código real y se justifica por qué se escribió así.
- **Parte III. Las decisiones y los errores.** Por qué se eligió cada cosa y qué problemas reales aparecieron por el camino, contados sin adornos, incluidos los errores que cometimos nosotros.
- **Parte IV. Diccionario.** Todos los términos técnicos en lenguaje llano, en orden alfabético.

Después vienen los **anexos**, que contienen el código completo tal cual está en el proyecto, para que nada quede fuera de este documento.

## Ficha técnica del proyecto

| Dato | Valor |
|---|---|
| Qué resuelve | Clasificar una foto entre 102 especies de flores |
| Datos de aprendizaje | Oxford Flowers-102: 8189 fotos (1020 entrenamiento, 1020 validación, 6149 prueba) |
| Técnica principal | Transfer learning sobre una red EfficientNet-B0 preentrenada con ImageNet |
| Lenguaje y librerías | Python con PyTorch, TorchVision, Gradio y Matplotlib |
| Dónde se ejecuta | Google Colab (gratuito) o cualquier ordenador con Python |
| Entrenamiento completo | 25 épocas, lotes de 32 imágenes de 224x224, 15-25 minutos en una GPU T4 |
| Entrenamiento rápido de prueba | 2 épocas sobre 408 imágenes, unos minutos |
| Resultado medido con 4 épocas | 37,8 % de acierto exacto y 67,4 % entre las 5 primeras propuestas, sobre las 6149 fotos oficiales de prueba |
| Cómo se sabe que funciona | 78 pruebas automáticas, una auditoría independiente y varias ejecuciones completas |
| Qué se entrega | Un archivo comprimido con el cuaderno de Colab, el código, las pruebas y los informes |

## Una advertencia útil

En este documento aparecen a veces números pequeños, como el 37,8 % de acierto. **No son el resultado final del proyecto**: son el resultado de una corrida corta de 4 épocas que se hizo para comprobar que todo funcionaba. El resultado bueno se obtiene con la corrida completa de 25 épocas, que tarda entre 15 y 25 minutos en una GPU de Colab y que tienes que lanzar tú. Cuando lo hagas, el propio cuaderno te imprimirá un resumen con tus cifras, listo para copiar en el informe.

Y una última cosa, quizá la más importante de este documento: al final de la parte III contamos **los errores que cometimos** y cómo los encontramos. En un proyecto real eso no es un detalle vergonzoso, es la parte que más enseña.
