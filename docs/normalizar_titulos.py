"""Normaliza la ortografia de los TITULOS de las secciones del documento.

Motivo: los encargos fijaban algunas primeras lineas literalmente y sin acentos
("# Guia practica: ...", "# Los cimientos: configuracion y utilidades"), mientras
que el cuerpo del texto de esas mismas secciones si lleva acentos. El resultado
seria un documento con titulos sin tildes y texto con tildes.

Se usa una LISTA EXPLICITA de titulos (y no reglas generales) a proposito: una
regla del tipo "que -> que con tilde" romperia titulos correctos como
"## Lo que vas a hacer" o "## La funcion que hace la prediccion". Solo se aplican
las tres reglas generales que son seguras en cualquier titulo ("Por que",
"Para que sirve", "Que hay dentro") y el resto va enumerado uno a uno.

Seguridad: solo se tocan lineas de encabezado y se ignoran las que estan dentro
de bloques de codigo (varios archivos tienen comentarios que empiezan por "#").
"""
from __future__ import annotations

import pathlib
import re
import sys

CONTENIDO = pathlib.Path(__file__).resolve().parent / "contenido"

# Titulos exactos -> titulo corregido.
EXACTOS: dict[str, str] = {
    # 03 guia de uso
    "# Guia practica: como usarlo sin saber programar": "# Guía práctica: cómo usarlo sin saber programar",
    "## Para que sirve esta guia": "## Para qué sirve esta guía",
    "## Que necesitas antes de empezar": "## Qué necesitas antes de empezar",
    "## Lo que vas a hacer, en cinco lineas": "## Lo que vas a hacer, en cinco líneas",
    "## Las dos corridas: primero la rapida, despues la completa": "## Las dos corridas: primero la rápida, después la completa",
    "## Que veras en pantalla, mensaje por mensaje": "## Qué verás en pantalla, mensaje por mensaje",
    "## La corrida rapida primero y la corrida completa despues": "## La corrida rápida primero y la corrida completa después",
    "## Como usar la demo": "## Cómo usar la demo",
    "## Como recoger los resultados y donde queda el informe": "## Cómo recoger los resultados y dónde queda el informe",
    "## Problemas frecuentes y como resolverlos": "## Problemas frecuentes y cómo resolverlos",
    "## Advertencia importante sobre la corrida rapida": "## Advertencia importante sobre la corrida rápida",
    # 04 cimientos
    "# Los cimientos: configuracion y utilidades": "# Los cimientos: configuración y utilidades",
    # 05 datos
    "# Los datos: como entran las fotos al sistema": "# Los datos: cómo entran las fotos al sistema",
    # 08 evaluacion
    "# Evaluar y predecir: como sabemos si acierta": "# Evaluar y predecir: cómo sabemos si acierta",
    # 09 demo
    "## La funcion que hace la prediccion: predecir": "## La función que hace la predicción: predecir",
    # 10 notebook y pruebas
    "## Que es un cuaderno (notebook) y que es una celda": "## Qué es un cuaderno (notebook) y qué es una celda",
    "## Recorrido del cuaderno, pasó a pasó": "## Recorrido del cuaderno, paso a paso",
    "## El empaquetado: `empaquetar_colab.py`": "## El empaquetado: empaquetar_colab.py",
    "## El archivo `requirements.txt`": "## El archivo requirements.txt",
    # 11 decisiones
    "# Por que se hizo así: las decisiones y los errores del camino": "# Por qué se hizo así: las decisiones y los errores del camino",
    "## 3. Por que se uso una red ya entrenada y no una red construida desde cero": "## 3. Por qué se usó una red ya entrenada y no una red construida desde cero",
    "## 10. Los errores reales que aparecieron y como se resolvieron": "## 10. Los errores reales que aparecieron y cómo se resolvieron",
    # 12 glosario
    "# Diccionario: todos los terminos en lenguaje llano": "# Diccionario: todos los términos en lenguaje llano",
}

# Reglas generales seguras en cualquier titulo.
GENERALES: list[tuple[str, str]] = [
    (r"^#+\s*\d*\.?\s*Por que\b", None),  # se resuelve abajo conservando el prefijo
    (r"\bPara que sirve\b", "Para qué sirve"),
    (r"\bQue hay dentro\b", "Qué hay dentro"),
]


def corregir_titulo(titulo: str) -> str:
    if titulo in EXACTOS:
        return EXACTOS[titulo]
    nuevo = titulo
    # "Por que" al principio del titulo (con o sin numeracion) siempre lleva tilde.
    nuevo = re.sub(r"^(#+\s*(?:\d+\.\s*)?)Por que\b", r"\1Por qué", nuevo)
    nuevo = re.sub(r"\bPara que sirve\b", "Para qué sirve", nuevo)
    nuevo = re.sub(r"\bQue hay dentro\b", "Qué hay dentro", nuevo)
    nuevo = nuevo.replace("`", "")  # las comillas invertidas se verian literales
    nuevo = re.sub(r"\s{2,}", " ", nuevo).strip()
    return nuevo


def main() -> int:
    cambiados = 0
    detalles: list[str] = []
    for ruta in sorted(CONTENIDO.glob("*.md")):
        lineas = ruta.read_text(encoding="utf-8").splitlines(keepends=True)
        dentro_de_codigo = False
        modificado = False
        for i, linea in enumerate(lineas):
            if linea.lstrip().startswith("```"):
                dentro_de_codigo = not dentro_de_codigo
                continue
            if dentro_de_codigo or not re.match(r"^#{1,4} ", linea):
                continue
            salto = "\n" if linea.endswith("\n") else ""
            cuerpo = linea.rstrip("\n")
            nuevo = corregir_titulo(cuerpo)
            if nuevo != cuerpo:
                lineas[i] = nuevo + salto
                modificado = True
                detalles.append(f"  {ruta.name}: {cuerpo}  ->  {nuevo}")
        if modificado:
            ruta.write_text("".join(lineas), encoding="utf-8")
            cambiados += 1

    print(f"archivos modificados: {cambiados}")
    for detalle in detalles:
        print(detalle)

    print("\n--- titulos finales por seccion ---")
    for ruta in sorted(CONTENIDO.glob("*.md")):
        dentro = False
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.lstrip().startswith("```"):
                dentro = not dentro
                continue
            if not dentro and re.match(r"^# ", linea):
                print(f"  {ruta.name}: {linea}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
