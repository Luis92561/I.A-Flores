"""Construye la documentacion del proyecto en DOCX (y luego se convierte a PDF).

Lee las secciones en Markdown de `docs/contenido/`, las convierte a un documento
Word con estilos e indice, y anade anexos con el CODIGO FUENTE COMPLETO leido de
los propios archivos del proyecto (asi el documento nunca puede quedarse
desactualizado respecto al codigo).

Uso:
    <python-con-python-docx> docs/construir_documentacion.py

Salida: Documentacion_IA_Flores.docx en la raiz del proyecto.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CONTENIDO = RAIZ / "docs" / "contenido"
SALIDA = RAIZ / "Documentacion_IA_Flores.docx"

FUENTE_TITULO = "Reconocimiento de flores por imagen"
FUENTE_SUBTITULO = (
    "Cómo se construyó un sistema de inteligencia artificial que identifica "
    "102 especies de flores"
)

# Datos academicos que aparecen en la portada.
AUTOR = "Luis Alfredo Jaramillo Restrepo"
MATERIA = "Inteligencia Artificial"
PROFESOR = "Andrés Atehortúa Gutiérrez"
TIPO_TRABAJO = "Proyecto final"

# Partes del documento y que secciones Markdown contiene cada una.
PARTES: list[tuple[str, list[str]]] = [
    ("Parte I. Para empezar", ["01_que_es_este_proyecto", "02_conceptos_desde_cero", "03_guia_de_uso"]),
    (
        "Parte II. El codigo explicado",
        [
            "04_cimientos",
            "05_datos",
            "06_modelo",
            "07_entrenamiento",
            "08_evaluacion_y_prediccion",
            "09_demo",
            "10_notebook_y_pruebas",
            "13_extra_flores_propias",
        ],
    ),
    ("Parte III. Las decisiones y los errores", ["11_decisiones"]),
    ("Parte IV. Diccionario", ["12_glosario"]),
]

# Archivos que se incluyen completos en el anexo A: (ruta, para que sirve).
ARCHIVOS_ANEXO: list[tuple[str, str]] = [
    ("src/config.py", "Todos los ajustes del proyecto en un solo sitio."),
    ("src/utils.py", "Utilidades comunes: semillas, dispositivo, guardar JSON."),
    ("src/data.py", "Descarga el dataset, lo transforma y lo reparte en lotes."),
    ("src/model.py", "El cerebro: backbone preentrenado, cabeza nueva y checkpoints."),
    ("src/train.py", "El bucle de entrenamiento."),
    ("src/custom.py", "Entrenamiento con tus propias flores (catalogo aparte)."),
    ("src/evaluate.py", "Mide el acierto sobre el conjunto de prueba y dibuja la matriz de confusión."),
    ("src/predict.py", "Predice la especie de una foto suelta."),
    ("src/app.py", "La demo interactiva de Gradio."),
    ("empaquetar_colab.py", "Crea el archivo comprimido que se sube a Colab."),
    ("tests/conftest.py", "Arreglo de los directorios temporales de las pruebas."),
    ("tests/test_data.py", "Pruebas de la carga de datos."),
    ("tests/test_model.py", "Pruebas del modelo y de los checkpoints."),
    ("tests/test_train.py", "Pruebas del entrenamiento."),
    ("tests/test_eval_predict.py", "Pruebas de la evaluación y de la predicción."),
    ("tests/test_custom.py", "Pruebas del entrenamiento con flores propias."),
    ("tests/test_app.py", "Pruebas de la demo y del uso del catálogo propio."),
    ("tests/test_smoke.py", "Prueba de humo: el pipeline entero de principio a fin."),
    ("pytest.ini", "Configuración de las pruebas automáticas."),
    ("requirements.txt", "Lista de librerías necesarias."),
    ("assets/cat_to_name.json", "Los 102 nombres de especie, para no depender de internet."),
]


# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
def preparar_estilos(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    if "Codigo" not in [s.name for s in doc.styles]:
        codigo = doc.styles.add_style("Codigo", WD_STYLE_TYPE.PARAGRAPH)
        codigo.base_style = doc.styles["Normal"]
        codigo.font.name = "Consolas"
        codigo.font.size = Pt(8.5)
        codigo.paragraph_format.space_after = Pt(0)
        codigo.paragraph_format.space_before = Pt(0)
        codigo.paragraph_format.line_spacing = 1.0
        codigo.paragraph_format.left_indent = Cm(0.4)


def numeracion_de_pagina(doc: Document) -> None:
    """Anade 'Pagina X' centrado en el pie de pagina."""
    for section in doc.sections:
        pie = section.footer.paragraphs[0]
        pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pie.text = "Página "
        run = pie.add_run()
        inicio = OxmlElement("w:fldChar")
        inicio.set(qn("w:fldCharType"), "begin")
        instruccion = OxmlElement("w:instrText")
        instruccion.set(qn("xml:space"), "preserve")
        instruccion.text = "PAGE"
        fin = OxmlElement("w:fldChar")
        fin.set(qn("w:fldCharType"), "end")
        run._r.append(inicio)
        run._r.append(instruccion)
        run._r.append(fin)


# --------------------------------------------------------------------------- #
# Mini-analizador de Markdown (solo los constructos que usan las secciones)
# --------------------------------------------------------------------------- #
def anadir_texto_con_formato(parrafo, texto: str) -> None:
    """Anade texto a un parrafo interpretando **negrita** y `codigo`."""
    piezas = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", texto)
    for pieza in piezas:
        if not pieza:
            continue
        if pieza.startswith("**") and pieza.endswith("**") and len(pieza) > 4:
            parrafo.add_run(pieza[2:-2]).bold = True
        elif pieza.startswith("`") and pieza.endswith("`") and len(pieza) > 2:
            run = parrafo.add_run(pieza[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        else:
            parrafo.add_run(pieza)


def _es_tabla(lineas: list[str], i: int) -> bool:
    return (
        i + 1 < len(lineas)
        and lineas[i].lstrip().startswith("|")
        and re.match(r"^\s*\|[\s:|-]+\|\s*$", lineas[i + 1] or "")
    )


def _celdas(linea: str) -> list[str]:
    return [c.strip() for c in linea.strip().strip("|").split("|")]


def volcar_markdown(doc: Document, texto: str) -> None:
    lineas = texto.splitlines()
    i = 0
    while i < len(lineas):
        linea = lineas[i]

        # --- bloque de codigo ---
        if linea.lstrip().startswith("```"):
            i += 1
            cuerpo: list[str] = []
            while i < len(lineas) and not lineas[i].lstrip().startswith("```"):
                cuerpo.append(lineas[i])
                i += 1
            i += 1  # cierre
            for fila in cuerpo:
                doc.add_paragraph(fila if fila.strip() else " ", style="Codigo")
            doc.add_paragraph()
            continue

        # --- tabla ---
        if _es_tabla(lineas, i):
            cabecera = _celdas(lineas[i])
            i += 2
            filas: list[list[str]] = []
            while i < len(lineas) and lineas[i].lstrip().startswith("|"):
                filas.append(_celdas(lineas[i]))
                i += 1
            tabla = doc.add_table(rows=1, cols=len(cabecera))
            tabla.style = "Table Grid"
            for celda, titulo in zip(tabla.rows[0].cells, cabecera):
                celda.text = ""
                anadir_texto_con_formato(celda.paragraphs[0], titulo)
                for run in celda.paragraphs[0].runs:
                    run.bold = True
            for fila in filas:
                celdas = tabla.add_row().cells
                for celda, valor in zip(celdas, fila):
                    celda.text = ""
                    anadir_texto_con_formato(celda.paragraphs[0], valor)
            doc.add_paragraph()
            continue

        # --- titulos ---
        if linea.startswith("#### "):
            doc.add_heading(linea[5:].strip(), level=4)
        elif linea.startswith("### "):
            doc.add_heading(linea[4:].strip(), level=3)
        elif linea.startswith("## "):
            doc.add_heading(linea[3:].strip(), level=2)
        elif linea.startswith("# "):
            doc.add_heading(linea[2:].strip(), level=1)

        # --- listas ---
        elif re.match(r"^\s*[-*] ", linea):
            parrafo = doc.add_paragraph(style="List Bullet")
            anadir_texto_con_formato(parrafo, re.sub(r"^\s*[-*] ", "", linea))
        elif re.match(r"^\s*\d+[.)] ", linea):
            parrafo = doc.add_paragraph(style="List Number")
            anadir_texto_con_formato(parrafo, re.sub(r"^\s*\d+[.)] ", "", linea))

        # --- cita ---
        elif linea.startswith("> "):
            parrafo = doc.add_paragraph(style="Quote")
            anadir_texto_con_formato(parrafo, linea[2:])

        # --- linea horizontal ---
        elif re.match(r"^\s*---+\s*$", linea):
            doc.add_paragraph()

        # --- parrafo normal ---
        elif linea.strip():
            parrafo = doc.add_paragraph()
            anadir_texto_con_formato(parrafo, linea)
        i += 1


# --------------------------------------------------------------------------- #
# Portada, indice y anexos
# --------------------------------------------------------------------------- #
def portada(doc: Document) -> None:
    doc.add_paragraph()
    titulo = doc.add_heading(FUENTE_TITULO, level=0)
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(FUENTE_SUBTITULO)
    run.italic = True
    run.font.size = Pt(13)

    doc.add_paragraph()
    doc.add_paragraph()
    for etiqueta, valor in (
        ("Autor", AUTOR),
        ("Materia", MATERIA),
        ("Profesor", PROFESOR),
    ):
        parrafo = doc.add_paragraph()
        parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        parrafo.add_run(f"{etiqueta}: ").bold = True
        parrafo.add_run(valor)

    doc.add_paragraph()
    tipo = doc.add_paragraph()
    tipo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = tipo.add_run(TIPO_TRABAJO)
    run.bold = True
    run.font.size = Pt(12)

    fecha = doc.add_paragraph()
    fecha.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fecha.add_run(_dt.date.today().strftime("%d/%m/%Y"))
    doc.add_page_break()


def indice(doc: Document, titulos: dict[str, list[tuple[int, str]]]) -> None:
    doc.add_heading("Índice", level=1)
    for nombre_parte, _ in PARTES:
        doc.add_paragraph(nombre_parte, style="Heading 2")
        for seccion, entradas in titulos.items():
            if seccion not in dict(PARTES)[nombre_parte]:
                continue
            for nivel, titulo in entradas:
                if nivel == 1:
                    parrafo = doc.add_paragraph()
                    parrafo.paragraph_format.left_indent = Cm(0.5)
                    anadir_texto_con_formato(parrafo, titulo)
                elif nivel == 2:
                    parrafo = doc.add_paragraph()
                    parrafo.paragraph_format.left_indent = Cm(1.1)
                    run = parrafo.add_run(titulo)
                    run.font.size = Pt(10)
    parrafo = doc.add_paragraph()
    parrafo.paragraph_format.left_indent = Cm(0.5)
    anadir_texto_con_formato(parrafo, "Anexo A. Codigo fuente completo")
    parrafo = doc.add_paragraph()
    parrafo.paragraph_format.left_indent = Cm(0.5)
    anadir_texto_con_formato(parrafo, "Anexo B. El cuaderno de Colab, celda por celda")
    parrafo = doc.add_paragraph()
    parrafo.paragraph_format.left_indent = Cm(0.5)
    anadir_texto_con_formato(parrafo, "Anexo C. Estructura del proyecto y como regenerarlo")
    doc.add_page_break()


def anexo_codigo(doc: Document) -> None:
    doc.add_page_break()
    doc.add_heading("Anexo A. Código fuente completo", level=1)
    doc.add_paragraph(
        "Los archivos que aparecen a continuación son exactamente los del proyecto, "
        "copiados sin ningún cambio. Se incluyen aquí para que este documento sea "
        "autosuficiente: quien lo lea no necesita abrir el ordenador para ver el código."
    )
    for ruta_relativa, para_que in ARCHIVOS_ANEXO:
        ruta = RAIZ / ruta_relativa
        if not ruta.is_file():
            continue
        texto = ruta.read_text(encoding="utf-8")
        doc.add_heading(ruta_relativa, level=2)
        parrafo = doc.add_paragraph()
        run = parrafo.add_run(f"{para_que} ({len(texto.splitlines())} líneas)")
        run.italic = True
        run.font.size = Pt(10)
        for linea in texto.splitlines():
            doc.add_paragraph(linea if linea.strip() else " ", style="Codigo")
        doc.add_page_break()


def anexo_notebook(doc: Document) -> None:
    doc.add_page_break()
    doc.add_heading("Anexo B. El cuaderno de Colab, celda por celda", level=1)
    doc.add_paragraph(
        "Un cuaderno de Colab es una lista de celdas: unas contienen texto "
        "explicativo y otras el código que se ejecuta. Aquí se reproduce el código "
        "de todas las celdas de código, en el mismo orden en que se ejecutan."
    )
    ruta = RAIZ / "notebooks" / "flores_colab.ipynb"
    if not ruta.is_file():
        doc.add_paragraph("No se encontró el cuaderno.")
        return
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    numero = 0
    for celda in nb.get("cells", []):
        if celda.get("cell_type") != "code":
            continue
        numero += 1
        doc.add_heading(f"Celda de codigo {numero}", level=3)
        for linea in "".join(celda.get("source", [])).splitlines():
            doc.add_paragraph(linea if linea.strip() else " ", style="Codigo")
        doc.add_paragraph()


def anexo_estructura(doc: Document) -> None:
    doc.add_page_break()
    doc.add_heading("Anexo C. Estructura del proyecto y cómo regenerarlo", level=1)
    doc.add_paragraph(
        "Así está organizado el proyecto. Las carpetas data y resultados no viajan "
        "en el paquete final: se crean solas al ejecutarlo (el dataset se descarga "
        "la primera vez)."
    )
    for linea in ARBOL.splitlines():
        doc.add_paragraph(linea, style="Codigo")
    doc.add_heading("Cómo volver a generar el paquete para Colab", level=2)
    doc.add_paragraph(
        "Si se modifica cualquier archivo del proyecto, este comando vuelve a crear "
        "el archivo comprimido que se sube a Colab:"
    )
    for linea in (
        "python empaquetar_colab.py",
        "",
        "# y para volver a generar esta documentación:",
        "python docs/construir_documentacion.py",
    ):
        doc.add_paragraph(linea if linea.strip() else " ", style="Codigo")


ARBOL = """IA flores/
  README.md                 guía rápida para quien llega nuevo
  el contrato interno              interfaces acordadas antes de programar
  PLAN_TRABAJO.md           plan inicial y riesgos previstos
  INFORME.md                informe académico (con huecos para tus resultados)
  requirements.txt          librerías necesarias
  pytest.ini                configuración de las pruebas
  empaquetar_colab.py       crea el ZIP que se sube a Colab
  IA_flores_colab.zip       el paquete que se sube a Colab
  assets/
    cat_to_name.json        los 102 nombres de especie (sin depender de internet)
  docs/
    VERIFICACION.md         auditoría independiente del proyecto
    construir_documentacion.py  genera este documento
    contenido/              las secciones de este documento en Markdown
  notebooks/
    flores_colab.ipynb      el cuaderno que se ejecuta en Colab
  src/
    config.py               todos los ajustes
    utils.py                utilidades comunes
    data.py                 dataset, transformaciones y lotes
    model.py                la red neuronal y los checkpoints
    train.py                entrenamiento
    evaluate.py             métricas y matriz de confusión
    predict.py              predicción de una foto
    app.py                  demo de Gradio
  tests/
    conftest.py             arreglo de los temporales
    test_data.py            pruebas de datos
    test_model.py           pruebas del modelo
    test_train.py           pruebas del entrenamiento
    test_eval_predict.py    pruebas de evaluación y predicción
    test_smoke.py           prueba de humo de todo el pipeline
  data/                     (se crea al ejecutar) el dataset, 330 MB
  resultados/               (se crea al ejecutar) checkpoints, métricas y figuras
"""


# --------------------------------------------------------------------------- #
def main() -> int:
    doc = Document()
    preparar_estilos(doc)
    # Margenes de 2 cm: dan mas ancho de texto para que las lineas de codigo del
    # anexo quepan sin partirse (el valor por defecto de Word es 3,17 cm a los lados).
    for seccion in doc.sections:
        seccion.top_margin = Cm(2.0)
        seccion.bottom_margin = Cm(2.0)
        seccion.left_margin = Cm(2.0)
        seccion.right_margin = Cm(2.0)
    numeracion_de_pagina(doc)
    portada(doc)

    # Recolecta titulos para el indice y comprueba que estan todas las secciones.
    titulos: dict[str, list[tuple[int, str]]] = {}
    faltantes: list[str] = []
    for _, secciones in PARTES:
        for seccion in secciones:
            ruta = CONTENIDO / f"{seccion}.md"
            if not ruta.is_file():
                faltantes.append(seccion)
                continue
            entradas: list[tuple[int, str]] = []
            dentro_de_codigo = False
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                if linea.lstrip().startswith("```"):
                    dentro_de_codigo = not dentro_de_codigo
                    continue
                if dentro_de_codigo:
                    continue
                m = re.match(r"^(#{1,2}) (.+)$", linea)
                if m:
                    entradas.append((len(m.group(1)), m.group(2).strip()))
            titulos[seccion] = entradas
    if faltantes:
        print("AVISO: faltan secciones:", ", ".join(faltantes))

    indice(doc, titulos)

    for nombre_parte, secciones in PARTES:
        doc.add_page_break()
        doc.add_heading(nombre_parte, level=1)
        for seccion in secciones:
            ruta = CONTENIDO / f"{seccion}.md"
            if not ruta.is_file():
                continue
            volcar_markdown(doc, ruta.read_text(encoding="utf-8"))

    anexo_codigo(doc)
    anexo_notebook(doc)
    anexo_estructura(doc)

    doc.save(SALIDA)

    parrafos = len(doc.paragraphs)
    tablas = len(doc.tables)
    palabras = sum(len(p.text.split()) for p in doc.paragraphs)
    print(f"DOCX creado: {SALIDA.name} ({SALIDA.stat().st_size / 1024:.0f} KB)")
    print(f"  párrafos: {parrafos} | tablas: {tablas} | palabras: {palabras}")
    if faltantes:
        print("  secciones ausentes:", ", ".join(faltantes))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
