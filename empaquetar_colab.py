"""Empaqueta el proyecto en un ZIP listo para subir a Google Colab.

El notebook (`notebooks/flores_colab.ipynb`) tiene un modo "zip" que pide subir
este archivo y luego busca `src/config.py` en la carpeta extraida.

Uso:
    .venv\\Scripts\\python.exe empaquetar_colab.py

Salida: `IA_flores_colab.zip` en la raiz del proyecto.

Se usa una LISTA BLANCA explicita (no se recorre todo el proyecto) por dos razones:
  1. `data/` (330 MB de imagenes) y `resultados/` no deben viajar en el ZIP.
  2. En este equipo el sandbox deja directorios temporales ilegibles
     (`.venv`, `.piptmp`, `.pytmp`, `pytest-cache-files-*`); recorrerlos falla.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DESTINO = RAIZ / "IA_flores_colab.zip"

# ---------------------------------------------------------------- lista blanca
ARCHIVOS = [
    ".gitignore",
    "CONTRATOS.md",
    "INFORME.md",
    "PLAN_TRABAJO.md",
    "README.md",
    "pytest.ini",
    "requirements.txt",
]
CARPETAS = ["src", "tests", "notebooks", "assets", "docs"]

EXCLUIR_NOMBRES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUIR_SUFIJOS = {".pyc", ".pyo"}


def _incluir(p: Path) -> bool:
    if any(part in EXCLUIR_NOMBRES for part in p.parts):
        return False
    return p.suffix not in EXCLUIR_SUFIJOS


def _recolectar() -> list[Path]:
    encontrados: list[Path] = []
    faltantes: list[str] = []

    for nombre in ARCHIVOS:
        p = RAIZ / nombre
        if p.is_file():
            encontrados.append(p)
        else:
            faltantes.append(nombre)

    for carpeta in CARPETAS:
        base = RAIZ / carpeta
        if not base.is_dir():
            faltantes.append(carpeta + "/")
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and _incluir(p):
                encontrados.append(p)

    if faltantes:
        print("AVISO: no existen todavia (se omiten):", ", ".join(faltantes))
    return encontrados


def main() -> int:
    archivos = _recolectar()
    if not archivos:
        print("ERROR: no se encontro ningun archivo que empaquetar")
        return 1

    with zipfile.ZipFile(DESTINO, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in archivos:
            z.write(p, p.relative_to(RAIZ).as_posix())

    # Verificacion: leer el ZIP y comprobar la estructura esperada por el notebook.
    with zipfile.ZipFile(DESTINO) as z:
        nombres = z.namelist()
        malos = z.testzip()

    print(f"\nZIP creado: {DESTINO.name}  ({DESTINO.stat().st_size / 1024:.0f} KB)")
    print(f"Archivos incluidos: {len(nombres)}")
    if malos:
        print("ERROR: entrada corrupta:", malos)
        return 1

    requeridos = ["src/config.py", "src/utils.py", "src/data.py", "src/model.py",
                  "notebooks/flores_colab.ipynb", "requirements.txt"]
    print("\nComprobacion de la estructura que espera el notebook:")
    todo_ok = True
    for req in requeridos:
        ok = req in nombres
        todo_ok &= ok
        print(f"  {'OK ' if ok else 'FALTA'}  {req}")

    print("\nContenido:")
    for n in sorted(nombres):
        print("  ", n)

    print("\nRESULTADO:", "estructura correcta" if todo_ok else "FALTAN ARCHIVOS")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
