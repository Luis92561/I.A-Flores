"""Infraestructura de tests del proyecto (propietario: orquestador / SA-00).

MOTIVO (peculiaridad de esta maquina, no del proyecto):
  `tempfile.mkdtemp()` y la fixture `tmp_path` de pytest crean directorios con
  modo 0o700. Bajo el sandbox de Windows de este entorno, esos directorios NO se
  pueden listar despues (WinError 5 / PermissionError), asi que cualquier test
  que use `tmp_path` falla con un error de permisos que no tiene nada que ver con
  el codigo probado.

  Solucion: se sustituyen `tmp_path` y `tmp_path_factory` por versiones que crean
  los directorios con el modo POR DEFECTO (que si se pueden listar y recorrer).
  Los tests existentes que piden `tmp_path` siguen funcionando sin cambios.

  NO uses `tempfile.mkdtemp()` en el codigo de `src/` ni en los tests: pide
  `tmp_path` o crea el directorio con `Path.mkdir()` sin modo explicito.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent / ".pytmp"
BASE.mkdir(parents=True, exist_ok=True)


def _nuevo_dir(prefijo: str = "case") -> Path:
    """Crea un directorio temporal con modo por defecto (listable)."""
    destino = BASE / f"{prefijo}-{uuid.uuid4().hex[:10]}"
    destino.mkdir()  # sin mode=: a proposito
    return destino


class _FactoryTemporal:
    """Sustituto compatible (lo justo) de `tmp_path_factory`."""

    def mktemp(self, basename: str = "case", numbered: bool = True) -> Path:
        return _nuevo_dir(basename or "case")

    def getbasetemp(self) -> Path:
        return BASE


@pytest.fixture
def tmp_path_factory() -> _FactoryTemporal:
    return _FactoryTemporal()


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    directorio = tmp_path_factory.mktemp("case")
    try:
        yield directorio
    finally:
        shutil.rmtree(directorio, ignore_errors=True)
