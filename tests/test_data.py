"""Tests de ``src/data.py`` (propietario SA-01).

Estructura:
  - Tests rapidos y offline (PNG sinteticos + dataset falso): formas de lote,
    rango de etiquetas, determinismo de val/test y estrategia de nombres.
  - ``test_real_flowers102_debug_mode`` (marcado ``slow``): dataset real en modo
    debug. Intenta la descarga como maximo una vez con timeout y hace
    ``pytest.skip`` si no hay red.

Comando de verificacion:
    .venv\\Scripts\\python.exe -m pytest tests\\test_data.py -q
"""

from __future__ import annotations

import json
import threading
import urllib.error
from pathlib import Path

import pytest

from src import data as data_mod
from src.config import Config

IMG_SIZE = 224
NUM_CLASSES = 102
DOWNLOAD_TIMEOUT_S = 600.0

# NOTA: la fixture ``tmp_path`` que usan estos tests la sustituye
# ``tests/conftest.py`` (crea los directorios con permisos por defecto, porque en
# este sandbox los directorios 0o700 de pytest quedan inaccesibles).


# --------------------------------------------------------------------- helpers
def _require_torch():
    """Salta el test si el entorno no tiene torch/torchvision (venv incompleto)."""
    pytest.importorskip("torch", reason="torch no disponible en el entorno")
    pytest.importorskip("torchvision", reason="torchvision no disponible en el entorno")
    try:
        data_mod._transforms()
    except ImportError as exc:  # p. ej. falta scipy y torchvision lo importa de forma eager
        pytest.skip(f"torchvision.transforms no disponible: {exc}")


def _make_pngs(root: Path, n_classes: int = 3, per_class: int = 4, size: int = 32, seed: int = 0) -> None:
    """Crea PNGs temporales repartidos en una carpeta por clase."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    for class_id in range(n_classes):
        folder = root / f"class_{class_id:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            array = rng.integers(0, 256, size=(size, size, 3), dtype="uint8")
            Image.fromarray(array).save(folder / f"img_{k:02d}.png")


def _make_fake_flowers(root: Path, n_classes: int = NUM_CLASSES, per_class: int = 3, size: int = 32) -> None:
    """PNGs planos ``cNNN_kk.png`` (etiqueta 0-based codificada en el nombre)."""
    import numpy as np
    from PIL import Image

    folder = root / "imgs"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for label in range(n_classes):
        for k in range(per_class):
            array = rng.integers(0, 256, size=(size, size, 3), dtype="uint8")
            Image.fromarray(array).save(folder / f"c{label:03d}_{k:02d}.png")


def _tiny_dataset(root: Path, transform):
    """Dataset minimo (sin depender de torchvision.datasets, que exige scipy)."""
    from PIL import Image
    from torch.utils.data import Dataset

    class _TinyImages(Dataset):
        def __init__(self, root: Path, transform):
            self.transform = transform
            self.samples: list[tuple[Path, int]] = []
            for class_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
                for img in sorted(class_dir.glob("*.png")):
                    self.samples.append((img, int(class_dir.name.split("_")[-1])))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            path, label = self.samples[index]
            image = Image.open(path).convert("RGB")
            return self.transform(image), label

    return _TinyImages(root, transform)


class _FakeFlowers102:
    """Imita la parte de ``torchvision.datasets.Flowers102`` que usa src/data.py."""

    def __init__(self, root, split: str = "train", transform=None, target_transform=None, download: bool = False):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.download = download
        self._image_files = sorted((self.root / "imgs").glob("*.png"))
        # etiquetas 0-based, como las entrega torchvision
        self._labels = [int(path.stem.split("_")[0][1:]) for path in self._image_files]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.open(self._image_files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self._labels[index]


def _write_cache(root: Path, prefix: str = "sp") -> Path:
    """Escribe un ``cat_to_name.json`` falso con claves 1-based ('1'..'102')."""
    root = Path(root)  # acepta tambien el ``cfg.data_root`` (str)
    root.mkdir(parents=True, exist_ok=True)
    path = root / data_mod.CAT_TO_NAME_FILENAME
    path.write_text(
        json.dumps({str(i + 1): f"{prefix}{i + 1:03d}" for i in range(NUM_CLASSES)}),
        encoding="utf-8",
    )
    return path


def _cfg(tmp_path: Path, **kwargs) -> Config:
    kwargs.setdefault("data_root", str(tmp_path / "data"))
    kwargs.setdefault("out_dir", str(tmp_path / "out"))
    return Config(**kwargs)


def _raise_no_network(*args, **kwargs):
    raise urllib.error.URLError("sin red (test offline)")


def _hide_dataset_names(monkeypatch):
    """Simula un torchvision que NO expone nombres propios del dataset.

    Necesario para probar de forma determinista la cache, el respaldo del repo, la
    descarga y el fallback: si torchvision esta instalado,
    ``Flowers102.classes`` tiene prioridad y ganaria siempre.
    """

    def _raise_import_error():
        raise ImportError("nombres del dataset no disponibles (test)")

    monkeypatch.setattr(data_mod, "_flowers102_class", _raise_import_error)


def _hide_offline_backups(monkeypatch):
    """Desactiva ``assets/cat_to_name.json`` para poder probar la red y el fallback."""
    monkeypatch.setattr(data_mod, "ASSET_CAT_TO_NAME", Path("no_existe_cat_to_name.json"))


#: Estado de la descarga real: garantiza como maximo UN intento fallido por sesion.
_DOWNLOAD_STATE: dict[str, str | None] = {"failed": None}


def _run_with_timeout(fn, timeout_s: float):
    """Ejecuta ``fn`` en un hilo y falla con TimeoutError si excede ``timeout_s``."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # se re-lanza en el hilo principal
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"la operacion supero {timeout_s:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---------------------------------------------------- transforms (offline)
def test_build_transforms_train_batch_shape_and_label_range(tmp_path):
    """Un lote de train sintetico: (B, 3, 224, 224) normalizado y etiquetas validas."""
    _require_torch()
    import torch
    from torch.utils.data import DataLoader

    _make_pngs(tmp_path / "imgs")
    cfg = _cfg(tmp_path, debug=True, batch_size=6)
    dataset = _tiny_dataset(tmp_path / "imgs", data_mod.build_transforms(cfg, train=True))
    assert len(dataset) == 12

    batch_x, batch_y = next(iter(DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)))
    assert tuple(batch_x.shape) == (6, 3, IMG_SIZE, IMG_SIZE)
    assert batch_x.dtype == torch.float32
    assert bool(torch.isfinite(batch_x).all())
    assert int(batch_y.min()) >= 0
    assert int(batch_y.max()) < 3
    # Normalize(MEAN, STD) aplicado: hay pixeles negativos (no queda en [0, 1])
    assert float(batch_x.min()) < 0.0
    assert float(batch_x.max()) <= 3.0


def test_build_transforms_eval_is_deterministic_and_uses_img_size(tmp_path):
    """Val/test: sin aleatoriedad y respetando ``cfg.img_size`` (no asumir 224)."""
    _require_torch()
    import torch

    _make_pngs(tmp_path / "imgs")
    cfg = _cfg(tmp_path, img_size=96, debug=True)
    dataset = _tiny_dataset(tmp_path / "imgs", data_mod.build_transforms(cfg, train=False))

    first = dataset[0][0]
    second = dataset[0][0]
    assert tuple(first.shape) == (3, 96, 96)
    assert torch.equal(first, second)

    default_cfg = _cfg(tmp_path, debug=True)
    default_dataset = _tiny_dataset(tmp_path / "imgs", data_mod.build_transforms(default_cfg, train=False))
    assert tuple(default_dataset[0][0].shape) == (3, IMG_SIZE, IMG_SIZE)


def test_build_transforms_train_is_random(tmp_path):
    """La augmentacion de train debe variar entre dos lecturas de la misma imagen."""
    _require_torch()
    import torch

    _make_pngs(tmp_path / "imgs", size=64, seed=3)
    cfg = _cfg(tmp_path, debug=True)
    dataset = _tiny_dataset(tmp_path / "imgs", data_mod.build_transforms(cfg, train=True))
    assert not torch.equal(dataset[0][0], dataset[0][0])


# ------------------------------------- build_dataloaders en debug (offline)
def test_build_dataloaders_debug_subset_shapes_and_names(tmp_path, monkeypatch):
    """Criterio de aceptacion: 3 loaders, 102 nombres, lote (B,3,224,224), etiquetas en [0,102)."""
    _require_torch()
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    data_root = tmp_path / "data"
    _make_fake_flowers(data_root, n_classes=NUM_CLASSES, per_class=3)
    _write_cache(data_root)
    monkeypatch.setattr(data_mod, "_flowers102_class", lambda: _FakeFlowers102)

    cfg = _cfg(tmp_path, debug=True, debug_per_class=1, batch_size=8)
    train_dl, val_dl, test_dl, class_names = data_mod.build_dataloaders(cfg)

    assert len(class_names) == NUM_CLASSES
    assert class_names[0] == "sp001"
    assert class_names[NUM_CLASSES - 1] == "sp102"

    assert len(train_dl.dataset) == NUM_CLASSES  # 1 imagen por clase
    assert len(val_dl.dataset) == NUM_CLASSES
    assert len(test_dl.dataset) == NUM_CLASSES
    assert train_dl.batch_size == 8
    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(val_dl.sampler, SequentialSampler)
    assert isinstance(test_dl.sampler, SequentialSampler)

    batch_x, batch_y = next(iter(train_dl))
    assert tuple(batch_x.shape) == (8, 3, IMG_SIZE, IMG_SIZE)
    assert int(batch_y.min()) >= 0
    assert int(batch_y.max()) < NUM_CLASSES

    # Correspondencia etiqueta <-> nombre (pitfall 1-based de cat_to_name.json)
    for index in range(len(train_dl.dataset)):
        _, label = train_dl.dataset[index]
        assert class_names[label] == f"sp{label + 1:03d}"


def test_debug_subset_is_reproducible(tmp_path, monkeypatch):
    """El subconjunto de debug no debe cambiar entre corridas con la misma semilla."""
    _require_torch()

    data_root = tmp_path / "data"
    _make_fake_flowers(data_root, n_classes=NUM_CLASSES, per_class=5)
    _write_cache(data_root)
    monkeypatch.setattr(data_mod, "_flowers102_class", lambda: _FakeFlowers102)

    cfg = _cfg(tmp_path, debug=True, debug_per_class=2)
    first = data_mod.build_dataloaders(cfg)[0].dataset.indices
    second = data_mod.build_dataloaders(cfg)[0].dataset.indices
    assert list(first) == list(second)
    assert len(first) == NUM_CLASSES * 2


def test_debug_subset_handles_classes_with_few_samples(tmp_path):
    """Si una clase tiene menos imagenes que debug_per_class, no debe fallar."""
    _require_torch()
    from torch.utils.data import Subset

    data_root = tmp_path / "data"
    _make_fake_flowers(data_root, n_classes=4, per_class=2)
    dataset = _FakeFlowers102(data_root)
    subset = data_mod._debug_subset(dataset, per_class=10, seed=0)
    assert isinstance(subset, Subset)
    assert len(subset) == 8  # 4 clases x 2 disponibles


# ------------------------------------------------------------ class names
def test_names_from_dataset_prefers_real_names_and_rejects_generic():
    """Los nombres propios del dataset ganan; los genericos se descartan."""

    class _WithRealNames:
        classes = ["pink primrose", "globe thistle"] + [f"especie real {i}" for i in range(100)]

        def __len__(self) -> int:
            return NUM_CLASSES

    class _WithGenericNames:
        classes = [f"class_{i}" for i in range(NUM_CLASSES)]

        def __len__(self) -> int:
            return NUM_CLASSES

    class _WithClassToIdx:
        class_to_idx = {f"flor_{i:03d}": i for i in range(NUM_CLASSES)}

        def __len__(self) -> int:
            return NUM_CLASSES

    class _WithWrongLength:
        classes = ["a", "b"]

        def __len__(self) -> int:
            return 2

    assert data_mod._names_from_dataset(_WithRealNames(), NUM_CLASSES)[0] == "pink primrose"
    assert data_mod._names_from_dataset(_WithGenericNames(), NUM_CLASSES) is None
    assert data_mod._names_from_dataset(_WithWrongLength(), NUM_CLASSES) is None
    assert data_mod._names_from_dataset(_WithClassToIdx(), NUM_CLASSES)[7] == "flor_007"
    assert data_mod._names_from_dataset(None, NUM_CLASSES) is None


def test_get_class_names_prefers_dataset_class_attribute(tmp_path, monkeypatch):
    """Prioridad 1 del contrato: los nombres propios del dataset ganan al cache."""
    _require_torch()
    try:
        flowers = data_mod._flowers102_class()
    except ImportError as exc:
        pytest.skip(f"torchvision.datasets.Flowers102 no disponible: {exc}")
    if not getattr(flowers, "classes", None):
        pytest.skip("esta version de torchvision no expone Flowers102.classes")

    cfg = _cfg(tmp_path)
    _write_cache(cfg.data_root, prefix="cache")  # el cache NO debe tener prioridad
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _raise_no_network)

    assert data_mod.get_class_names(cfg) == list(flowers.classes)


def test_get_class_names_from_cache_uses_1based_keys(tmp_path, monkeypatch):
    """``class_names[i]`` debe salir de ``cat[str(i + 1)]`` (claves 1-based)."""
    _hide_dataset_names(monkeypatch)
    cfg = _cfg(tmp_path)
    _write_cache(cfg.data_root)

    names = data_mod.get_class_names(cfg)
    assert len(names) == NUM_CLASSES
    assert names[0] == "sp001"
    assert names[NUM_CLASSES - 1] == "sp102"


def test_get_class_names_downloads_and_caches(tmp_path, monkeypatch):
    """Sin cache y sin respaldo offline: descarga (simulada) y deja el JSON."""
    _hide_dataset_names(monkeypatch)
    _hide_offline_backups(monkeypatch)
    cfg = _cfg(tmp_path)
    payload = json.dumps({str(i + 1): f"flor_{i + 1:03d}" for i in range(NUM_CLASSES)}).encode("utf-8")
    calls: list[str] = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return payload

    def _fake_urlopen(url, timeout=None):
        calls.append(str(url))
        return _FakeResponse()

    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _fake_urlopen)

    names = data_mod.get_class_names(cfg)
    assert calls == [data_mod.CAT_TO_NAME_URL]
    assert names[0] == "flor_001"
    assert names[NUM_CLASSES - 1] == "flor_102"

    cached = json.loads((Path(cfg.data_root) / data_mod.CAT_TO_NAME_FILENAME).read_text(encoding="utf-8"))
    assert cached["1"] == "flor_001"
    assert cached["102"] == "flor_102"

    # Segunda llamada: usa el cache, sin volver a la red
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _raise_no_network)
    assert data_mod.get_class_names(cfg) == names


def test_get_class_names_without_internet_returns_fallback(tmp_path, monkeypatch):
    """Sin red, sin cache y sin respaldo: ``['especie_001', ..., 'especie_102']``."""
    _hide_dataset_names(monkeypatch)
    _hide_offline_backups(monkeypatch)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _raise_no_network)

    names = data_mod.get_class_names(cfg)
    assert len(names) == NUM_CLASSES
    assert names[0] == "especie_001"
    assert names[NUM_CLASSES - 1] == "especie_102"
    assert not (Path(cfg.data_root) / data_mod.CAT_TO_NAME_FILENAME).exists()


def test_get_class_names_ignores_incomplete_cache(tmp_path, monkeypatch):
    """Un cache parcial o corrupto se descarta (y no rompe la carga)."""
    _hide_dataset_names(monkeypatch)
    _hide_offline_backups(monkeypatch)
    cfg = _cfg(tmp_path)
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / data_mod.CAT_TO_NAME_FILENAME).write_text(json.dumps({"1": "solo una"}), encoding="utf-8")
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _raise_no_network)

    names = data_mod.get_class_names(cfg)
    assert len(names) == NUM_CLASSES
    assert names[0] == "especie_001"

    (data_root / data_mod.CAT_TO_NAME_FILENAME).write_text("{no es json", encoding="utf-8")
    assert len(data_mod.get_class_names(cfg)) == NUM_CLASSES


def test_names_from_mapping_accepts_1based_and_0based_keys():
    """Acepta las dos convenciones: Oxford ("1".."102") y asset del repo ("0".."101")."""
    one_based = {str(i + 1): f"sp{i + 1:03d}" for i in range(NUM_CLASSES)}
    zero_based = {str(i): f"z{i:03d}" for i in range(NUM_CLASSES)}

    assert data_mod._names_from_mapping(one_based, NUM_CLASSES)[0] == "sp001"
    assert data_mod._names_from_mapping(one_based, NUM_CLASSES)[101] == "sp102"
    assert data_mod._names_from_mapping(zero_based, NUM_CLASSES)[0] == "z000"
    assert data_mod._names_from_mapping(zero_based, NUM_CLASSES)[101] == "z101"
    assert data_mod._names_from_mapping({"1": "solo una"}, NUM_CLASSES) is None
    assert data_mod._names_from_mapping({}, NUM_CLASSES) is None


def test_names_from_asset_backup_matches_dataset_order():
    """El respaldo del repo debe coincidir con el orden de etiqueta de torchvision."""
    if not Path(data_mod.ASSET_CAT_TO_NAME).is_file():
        pytest.skip("assets/cat_to_name.json no esta en el repo")

    names = data_mod._names_from_asset(NUM_CLASSES)
    assert names is not None
    assert len(names) == NUM_CLASSES
    assert names[0] == "pink primrose"  # etiqueta 0
    assert names[101] == "blackberry lily"  # etiqueta 101


def test_get_class_names_uses_offline_asset_backup(tmp_path, monkeypatch):
    """Sin nombres del dataset, sin cache y sin red: usa ``assets/cat_to_name.json``."""
    if not Path(data_mod.ASSET_CAT_TO_NAME).is_file():
        pytest.skip("assets/cat_to_name.json no esta en el repo")

    _hide_dataset_names(monkeypatch)
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", _raise_no_network)

    names = data_mod.get_class_names(_cfg(tmp_path))
    assert len(names) == NUM_CLASSES
    assert names[0] == "pink primrose"
    assert names[101] == "blackberry lily"


# --------------------------------------------------- dataset real (lento)
def _build_or_skip(cfg: Config):
    """Construye los loaders reales; si la red falla, ``pytest.skip``.

    La descarga real se intenta como maximo UNA vez por sesion (con timeout).

    Nota de entorno (solo Windows + sandbox): ``tarfile`` crea los directorios del
    ``.tgz`` con ``os.mkdir(..., 0o700)`` y en este sandbox un directorio 0o700
    queda inaccesible, asi que la extraccion falla con PermissionError. Solucion:
    crear antes ``data/flowers-102/jpg`` con permisos normales (en Colab/Linux no
    hace falta nada de esto).
    """
    _require_torch()
    pytest.importorskip("scipy", reason="Flowers102 necesita scipy (imagelabels.mat / setid.mat)")

    if _DOWNLOAD_STATE["failed"]:
        pytest.skip(f"intento previo de descarga fallido: {_DOWNLOAD_STATE['failed']}")
    try:
        return _run_with_timeout(lambda: data_mod.build_dataloaders(cfg), DOWNLOAD_TIMEOUT_S)
    except TimeoutError as exc:
        _DOWNLOAD_STATE["failed"] = str(exc)
        pytest.skip(f"descarga de Flowers-102 no terminada: {exc}")
    except PermissionError as exc:
        _DOWNLOAD_STATE["failed"] = f"{type(exc).__name__}: {exc}"
        pytest.skip(
            f"extraccion bloqueada por permisos ({exc}). En este sandbox, crea antes "
            "data/flowers-102/jpg con mkdir normal y repite (tarfile usa mode 0o700 "
            "para los directorios del .tgz)"
        )
    except Exception as exc:  # sin red, URL caida, zip corrupto...
        _DOWNLOAD_STATE["failed"] = f"{type(exc).__name__}: {exc}"
        pytest.skip(f"dataset real no disponible ({type(exc).__name__}: {exc})")


@pytest.fixture(scope="session")
def real_debug_loaders():
    """Loaders reales en modo debug (data_root por defecto: 'data')."""
    return _build_or_skip(Config(debug=True, debug_per_class=4, batch_size=8))


@pytest.mark.slow
def test_real_flowers102_debug_mode(real_debug_loaders):
    """Dataset real en debug: 408 imagenes por split, lote (B,3,224,224), etiquetas en [0,102)."""
    import torch

    train_dl, val_dl, test_dl, class_names = real_debug_loaders

    assert len(class_names) == NUM_CLASSES
    assert all(isinstance(name, str) and name.strip() for name in class_names)
    # Si salieron nombres reales (dataset/cache/descarga), la etiqueta 0 debe ser
    # la especie 1 oficial: valida el pitfall 1-based de cat_to_name.json.
    if not any(name.startswith("especie_") for name in class_names):
        assert class_names[0] == "pink primrose"
        assert len(set(class_names)) == NUM_CLASSES

    expected = NUM_CLASSES * 4  # debug_per_class=4
    assert len(train_dl.dataset) == expected
    assert len(val_dl.dataset) == expected
    assert len(test_dl.dataset) == expected

    batch_x, batch_y = next(iter(train_dl))
    assert batch_x.ndim == 4
    assert tuple(batch_x.shape[1:]) == (3, IMG_SIZE, IMG_SIZE)
    assert int(batch_y.min()) >= 0
    assert int(batch_y.max()) < NUM_CLASSES
    assert bool(torch.isfinite(batch_x).all())

    # etiqueta -> nombre consistente para todas las muestras del subconjunto
    for index in range(len(train_dl.dataset)):
        _, label = train_dl.dataset[index]
        assert 0 <= int(label) < NUM_CLASSES
        assert class_names[int(label)]


@pytest.mark.slow
def test_real_flowers102_official_sizes():
    """Los splits oficiales deben ser 1020 / 1020 / 6149 (contrato §3)."""
    train_dl, val_dl, test_dl, class_names = _build_or_skip(Config(debug=False, num_workers=0, batch_size=4))

    assert (len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset)) == (1020, 1020, 6149)
    assert len(class_names) == NUM_CLASSES

