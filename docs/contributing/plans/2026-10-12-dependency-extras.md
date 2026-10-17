# Dependency Cleanup and Extras Implementation Plan

> Historical implementation plan. The extras now declared in `pyproject.toml`
> are documented in [Installation](../../install.md). The dependency list and
> unchecked steps below record the proposed implementation, including packages
> subsequently changed; they are not a current install recipe.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pip install geosave` installs data, STAC, I/O, plotting, `model.spec`
and the CLI; training, Hub releases, Prefect flows, cloud filesystems and MLflow
come from extras; no declared dependency is unused and no imported one is
undeclared.

**Architecture:** `import geosave` already loads no torch, Lightning, HoloViz
or typer, so packages split cleanly at module boundaries. A packaging test reads
`pyproject.toml` and the AST of every module, and checks two things: each
package's module-level third-party imports are declared in core or in that
package's extras, and every declared dependency is imported somewhere or is
listed as runtime-only. Function-level (lazy) imports may come from any extra:
`.gs.to_tensor()` needs `[ml]` and fails with Python's own `ModuleNotFoundError`
when it is missing.

**Tech Stack:** Hatchling, uv (dependency groups, self-referencing extras),
`tomllib`, `ast`, `importlib.metadata.packages_distributions`.

**Spec:** this conversation's agreed design (2026-10-12): core = data +
plotting; extras `[ml]`, `[hub]`, `[flows]`, `[cloud]`, `[all]`.

## Global Constraints

- Python `>=3.12`; packaging stays Hatchling; one distribution `geosave`
  holding `src/geosave` and `src/geosave_flows`.
- `import geosave` must not load torch, lightning, holoviews, panel, typer or
  prefect (already true; keep it true).
- `geosave` never imports `geosave_flows` (existing layering test).
- Run formatters only on files this plan changes.
- Use existing floors where a dependency is already declared; new declarations
  use the currently locked `major.minor` as their floor.

## Review Focus

1. **Plain `uv sync` without `--all-extras`** must keep the full dev
   environment. Expected: the `dev` group includes `geosave[all]`, so torch and
   prefect stay installed. Tested in Task 2, Step 6.
2. **Core-only install, plotting.** Expected: `ds.gs.plot()` works with no
   extras. Smoke-tested in Task 3.
3. **Core-only install, training import.** Expected: `import geosave.ml`
   raises `ModuleNotFoundError` naming `lightning` or `torch`, not an obscure
   error deep in GeoSave. Smoke-tested in Task 3.
4. **`MLFLOW_TRACKING_URI` set without MLflow installed.** `geosave.ml.cli`
   adds an `MLFlowLogger`, and Lightning raises `ModuleNotFoundError: mlflow`.
   Expected: the install docs name the `[mlflow]` extra. Covered in Task 4.
5. **`s3://` or `gs://` paths without `[cloud]`.** fsspec raises its own
   "install s3fs" `ImportError`. Expected: the docs name `[cloud]`. Covered in
   Task 4.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `tests/test_packaging.py` (create) | Declared dependencies match imports per package. |
| `pyproject.toml` (modify) | Core dependencies, extras, dev group. |
| `uv.lock` (regenerate) | Lock for the new dependency sets. |
| `docs/install.md` (modify) | Extras table and install commands. |
| `README.md` (modify) | Install section names `[all]` for notebooks. |
| `AGENTS.md` (modify) | `uv sync` note in Commands. |

### Task 1: Packaging guard test

**Files:**
- Create: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `pyproject.toml` `[project]` `dependencies` and
  `optional-dependencies`.
- Produces: `AREAS: dict[str, set[str]]` (package path prefix → extras whose
  dependencies its module-level imports may use), and
  `RUNTIME_ONLY: set[str]` (normalized distribution names declared for their
  runtime role). Task 2 edits `pyproject.toml` until these tests pass.

- [ ] **Step 1: Write the failing test**

```python
"""Declared dependencies match what each package imports."""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
SOURCES = [
    path
    for path in (ROOT / "src").rglob("*.py")
    if "templates" not in path.relative_to(ROOT / "src").parts
]

# Module-level imports of each package resolve from core plus these extras.
# The longest matching prefix wins.
AREAS: dict[str, set[str]] = {
    "geosave": set(),
    "geosave/model": {"ml", "hub"},
    "geosave/model/spec": set(),
    "geosave/ml": {"ml"},
    "geosave_flows": {"flows"},
}
# Installed for a runtime role, never imported by name.
RUNTIME_ONLY = {
    "jupyter-bokeh",  # panel widgets in notebooks
    "netcdf4",  # xarray's default netCDF engine
    "tensorboard",  # Lightning's TensorBoardLogger, the CLI default
    "mlflow",  # Lightning's MLFlowLogger, added when MLFLOW_TRACKING_URI is set
    "s3fs",  # fsspec s3:// protocol
    "gcsfs",  # fsspec gs:// protocol
}
FIRST_PARTY = {"geosave", "geosave_flows", "__future__"}
DISTRIBUTIONS = packages_distributions()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    return _normalize(re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0])


def _declared(extras: set[str]) -> set[str]:
    names = {_requirement_name(item) for item in PROJECT["dependencies"]}
    for extra in extras:
        names |= {
            _requirement_name(item)
            for item in PROJECT["optional-dependencies"][extra]
        }
    return names - {"geosave"}


def _third_party(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        modules = [node.module]
    else:
        return []
    tops = [module.split(".")[0] for module in modules]
    return [
        top for top in tops if top not in FIRST_PARTY and top not in sys.stdlib_module_names
    ]


def _module_level(tree: ast.Module) -> list[str]:
    """Imports a module runs on import; TYPE_CHECKING blocks never run."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            continue
        found += _third_party(node)
    return found


def _area(path: Path) -> str:
    package = path.relative_to(ROOT / "src").as_posix()
    return max((prefix for prefix in AREAS if package.startswith(prefix)), key=len)


def _distributions(module: str) -> set[str]:
    return {_normalize(name) for name in DISTRIBUTIONS.get(module, [module])}


def test_module_level_imports_are_declared_for_their_package():
    missing = {}
    for path in SOURCES:
        area = _area(path)
        allowed = _declared(AREAS[area])
        for module in _module_level(ast.parse(path.read_text())):
            if not _distributions(module) & allowed:
                missing.setdefault(module, set()).add(area)
    assert missing == {}


def test_every_declared_dependency_is_imported_or_runtime_only():
    imported = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.stmt):
                for module in _third_party(node):
                    imported |= _distributions(module)
    declared = _declared(set(PROJECT["optional-dependencies"]))
    assert declared - imported - RUNTIME_ONLY == set()


def test_all_extra_installs_every_other_extra():
    extras = set(PROJECT["optional-dependencies"]) - {"all"}
    (requirement,) = PROJECT["optional-dependencies"]["all"]
    assert requirement.startswith("geosave[")
    assert set(requirement[len("geosave[") : -1].split(",")) == extras
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: three FAILs.
- First test: `missing` lists `affine`, `dask`, `holoviews`, `shapely`,
  `urllib3` (geosave), `timm`, `torchmetrics` (model/ml).
- Second: lists `botocore`, `datashader`, `gdown`, `litdata`,
  `openpyxl`, `python-box`, `segmentation-models-pytorch`, `tenacity`, `toml`,
  `torchsummary`, `tqdm`, `watchfiles`.
- Third: `KeyError: 'all'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_packaging.py
git commit -m "test: declared dependencies match imports per package"
```

### Task 2: Core dependencies and extras

**Files:**
- Modify: `pyproject.toml` (`[project] dependencies`,
  `[project.optional-dependencies]`, `[dependency-groups] dev`)
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: `AREAS`, `RUNTIME_ONLY` from Task 1.
- Produces: extras `ml`, `hub`, `flows`, `cloud`, `mlflow`, `all`, which
  Task 4 documents.

- [ ] **Step 1: Replace `dependencies`**

```toml
dependencies = [
    "affine>=2.4",
    "cql2>=0.5.6",
    "dask>=2026.3",
    "fsspec>=2026.3.0",
    "geopandas>=1.0.0",
    "geopy>=2.4.0",
    "holoviews>=1.23",
    "hvplot>=0.12.2",
    "jupyter-bokeh>=4.0",
    "matplotlib>=3.7.5",
    "netcdf4>=1.6.0",
    "numpy>=1.24.4",
    "odc-geo>=0.5.1",
    "odc-stac>=0.5.2",
    "orjson>=3.12.0",
    "pandas>=2.0.3",
    "panel>=1.9.4",
    "planetary-computer>=1.0.0",
    "pyarrow>=17.0.0",
    "pydantic>=2.0.0",
    "pyproj>=3.6.0",
    "pystac>=1.12.0",
    "pystac-client>=0.7.6",
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.3",
    "questionary>=2.1.1",
    "rasterio>=1.3.11",
    "rioxarray>=0.15.0",
    "rustac[arrow]>=0.9.17",
    "s2cloudless>=1.7.3",
    "scipy>=1.11.0",
    "shapely>=2.1",
    "stac-geoparquet>=0.8.2",
    "tiler>=0.6.0",
    "tomlkit>=0.15.1",
    "typer>=0.20.1",
    "typing-extensions>=4.13.0",
    "universal-pathlib>=0.3.10",
    "urllib3>=2.6",
    "xarray>=2026.4.0",
    "zarr>=3.0.0",
]
```

Removed with no import anywhere: `botocore` (comes with `s3fs`),
`datashader`, `gdown`, `litdata` (re-add when remote training data is built),
`openpyxl`, `python-box`, `segmentation-models-pytorch`, `tenacity`, `toml`
(`tomlkit` is the one in use), `torchsummary`, `tqdm`, `watchfiles`.

- [ ] **Step 2: Replace `[project.optional-dependencies]`**

```toml
[project.optional-dependencies]
ml = [
    "huggingface-hub>=1.0.0",
    "kornia>=0.8.2",
    "lightning>=2.3.3",
    "tensorboard>=2.14.0",
    "terratorch>=1.2.7",
    "timm>=1.0",
    "torch>=2.6",
    "torchmetrics>=1.9",
]
hub = ["geosave[ml]", "transformers>=5.17.0,<6"]
flows = ["prefect>=3.8.6"]
cloud = ["gcsfs>=2026.3.0", "s3fs>=2026.3.0"]
mlflow = ["mlflow>=2.17.2"]
all = ["geosave[ml,hub,flows,cloud,mlflow]"]
```

- [ ] **Step 3: Replace the `prefect` line in the `dev` group**

Replace `"prefect>=3.8.6",` with `"geosave[all]",` so that a plain
`uv sync` keeps every extra installed for tests and notebooks.

- [ ] **Step 4: Lock and sync**

Run: `uv lock && uv sync`
Expected: the lock resolves; the sync removes the dropped packages (gdown,
python-box, …) and keeps torch, lightning, transformers, prefect and mlflow.

- [ ] **Step 5: Run the packaging tests**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: 3 passed.

- [ ] **Step 6: Verify plain sync keeps the dev environment (Review Focus 1)**

Run: `uv sync && uv run python -c "import torch, lightning, transformers, prefect, mlflow; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Run the full suite and layering**

Run: `uv run pytest && uv run pytest -m slow tests/test_layering.py`
Expected: all pass (1852 + 3 new tests at time of writing).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: core installs data and plotting; training, hub, flows, cloud, mlflow as extras"
```

### Task 3: Core-only install smoke

**Files:** none changed; scratch venv only.

**Interfaces:**
- Consumes: the wheel built from Task 2's `pyproject.toml`.

- [ ] **Step 1: Build and install core only into a fresh venv**

```bash
S=$(mktemp -d)
uv build --wheel -o "$S/dist"
uv venv "$S/venv" --python 3.12
VIRTUAL_ENV="$S/venv" uv pip install "$S"/dist/geosave-*.whl
```
Expected: the install succeeds, and no `torch` wheel is downloaded.

- [ ] **Step 2: Data and plotting work without extras (Review Focus 2)**

```bash
"$S/venv/bin/python" - <<'EOF'
import sys
import numpy as np
from odc.geo.geobox import GeoBox
import geosave as gs

box = GeoBox.from_bbox((10, 50, 11, 51), "EPSG:4326", shape=(8, 8))
ds = gs.raster({"red": (("y", "x"), np.zeros((8, 8), "float32"))}, box)
print(type(ds.gs.plot()).__name__)
print("torch loaded:", "torch" in sys.modules)
EOF
```
Expected: a HoloViews element name such as `Image`, then `torch loaded: False`.

- [ ] **Step 3: Training import names its missing extra (Review Focus 3)**

```bash
"$S/venv/bin/python" -c "import geosave.ml" 2>&1 | tail -1
"$S/venv/bin/geosave" --help | head -3
```
Expected: `ModuleNotFoundError: No module named 'lightning'` (or `'torch'`).
The `geosave` CLI help prints.

- [ ] **Step 4: Remove the scratch directory**

Run: `rm -rf "$S"`

### Task 4: Install documentation

**Files:**
- Modify: `docs/install.md`
- Modify: `README.md` (Installations section)
- Modify: `AGENTS.md` (Commands block)

- [ ] **Step 1: Add the extras table to `docs/install.md` after the first `pip install geosave` block**

~~~markdown
`pip install geosave` covers opening, inspecting, plotting, STAC search and
loading, writing, model specs, and the `geosave` CLI. Add extras for the rest:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `ml` | torch, Lightning, Kornia, timm, TerraTorch, torchmetrics, TensorBoard | Model stages, training tasks, `.gs.to_tensor()` |
| `hub` | `ml` + Transformers | Saving and loading native model releases |
| `flows` | Prefect | `geosave_flows` and `geosave-flows ingest` |
| `cloud` | s3fs, gcsfs | `s3://` and `gs://` paths |
| `mlflow` | MLflow | The MLflow logger added when `MLFLOW_TRACKING_URI` is set |
| `all` | every extra above | Notebooks and the examples series |

```bash
pip install "geosave[all]"
```
~~~

Then replace the old "Install the Hub extra" text, which shows
`pip install 'geosave[hub]'`, so that it refers to the table.

- [ ] **Step 2: In `README.md`, follow the `uv add geosave` block with**

```markdown
Training, Hub releases, Prefect flows and cloud paths are extras;
`pip install "geosave[all]"` installs everything the examples use.
```

- [ ] **Step 3: In the `AGENTS.md` Commands block, add**

```bash
uv sync                      # dev group pulls geosave[all]
```

- [ ] **Step 4: Build the docs**

Run: `uv run zensical build --clean 2>&1 | grep -iE "warn|error" | grep -v griffe`
Expected: no new warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/install.md README.md AGENTS.md
git commit -m "docs: install extras"
```
