# Maintain the documentation

```bash
uv sync
uv run python scripts/api_reference.py
uv run python scripts/check_doc_examples.py
uv run zensical build
uv run python scripts/api_reference.py --check --site site
```

The first command installs the development tools. The generator refreshes
the [public API index](../reference/public-api.md) and its contract pages.
Zensical renders guides and the mkdocstrings reference to `site/`.
The final command verifies that the generated Markdown matches the source
and every inventoried symbol has a rendered heading.

Preview locally with `uv run zensical serve`. Building the docs does not
publish them; the existing GitHub Pages workflow owns publication.

## Write for the reader's next step

| Section | Reader's question | Include |
| --- | --- | --- |
| Home | What makes this useful to me? | Concrete benefits and one small example. |
| Basics | What are these objects and how do I start? | Small native objects, visible results, a few new concepts at a time. |
| How-to guides | How do I solve this problem? | Starting input, runnable steps, resulting object or artifact, next step. |
| Concepts | Why does it behave this way? | Data flow, invariants, laziness, shape and metadata rules. |
| API reference | Exactly what can I pass and what comes back? | Signatures, fields, defaults, return values, errors, examples. |
| Common questions | Where is the answer to a recurring question? | A direct answer and a link to its full guide. |

Introduce optional orchestration and project generation only in the activities
that need them. Start ordinary file and model examples with native Python APIs.
Use the library's existing objects and dependencies; a guide should not invent
an adapter to make a proposed workflow appear supported.

## Make examples useful outside GeoSave

Start with a geospatial question: compare vegetation between management zones,
acquire imagery for labelled locations, or map land cover in another area.
Name the expected artifact before introducing the APIs that produce it.

| A useful tutorial provides | Readers give up when |
| --- | --- |
| A real input, small runnable example, and visible result | The first task is constructing abstractions or unexplained synthetic arrays |
| Download size, credentials, dependencies, and version | A hidden account, GPU, or unavailable package blocks the first run |
| A map beside code and an expected output | Long prose gives no evidence of what success looks like |
| Explicit band order, scale, grid, nodata, and split decisions | Code runs but quietly trains on misaligned or incorrectly decoded values |
| A short adaptation table and links at decision points | Users must read the whole manual to change their area or data |
| Provenance, license, and limits of the result | A teaching model or index is presented as validated environmental evidence |

Use practitioner references to choose the question and evaluation method;
explain which parts the starter actually implements. Synthetic examples remain
useful for offline mechanics, but label them clearly and keep the real-data
project easy to find. Keep maps and tables close to the code that creates them.
A source-code tour belongs in contributor documentation, not the first lesson.

## Keep API inputs and outputs accurate

The inventory uses Griffe, already installed with mkdocstrings, to read source
without importing the training stack or contacting services. It includes
explicit exports and public definitions, deduplicates reexports, and documents
GeoSave-owned class members. Native ecosystem classes retain their upstream
reference. Command-line inputs are documented through command guides and
their actual `--help` output.

The generated contract pages use standard mkdocstrings directives rather than
copied signatures. Public APIs need concise Google-style docstrings:

```python
def select_raster(self, raster: xr.Dataset) -> xr.Dataset:
    """Select and validate the raster variables declared by this requirement.

    Args:
        raster: Dataset whose stored variables and metadata will be checked.

    Returns:
        Dataset with selected variables in the declared order.

    Raises:
        KeyError: A declared variable is absent.
        ValueError: A selected variable fails the requirement.
    """
```

Describe units, array axes, accepted values, mutation, and eager execution
where they affect a caller's decision. Forwarded `**options` should point to
the native backend or the corresponding typed options. Pydantic model fields
are keyword inputs; TypedDict fields describe dictionary inputs or outputs.
For a property, its annotation and short docstring can be enough.

Use `uv run python scripts/api_reference.py --check --audit` to list missing
source annotations, parameter descriptions, schema fields, and return sections.
A rendered heading proves coverage, not the correctness of prose. Use
`Attributes:` for Pydantic and TypedDict fields so Griffe reads their real
annotations, and `Args:` for normal function and constructor parameters.

## Validate examples before settling the explanation

Run a small example through the actual public API before finalizing a guide.
Check the resulting native object, shapes, pixel values, grid, missing values,
and dtype where relevant. Check laziness before computation and persistence
with a round trip. Keep offline examples self-contained; label examples that
require data, credentials, or a network download.

Do not treat a successful site build as proof that a Python example runs.
Exercise a short CPU training run for training changes and compare predictions
before and after a checkpoint or release round trip. Avoid large downloads
when cached weights or a small offline model can verify the contract.

After a public API change, refresh the inventory, update the activity example
that calls it, and rerun the relevant smoke check. The docs workflow builds on
Python source changes and checks both reference freshness and rendered coverage.

## Keep one current explanation

The README is a short entry point; working tutorials belong in guides. Keep
one canonical explanation of each public contract and link to it from concepts
and reference pages. Replace fragile full object representations with the values,
shapes, or metadata that teach the result when editing an example.

Separate runnable offline examples from fragments requiring local classes,
credentials, or remote assets. A migration blocker belongs beside the affected
command, with a working route where available; generated API coverage cannot
catch a broken cross-package workflow.

Review plans and specs against the implementation before presenting them as
current directions. Preserve unresolved decisions and useful rationale, but
retire completed task lists once their remaining constraints are recorded in
the architecture or open-decisions page. The
[2026-11-04 review](review-2026-11-04.md) records the current audit and validation limits.

## Protect the first successful workflow

`python scripts/check_doc_examples.py` executes the plain Python fences in the
first-raster, file-ingestion, chip, and training tutorials in a temporary
directory. It checks lazy arrays, saved pixel values and grids, tensor shapes,
and predictions before/after checkpoint loading. Titled Python fences are
explicit adaptation snippets requiring user data; they are not executed.

Use `--figures` to refresh the static tutorial figures from actual outputs,
and `--network` to also exercise public Earth Search acquisition. Network
checks are opt-in because provider availability is outside this repository.
The bundled sample's provenance is recorded in [sample data](../assets/README.md).

Every introductory guide should name its prerequisites and resulting artifact,
show the result, provide a complete path to it, explain what to change for the
reader's data, and offer one relevant next step. Keep important grid, label,
and value constraints beside the operation they affect. Move exhaustive
options and implementation details to reference and concept pages.

Before publishing navigation or layout changes, inspect desktop and narrow
mobile views in both themes, follow the main entry links, and check local
links and image assets in the rendered site. A site build does not verify
visual layout or example behavior.
