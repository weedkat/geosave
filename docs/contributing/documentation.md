# Maintain the documentation

```bash
uv sync
uv run python scripts/api_reference.py
uv run python scripts/opener_docstrings.py --check
uv run python scripts/check_doc_examples.py
uv run zensical build
uv run python scripts/api_reference.py --check --site site
```

The first command installs the development tools. The API generator refreshes
the [public API index](../reference/public-api.md) and its contract pages.
Zensical renders guides and the mkdocstrings reference to `site/`.
The final command verifies that the generated Markdown matches the source
and every inventoried symbol has a rendered heading.

`scripts/opener_docstrings.py` generates IDE-readable reader docstrings from
its opener contracts and typed format options. If its check fails, correct the
generator contract first, then run it without `--check`; regenerating an
outdated contract can restore stale behavior descriptions.

## Automatic API reference

[`scripts/api_reference.py`](https://github.com/weedkat/geosave/blob/main/scripts/api_reference.py)
scans `src/geosave` and `src/geosave_flows` with Griffe, without importing
runtime dependencies. It writes the symbol index and four contract pages;
Zensical renders their signatures and docstrings from source.

| Script option | Checks |
| --- | --- |
| No option | Regenerate the five Markdown files. |
| `--check` | Fail when generated files are stale. |
| `--check --site site` | Also check rendered symbol/member headings after building. |
| `--check --audit` | Report annotation/docstring gaps; audit findings are informational. |

The scan excludes tests, examples, private modules/names, CLI implementations,
workspace templates, selected internal modules and upstream classes. See
`inventory()` for exact rules. Edit source docstrings, then regenerate;
the generator does not verify behavior. Guides remain authored documentation.

Preview with `uv run zensical serve`. GitHub Pages publication uses the
existing docs workflow, which runs freshness, example, build and coverage checks.

## Write for the reader's next step

| Section | Reader's question | Include |
| --- | --- | --- |
| Home | What makes this useful to me? | Concrete benefits and one small example. |
| Basics | What can I run first, and what are these objects? | Short steps, native objects, code followed by output. |
| Advanced | How do I combine operations for a larger task? | Starting input, runnable steps, resulting object or artifact. |
| Concepts | Why does it behave this way? | Data flow, invariants, laziness, shape and metadata rules. |
| API reference | Exactly what can I pass and what comes back? | Signatures, fields, defaults, return values, errors, examples. |
| MLOps | How do I reproduce and release a model? | Data and experiment versions, evaluation, three repositories and release lineage. |
| Examples | What can I build, and what will it produce? | Small complete tasks, previews, prerequisites and project status. |
| Contributing | Where is the library heading? | Direction, current focus, open decisions and contribution steps. |
| Common questions | Where is the answer to a recurring question? | A direct answer and a link to its full guide. |

Introduce optional orchestration and project generation only in the activities
that need them. Start ordinary file and model examples with native Python APIs.
Use the library's existing objects and dependencies; a guide should not invent
an adapter to make a proposed workflow appear supported.
Use direct assignments such as `scene = gs.open_raster(...)` in walkthroughs.
Keep reader context managers and cleanup boilerplate out of the lesson unless
file lifetime is the topic.

Concise does not mean unexplained. Say why the operation is needed, introduce
a new object when it enters the workflow, and explain how to read its output.
Keep the context needed to adapt the example; move exhaustive options to reference.

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

Place each result directly below its code: a printed value, native object
display, plot or explorer screenshot. Capture plots from the actual `.gs.plot`
result and explorers from a running browser. Start tutorials from small real
files; put synthetic data construction in an explicit advanced example.
MLOps recommends a reproducible workflow; users choose their own project layout.

## Keep API inputs and outputs accurate

Public APIs need concise Google-style docstrings:

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
the architecture or open-decisions page. Record current limitations in [open decisions](open-decisions.md).

## Protect the first successful workflow

`python scripts/check_doc_examples.py` executes the plain Python fences in the
landing page, first vegetation map, zone comparison, raster/value/vector/I/O
basics, cropping, large-data processing, chip reading, catalog publication,
model construction, input preparation and related concepts in a temporary
directory; `PAGES` in the script is the list. It checks lazy arrays, saved pixel
values and grids, tensor shapes, and release prediction round trips. It also checks catalog
publication/loading and metadata-decoding examples; a build alone does not
exercise those calls. Titled Python fences are
explicit adaptation snippets requiring user data; they are not executed.

The Lightning workspace uses shell commands, so validate its one-epoch fit
separately. The Python-fence checker does not execute that training run.

The release guide's offline fences are in `PAGES`. They save and reload a chain
and processor locally, then compare predictions, grids, dtype, nodata and class
metadata. Its Hub upload fence has a title and requires explicit credentials;
the offline checker does not publish anything.

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

The [newcomer review](documentation-review.md) records layout criticism,
changes and remaining gaps.
