# Documentation and test review — 2026-10-13

This review checks the current working tree, including existing uncommitted
changes. It covers documentation and Python tests; notebook contents and
production behavior are outside this cleanup.

## Plan and ownership

1. Snapshot the initial docs/tests and collect/run the default suite with
   durations before changing tests.
2. Review every documentation page against current imports, signatures,
   storage/data flows and source ownership. Fix active instructions and mark
   dated design/audit records as historical where necessary.
3. Review geodata tests against the source contracts. Remove unused or proven
   duplicate work, repair inaccurate test claims, and place cases with their
   source module. Preserve metadata round trips, laziness and numerical edges.
4. Review model, ML, CLI and flow tests by the same criteria. Reduce repeated
   setup/computation without removing distinct behavior checks.
5. Review shared test infrastructure and package-wide contracts, reconcile
   the coverage ledgers with the live file inventory, then verify collection,
   default and relevant slow coverage, Ruff, docs generation/build and diffs.

| Owner | Scope | Detailed coverage |
| --- | --- | --- |
| Documentation agent | `docs/`, `README.md`, navigation/reference generation | [Documentation review](2026-10-13-documentation-review.md) |
| Geodata test agent | `tests/geodata/` | [Geodata test review](2026-10-13-geodata-test-review.md) |
| Model/ML test agent | `tests/model/`, `tests/ml/`, `tests/cli/`, `tests/flows/` | [Model/ML test review](2026-10-13-model-ml-test-review.md) |
| Root | Root/shared tests, example test harness, integration and remaining findings | This record |

Changes are based on traced callers and assertions, rather than symbol counts.
Cross-package dependency/packaging tests remain at the test root because they
own repository-wide contracts. `tests/flows/` mirrors `src/geosave_flows/`.

## Baseline

- Collection: **2,087 items; 2,052 selected, 35 deselected**.
- The runtime-enabled baseline collected **2,088 items** after a concurrent
  regression addition: **2,045 passed, 2 skipped, 4 xfailed, 35 deselected**, and
  **2 packaging-check failures** in 253.42 seconds. Both failures were
  `FileNotFoundError` for `io/readers.py`, moved externally during the run after
  collection had cached its path. They were not failed runtime assertions.
- Restricted Zarr synchronization stalled. The identical local round trip
  passed in **0.27 seconds** with local runtime/socket access, which was used
  for integrated validation. No test was disabled to work around the sandbox.
- Initial snapshots and logs: `/tmp/geosave-docs-tests-audit/`.

## Shared tests and infrastructure

| Initial file | Evidence and disposition |
| --- | --- |
| `tests/conftest.py` | Read all fixtures/helpers and traced injection, explicit calls and indirect registrations. Removed `dw_tif_path` and `catalog_http`, which had no callers. Moved the local STAC endpoint fixture chain into `tests/flows/conftest.py`, its only consumer area; kept shared raw/model/grid/memory-store fixtures. |
| `tests/test_root.py` | Moved to `tests/test_init.py`, matching the package initializer. Combined two fresh interpreters into one while preserving lazy-module, accessor-registration and heavy-import assertions. |
| `tests/test_layering.py` | Retained global import-direction contracts. Moved registry discovery to its model owner and corrected the obsolete dictionary expectation there. Removed the subclass-only segmentation smoke because all six task pairs already run through the Trainer contract test. Combined flow/Prefect import checks into one subprocess. |
| `tests/test_packaging.py` | Reused the existing import-time traversal instead of checking only top-level statements. The smoke initially missed conditional/class imports; it now sees both and still excludes function bodies and typing-only imports. Parse each source once, using the same snapshot for both dependency checks. |
| `tests/examples/test_notebooks.py` | Read the Python harness, fixture paths, external-service markers and dependencies. Retained it as example integration coverage; notebook contents and execution were excluded by the task's repository instructions. |
| `tests/data/` | Retained the real GeoTIFF fixture; removing an unused path fixture does not establish that the data artifact is obsolete. |

Package-wide tests and the example integration harness are intentional source
mapping exceptions. Extra numerical/persistence/data-flow regression files
stay under their owning package when they combine several source modules or
keep a large module's edge coverage readable. Retired module filenames and
mixed-owner suites were moved or split; the detailed test ledgers record each
destination and preserved assertion.

`TODO.txt` closes the torch-free coverage, inaccurate crop edge case and
packaging scanner findings. Existing product/typing concerns remain deferred;
the previous type-check count is now explicitly historical.

Repository clone instructions, notebook links and package repository URLs now
match the live `git@github.com:weedkat/geosave.git` origin. The separately
configured published documentation mount remains `/geosave-engine/`.

## Validation and independent review

- Shared/default checks: **11 passed, 5 deselected** in 4.01 seconds.
- Shared slow import checks: **5 passed, 11 deselected** in 19.39 seconds.
- Import-time scanner smoke: conditional/class imports detected; runtime-only
  and typing-only imports excluded.
- Geodata final affected-scope checks: **719 passed**, including plotting
  tests formerly skipped by obsolete hvplot guards. With torch unavailable,
  all **12 original NumPy conversion cases pass** and only **11 tensor cases
  skip**. Details and round-trip/laziness evidence are in the geodata ledger.
- Model/ML focused checks: **363 passed, 2 deselected, 1 expected failure**;
  fixture/registry/flow follow-up: **75 passed**. The retained starter xfail
  documents a real placeholder Dataset rather than hiding a stale assertion.
- Documentation: **80 local blocks across 17 pages exercised**, **110 active
  blocks parsed**, **69 navigation targets exist**, and **1,450/1,450** API
  definitions/fields/methods rendered. External-service and notebook examples
  were excluded; dated records are explicitly historical.
- Repository-wide Ruff: passed.
- Integrated default suite: **2,037 passed, 35 deselected, 4 expected failures**
  in 162.89 seconds, with no skipped tests. The four xfails are the existing
  placeholder starter Dataset and three narrow ODC numerical limitations.
- After the final slow-test strategy and package URL edits: default task/root/
  packaging smoke **17 passed, 7 deselected** in 17.83 seconds.
- Final Zensical build: passed. API freshness and rendered coverage:
  **1,450/1,450**. All **26,143 local file/anchor links across 89 authored
  rendered pages** resolve. The generated theme's 404 page is outside this
  authored-page check; its mount-relative links were accounted for separately.
- Inventory reconciliation: all **85 original documentation pages**, **92
  original geodata Python files**, **56 original model/ML/CLI/flow Python files**
  and **5 root/example Python files** are accounted for. Current inventory:
  **89 Markdown pages, 154 Python test/support files**. Every test package
  directory mirrors source, except the retained example integration harness.
  The external `test_readers.py -> test_openers.py` rename is reconciled.
- Independent bounded review found no actionable regressions. Snapshot/AST
  comparisons confirmed that moved decorators/assertions and consolidated
  parameter cases preserve unique behavior, persistence and laziness checks.
- Full local slow suite after the distributed fix: **11 passed, 2,065
  deselected** in 90.49 seconds, including Prefect before distributed training.
- Final repository-wide Ruff and `git diff --check`: passed.

## Distributed test ordering

The first full local slow run had **9 passed, 2 failed**. Both distributed
metric cases passed alone and failed only after the Prefect flow. Training
completed; inherited Prefect thread shutdown callbacks failed during child
process teardown after `fork`. A temporary diagnostic plugin captured the
exact callback assertion, without modifying production code.

The test now uses Lightning's supported `forkserver` strategy. It preserves
two devices, the collective timeout, datasets and both train/validation metric
assertions. Prefect followed by the two distributed cases passed **3 tests**.
The [model/ML ledger](2026-10-13-model-ml-test-review.md#distributed-slow-suite-follow-up)
records the complete reproduction and dependency-source evidence.

Timing comparisons are diagnostic only: source and tests were edited
concurrently during the baseline. The final run records a source fingerprint
and confirmed **no source changes throughout final verification**.

## Handoff and remaining scope

The review is complete. Existing working-tree edits were preserved. No GeoSave
runtime implementation or public API was changed by this documentation/test
cleanup; test paths and helper ownership changed, and package repository URLs
were corrected without changing dependencies.

External-service integration and notebook execution were excluded. Four
existing default-suite xfails remain documented above. Source/typing concerns
listed in `TODO.txt` remain separate product work; historical BasedPyright
counts are not represented as a current repository-wide typing result.
