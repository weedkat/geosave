---
search:
  exclude: true
---

# Documentation review

Reviewed on 16 November 2026 against the working source, Git history and
rendered site.

## Reading order and layout

| Problem for a new reader | Change |
| --- | --- |
| Several competing starts and a quickstart full of tensor plumbing. | Start with a vegetation map; prepare real image/label chips next. |
| Guides explained APIs without a clear result. | Give each guide a task, inputs and visible output. Keep exhaustive contracts in API Reference. |
| Learning, project organization and contributor proposals were mixed together. | Move walkthroughs into Basics and Advanced; separate Concepts, API Reference, MLOps, Examples and Contributing. |
| The homepage did not make the general features clear. | Lead with imagery and its vegetation map, then link features to tasks. |
| Project patterns had little purpose as a separate section. | Replace it with MLOps: data versions, experiment records, evaluation and traceable releases across three repositories. |
| API generation existed but its scope was unclear. | Document the source scanner, commands and exclusions. |
| Older notebook descriptions implied a current runnable course. | Identify notebooks needing migration beside their links. |

The history review compared 13 quickstart revisions. The 12 October version
(`d30395bc`) had the clearest image/label progression; the 17 October homepage
(`1bc89218`) was a useful concise baseline. Current examples use current public
workflows, rather than copying obsolete APIs from those revisions.

## Unsupported material flagged and removed

“Unsupported” below means the documentation presented a scenario without a
linked issue, reproduced failure or verified project example. It does not mean
every underlying technical statement was false.

| Flag | Material | Action |
| --- | --- | --- |
| Unverified troubleshooting | Generic missing-method, import, file, band, network/authentication, memory and blank-map symptoms presented as known GeoSave problems. | Delete the troubleshooting page and its navigation/cross-links. |
| Unverified troubleshooting | Hypothetical model-shape, cross-entropy and suspicious-validation failures. | Remove the symptom/fix recipes; retain actual input and evaluation contracts in their owning pages. |
| Obsolete example | Synthetic imagery/labels and custom tensor, training and prediction glue presented as the introductory path. | Replace with real aligned files and supported chipping; use the supplied Lightning entry point for workspace training. |
| Test presented as a feature | Copying a red band through a merger as a stand-in prediction. | Replace with overlapping image/label samples; link actual raster prediction. |
| Reader overhead | Repeated reader context managers and cleanup blocks interrupted the task. | Use direct lazy xarray results in walkthroughs. |
| Invented data preparation | Hand-built NumPy/Dask arrays and arbitrary pixel slices in file/value/large-data lessons. | Use bundled imagery and purposeful geographic crops. |
| Stale cross-links | FAQ/activity pages promised a complete raster-training tutorial that the current guide does not supply. | Correct the descriptions and link recipe status. |

Confirmed implementation limits stay in
[open decisions](open-decisions.md) and the relevant reference pages.
Guides should add troubleshooting only when it has evidence, not speculate
about failures.

## Remaining work

The notebook course and several workspace recipes still need migration.
The generated API contract pages are long and can duplicate search results;
reference lookup should be checked separately from the introductory path.

The current examples passed 18 offline Python checks and a one-epoch Lightning
workspace run. Desktop/mobile views, both themes and feature search were checked;
all inventoried API headings rendered. The theme's generated 404 page
still has a broken `#__skip` link; authored page links passed.

See [documentation maintenance](documentation.md) for runnable-example,
reference-coverage and rendered-layout checks.
