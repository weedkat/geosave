# cf-xarray generated API audit

Date: 2026-11-10. Scope: step 1, dependency investigation only. No production migration is authorized by this document.

```python
bounds = ds.cf.get_bounds("time")  # relationship lookup, not shape validation
is_flags = flags.cf.is_flag_variable  # declaration recognition, not full validation
cloud = flags.cf == "cloud"  # named flag expression
```

## Coverage and evidence

Read all **60 generated pages directly linked from the main body of the [API index](https://cf-xarray.readthedocs.io/en/latest/api.html)**: 8 top-level, 5 geometry, 1 grouper, 24 DataArray and 22 Dataset entries. Two geometry pages alias top-level functions, leaving 58 distinct documented paths after alias removal; shared accessor implementations remain separately documented paths. Also read the public members nested in GridMapping, GeometryNames and FlagGrouper class pages.

The contracts below were compared with installed **cf-xarray 0.11.3** source: accessor base and concrete classes; top-level bounds/compression helpers; geometry functions/classes; options; GridMapping; FlagGrouper. Source inspection here means the named entry's body, not every transitive dependency. Generated pages were fetched on this date; `latest` is moving documentation, not a version guarantee.

**Every row: documentation read and entry-point source inspected. No row by itself claims execution or numerical validation.** Executed probes belong in the separate smoke-check record. Mutation and limitation notes distinguish source behavior from generic docstrings.

API pages are not the full public surface: inherited/dynamic xarray wrappers and concrete public members omitted by the index require a separate appendix. “All 60 pages read” must not be reported as every possible wrapped xarray method tested.

## Per-entry contracts

DA = DataArray; DS = Dataset. “New object” does not imply deep pixel copy or metadata independence at every nested level.

| API and documentation source | Inputs → output | Mutation / execution | Limits and adoption implications |
|---|---|---|---|
| [cf_xarray.bounds_to_vertices](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.bounds_to_vertices.html) | Bounds DA + bounds dimension → vertex DA | New value; `apply_ufunc(dask="parallelized")` | Only 1D/2D cell layouts; core-coordinate values read eagerly; Dask core chunk constraints. |
| [cf_xarray.vertices_to_bounds](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.vertices_to_bounds.html) | Vertex DA + output dimension names → bounds DA | New value; NumPy stacking | 1D/2D only; bounds dimension is last in actual output, unlike return-shape prose; no general laziness promise. |
| [cf_xarray.shapely_to_cf](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.shapely_to_cf.html) | Homogeneous geometry sequence/DA → CF Dataset | New variables; same function as geometry alias | First element determines type; Point/Line/Polygon incl multipart; explicit grid_mapping with DA raises DeprecationWarning; no container_name argument despite prose. |
| [cf_xarray.cf_to_shapely](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.cf_to_shapely.html) | CF Dataset + container name → geometry DA | New geometry values; same function as geometry alias | Requires container and referenced variables; supported types point/line/polygon. |
| [cf_xarray.set_options](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.set_options.html) | Criteria/warning keyword options → context manager | Immediately changes module-global options; restores on context exit | Not per-Dataset configuration; missing-reference warning is not validation. |
| [cf_xarray.encode_multi_index_as_compress](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.encode_multi_index_as_compress.html) | Dataset + optional index names → encoded Dataset | New Dataset; builds flattened integer indexes | CF gathered compression, not pixel packing; requires MultiIndex and no preexisting compress declaration. |
| [cf_xarray.decode_compress_to_multi_index](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.decode_compress_to_multi_index.html) | Encoded Dataset + optional index names → MultiIndex Dataset | New Dataset/indexes | Uses compress in attrs and index coordinates; constructs index arrays eagerly; preservation of unrelated coords needs workflow probe. |
| [cf_xarray.GridMapping](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.GridMapping.html) | name, CRS, mapping DA, coordinate names → frozen descriptor | No raster transformation | Descriptor is not a geobox or grid validator; nested DA is still mutable. |
| [cf_xarray.geometry.decode_geometries](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.geometry.decode_geometries.html) | Encoded Dataset → geometry Dataset | Constructs result, but updates input container attrs with encoding | Not pure: source directly mutates container attrs; no container raises; no blanket metadata/laziness guarantee. |
| [cf_xarray.geometry.encode_geometries](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.geometry.encode_geometries.html) | Dataset containing geometries → CF Dataset | Returns same input if no geometry; otherwise rebuilds Dataset | Detects geometry dtype/first object; root attrs not supplied to rebuilt Dataset; relationships currently assume vector-cube patterns. |
| [cf_xarray.geometry.shapely_to_cf](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.geometry.shapely_to_cf.html) | Geometry sequence/DA → CF Dataset | Alias of top-level shapely_to_cf | Same restrictions as top-level entry; not an additional implementation. |
| [cf_xarray.geometry.cf_to_shapely](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.geometry.cf_to_shapely.html) | CF Dataset → geometry DA | Alias of top-level cf_to_shapely | Same restrictions as top-level entry. |
| [cf_xarray.geometry.GeometryNames](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.geometry.GeometryNames.html) | Suffix/mapping names → naming helper | Creates name/attr fields; coords() returns DA mapping | Helper, not public GeoSave representation; geometry_container_attrs returns declaration mapping; coords() and constructor reviewed. |
| [cf_xarray.groupers.FlagGrouper](https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.groupers.FlagGrouper.html) | Flag DA through xarray groupby → encoded categorical groups | factorize constructs grouping codes and named labels | Requires values+meanings; no bitwise grouping; reset() raises NotImplementedError, contrary generic method description. |
| [xarray.DataArray.cf.axes](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.axes.html) | Object → axis-name to coordinate-name lists | Read-only discovery | Only coords; can be multiple matches; axis is not necessarily a dimension. |
| [xarray.DataArray.cf.cell_measures](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.cell_measures.html) | Object → measure-name to variable-name lists | Read-only discovery; may warn | Attrs/encoding relationships; malformed declarations ignored with warning; missing targets filtered. |
| [xarray.DataArray.cf.cf_roles](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.cf_roles.html) | Object → role-name to variable-name lists | Read-only discovery | Attrs/encoding; DA scans coordinates, DS scans all variables; does not validate DSG/mesh structure. |
| [xarray.DataArray.cf.coordinates](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.coordinates.html) | Object → latitude/longitude/vertical/time coordinate-name lists | Read-only discovery | Only coords; no uniqueness, grid regularity or dimension guarantee. |
| [xarray.DataArray.cf.formula_terms](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.formula_terms.html) | DA → term-to-variable mapping | Read-only parse | DA output is flat despite shared return doc; DS filters missing variables; not formula evaluation. |
| [xarray.DataArray.cf.grid_mapping_name](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.grid_mapping_name.html) | Object → one CF projection name | Read-only | Raises on none or multiple distinct mapping names; not necessarily one mapping variable. |
| [xarray.DataArray.cf.grid_mappings](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.grid_mappings.html) | Object → tuple of GridMapping descriptors | Builds pyproj CRS objects | Requires pyproj; attrs/encoding behavior differs from name discovery; DA encoding-only path raises AttributeError in the recorded probe; not a complete stale-carrier inventory. |
| [xarray.DataArray.cf.is_flag_variable](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.is_flag_variable.html) | DA → bool | Read-only presence check | Only checks meanings and values-or-masks keys; does not validate length, uniqueness, dtype or actual pixels. |
| [xarray.DataArray.cf.standard_names](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.standard_names.html) | Object → standard-name to variable-name lists | Read-only | Reads attrs only; DA scans coords, DS scans all variables; names are not checked against CF table. |
| [xarray.DataArray.cf.plot](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.plot.html) | Object → wrapped native plot interface | Plotting creates artists and may compute pixels | Semantic x/y/hue arguments; delegates plotting behavior; not lazy processing. |
| [xarray.DataArray.cf.__getitem__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__getitem__.html) | Semantic/name key → DA | New selection, with associated variables attached | DA rejects list selection; scalar ambiguity raises; DA cannot retain extra-dimensional bounds; relationship discovery is not schema validation. |
| [xarray.DataArray.cf.__repr__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__repr__.html) | Object → text summary | Read-only metadata formatting | Human inspection only; does not certify CF compliance. |
| [xarray.DataArray.cf.add_canonical_attributes](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.add_canonical_attributes.html) | Object + table/override/skip options → same object type | New object; adds attrs and history; may fetch CF table | Does not convert units/pixels; avoids units for datetime; provide table source for deterministic offline use. |
| [xarray.DataArray.cf.differentiate](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.differentiate.html) | Coordinate key + native options → same object type | New numerical result | Native derivative limits apply; optional positive_upward requires valid up/down metadata; no geodesic derivative. |
| [xarray.DataArray.cf.guess_coord_axis](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.guess_coord_axis.html) | Object + verbose → same object type | Shallow copy with coordinate attrs added | Name/datetime heuristics; existing attrs preserved; not reliable source-derived grid metadata. |
| [xarray.DataArray.cf.keys](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.keys.html) | Object → set of semantic keys | Read-only | No uniqueness guarantee; installed DA mapping-name expansion adds characters in the recorded probe; not an authoritative validity checker. |
| [xarray.DataArray.cf.rename_like](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.rename_like.html) | Other native object + skip → renamed same object type | New object and rewritten relationship attrs | Source matches shared semantic keys beyond four coordinates claimed in prose; skips conflicts with warning; rewrites only bounds/coordinates/cell_measures attrs, not all encoding references. |
| [xarray.DataArray.cf.__lt__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__lt__.html) | Flag-meaning string → boolean DA | New pixel expression | Rejects bit masks; orders underlying enumeration values. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.__le__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__le__.html) | Flag-meaning string → boolean DA | New pixel expression | Rejects bit masks; orders underlying enumeration values. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.__eq__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__eq__.html) | Flag-meaning string → boolean DA | New pixel expression | Supports values, masks, combined flags. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.__ne__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__ne__.html) | Flag-meaning string → boolean DA | New pixel expression | Supports values, masks, combined flags. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.__ge__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__ge__.html) | Flag-meaning string → boolean DA | New pixel expression | Rejects bit masks; orders underlying enumeration values. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.__gt__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.__gt__.html) | Flag-meaning string → boolean DA | New pixel expression | Rejects bit masks; orders underlying enumeration values. Generated docs incorrectly describe DA input and bool output. |
| [xarray.DataArray.cf.isin](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.DataArray.cf.isin.html) | Meaning-name sequence → boolean DA | New OR of named flag expressions | Concrete DA override supports masks; unknown names are filtered rather than rejected; empty match → false; not ordinary numeric isin. |
| [xarray.Dataset.cf.axes](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.axes.html) | Object → axis-name to coordinate-name lists | Read-only discovery | Only coords; can be multiple matches; axis is not necessarily a dimension. |
| [xarray.Dataset.cf.bounds](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.bounds.html) | Object → variable/semantic-key to bounds-name lists | Read-only discovery | Reads attrs then encoding; only existing references included; no shape validation. |
| [xarray.Dataset.cf.cell_measures](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.cell_measures.html) | Object → measure-name to variable-name lists | Read-only discovery; may warn | Attrs/encoding relationships; malformed declarations ignored with warning; missing targets filtered. |
| [xarray.Dataset.cf.cf_roles](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.cf_roles.html) | Object → role-name to variable-name lists | Read-only discovery | Attrs/encoding; DA scans coordinates, DS scans all variables; does not validate DSG/mesh structure. |
| [xarray.Dataset.cf.coordinates](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.coordinates.html) | Object → latitude/longitude/vertical/time coordinate-name lists | Read-only discovery | Only coords; no uniqueness, grid regularity or dimension guarantee. |
| [xarray.Dataset.cf.formula_terms](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.formula_terms.html) | DS → vertical-dimension-to-term mappings | Read-only parse | DA output is flat despite shared return doc; DS filters missing variables; not formula evaluation. |
| [xarray.Dataset.cf.grid_mapping_names](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.grid_mapping_names.html) | Object → projection-name to variable-name lists | Read-only | Scans all variables declaring grid_mapping_name, including unreferenced ones; not just active mappings. |
| [xarray.Dataset.cf.grid_mappings](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.grid_mappings.html) | Object → tuple of GridMapping descriptors | Builds pyproj CRS objects | Requires pyproj; attrs/encoding behavior differs from name discovery; DA encoding-only path raises AttributeError in the recorded probe; not a complete stale-carrier inventory. |
| [xarray.Dataset.cf.geometries](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.geometries.html) | Object → geometry-type to container-name lists | Read-only relationship discovery | Reads geometry attrs/encoding; referenced container must be accessible; not a decoder. |
| [xarray.Dataset.cf.standard_names](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.standard_names.html) | Object → standard-name to variable-name lists | Read-only | Reads attrs only; DA scans coords, DS scans all variables; names are not checked against CF table. |
| [xarray.Dataset.cf.__getitem__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.__getitem__.html) | Scalar key → DA; list → DS | New selection, with associated variables attached | DA rejects list selection; scalar ambiguity raises; DA cannot retain extra-dimensional bounds; relationship discovery is not schema validation. |
| [xarray.Dataset.cf.__repr__](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.__repr__.html) | Object → text summary | Read-only metadata formatting | Human inspection only; does not certify CF compliance. |
| [xarray.Dataset.cf.add_bounds](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html) | Coordinate names + dimension options → Dataset | New Dataset with guessed bounds coords and bounds attrs | Estimates midpoint edges; not acquisition coverage; 1D/2D; rejects naming/dimension-size conflicts. |
| [xarray.Dataset.cf.add_canonical_attributes](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_canonical_attributes.html) | Object + table/override/skip options → same object type | New object; adds attrs and history; may fetch CF table | Does not convert units/pixels; avoids units for datetime; provide table source for deterministic offline use. |
| [xarray.Dataset.cf.bounds_to_vertices](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.bounds_to_vertices.html) | Coordinate keys/order → Dataset | New Dataset with vertex variables | Can overwrite vertex names; requires supported bounds; not grid reprojection. |
| [xarray.Dataset.cf.decode_vertical_coords](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.decode_vertical_coords.html) | Output-name mapping → None | Mutates Dataset coords in place | Dispatches installed parametric transform registry; generated prose lists fewer transforms than source registry; no numerical verification here. |
| [xarray.Dataset.cf.differentiate](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.differentiate.html) | Coordinate key + native options → same object type | New numerical result | Native derivative limits apply; optional positive_upward requires valid up/down metadata; no geodesic derivative. |
| [xarray.Dataset.cf.get_bounds](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.get_bounds.html) | Coordinate/semantic key → DA or Dataset | Read-only selection | No results raises; multiple bounds return Dataset despite DA-only docs; attrs/encoding supported. |
| [xarray.Dataset.cf.get_bounds_dim_name](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.get_bounds_dim_name.html) | Coordinate key → dimension name | Read-only checks | Installed code reads bounds in attrs only; asserts one extra dimension of size 2 or 4; not equivalent to get_bounds lookup. |
| [xarray.Dataset.cf.guess_coord_axis](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.guess_coord_axis.html) | Object + verbose → same object type | Shallow copy with coordinate attrs added | Name/datetime heuristics; existing attrs preserved; not reliable source-derived grid metadata. |
| [xarray.Dataset.cf.keys](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.keys.html) | Object → set of semantic keys | Read-only | No uniqueness guarantee; installed DA mapping-name expansion adds characters in the recorded probe; not an authoritative validity checker. |
| [xarray.Dataset.cf.rename_like](https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.rename_like.html) | Other native object + skip → renamed same object type | New object and rewritten relationship attrs | Source matches shared semantic keys beyond four coordinates claimed in prose; skips conflicts with warning; rewrites only bounds/coordinates/cell_measures attrs, not all encoding references. |

## Corrections to earlier conclusions

- `is_flag_variable` is shallow recognition. Keep Legend's coherent declaration validation and palette ownership; use flag expressions for interpretation after the needed checks.
- The index's statement that masks are unsupported is stale: equality/inequality and the concrete DA `isin` implementation handle them. Ordered comparisons do not.
- Dataset `grid_mapping_names` includes unreferenced declared mappings. Do not conflate it with active relationship resolution or assume all mapping APIs have identical scope.
- Bounds helpers are not interchangeable: `get_bounds` can return multiple variables; `get_bounds_dim_name` has stricter shape assumptions and different metadata-location behavior.
- Several methods generate or mutate metadata, including estimated bounds, guessed axes, canonical attributes and in-place vertical decoding. They must not run silently as generic validation.
- Geometry encoding/decoding has separate preservation and eager-execution questions; it cannot be adopted wholesale as a supposedly transparent serialization layer.
- Recognition and named lookup do not guarantee that an ODC operation supports the discovered coordinate topology.

## Remaining gates before adoption

1. Reproduce consequential source/doc discrepancies with bounded probes, recording exact installed versions.
2. Inventory public members absent from the API index and dynamic method translation separately.
3. Map these contracts to GeoSave producers and consumers; retain attrs groups and identify duplicate interpretation explicitly.
4. Establish input/output representation contracts before deleting helpers or changing persistence.

No library-wide integration claim follows from this dependency audit. Formula correctness, every geometry type, every projection and all lazy/chunk arrangements have not been numerically validated.

## Installed public surface beyond the index

Inspected on cf-xarray 0.11.3. These are concrete accessor members absent from the index for one or both object types, plus the dynamic dispatch mechanism. Evidence is the installed `cf_xarray/accessor.py`; upstream [accessor source](https://cf-xarray.readthedocs.io/en/latest/_modules/cf_xarray/accessor.html) is useful for comparison but can change independently.

| API | Input → output | Behavior and limits |
|---|---|---|
| DA/DS `curvefit` | Coordinate keys, callable, native options → Dataset | Resolves coordinates and reduction dimensions before native fitting. Native numerical/execution requirements apply. Source inspected; fitting not executed. |
| DA/DS `describe` | No arguments → None | Deprecated; warns and prints accessor representation. Do not adopt. |
| DA/DS `get_associated_variable_names` | Variable name, error option → relationship-to-names mapping | Reads attrs and encoding. Missing targets are removed, optionally with warnings. `error=False` is not a comprehensive reference-validation policy. |
| DA/DS `drop` | Native-style arguments → exception | Explicitly unimplemented; directs callers to `drop_vars` or `drop_sel`. |
| DA/DS `stack` | New dimensions mapped to semantic dimensions → native result | Resolves dimension names and **mutates the caller-supplied mapping** before delegation. |
| DA `flags` | Flag declarations → Dataset of boolean arrays | Named flag expressions, not validation of an entire categorical raster or palette. |
| DA `grid_mapping_names` | Object → mapping-kind-to-carrier names | Discovery is broader than successful CRS construction; do not assume equivalence to `grid_mappings`. |
| DA `geometries` | Object → geometry-type-to-container names | Reads the array's geometry relation from attrs/encoding and looks up the container coordinate. No decoding or complete schema validation. |
| DS `isin` | Values → exception | Inherited implementation rejects Dataset input. |
| DS `plot` | Plot call → native artists | Contrary to blanket guide wording, semantic Dataset scatter works in 0.11.3. Other plot types were not tested. |
| Dynamic methods and mappings | Native operations with semantic arguments → native results/wrappers | Translates recognized argument names; wraps selected grouped/resampled/rolling/weighted objects. Does not validate or repair CF structure. The native xarray method surface is open-ended, not 60 fixed methods. |

## Executed evidence

Reproduce bounded observations with:

```bash
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python docs/contributing/reviews/cf_xarray_smoke.py
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python docs/contributing/reviews/cf_xarray_api_smoke.py
```

These scripts print observed behavior; they are research probes, not assertions that dependency quirks should become GeoSave contracts.

| Probe | Observed result in installed 0.11.3 |
|---|---|
| Semantic mean, selection, rolling and weighted graph construction | Zero Dask tasks during construction; subsequent computation returns expected small-array results. |
| Semantic `sizes` | Maps Y/X to the corresponding dimension lengths. |
| `stack` with a supplied mapping | Input mapping changes from semantic names to actual dimension names. |
| Missing ancillary reference | Associated-variable discovery omits it and emits a warning. |
| `drop`; Dataset `isin` | NotImplementedError; ValueError respectively. |
| Bounds relationship stored only in encoding | `get_bounds` selects the bounds; `get_bounds_dim_name` raises ValueError. |
| DataArray grid mapping reference in attrs | `grid_mappings` builds one descriptor. |
| Same DataArray reference only in encoding | `grid_mappings` raises AttributeError. |
| DataArray `keys()` with latitude_longitude mapping | Includes individual characters from the mapping name instead of the full mapping name. |
| Dataset semantic scatter (separate Agg-backend probe) | Returns Matplotlib PathCollection for `x="air_temperature"`; blanket unsupported claim is inaccurate. |

The earlier [capability smoke script](cf_xarray_smoke.py) also reproduces shallow flag recognition, combined masks, auxiliary-coordinate discovery versus dimension selection, and bounds variables disrupting GeoSave Dataset spatial discovery. Geometry mutation, vertical formulas, curve fitting and compression are source-reviewed here, not numerically validated.

## Next step: representation contracts

The indexed API reading and omitted-member inventory are complete for this snapshot. Next, trace a small ODC-loaded Dataset through STAC stamping, accessors, transforms and persistence. Explicitly classify raster variables, coordinates and support variables; establish where declarations and relationships live before choosing replacements. Keep attrs groups for coherent declarations and GeoSave-specific checks. Do not treat `.cf` discovery as full CF validation or remove native xarray codecs.
