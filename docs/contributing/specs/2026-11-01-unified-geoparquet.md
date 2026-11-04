# One GeoParquet writer

```python
features.gs.to_geoparquet("features.parquet", compression="zstd")
catalog.gs.to_geoparquet("catalog.parquet", collections=[collection])
```

Use one public GeoVector.to_geoparquet method. Its implementation delegates
to io.vector.geoparquet.write, which selects the encoder using live frame
metadata when called. Preserve the stable GeoVector accessor and native GDFs.
Remove to_stac_geoparquet without a compatibility alias. Separate internal
ordinary-vector and STAC encoders remain appropriate.

The presence of attrs['stac-geoparquet'] identifies catalog intent, including
an empty metadata dictionary. Validate that metadata is a mapping containing
a mapping of Collections when supplied, and require the Item fields before
encoding. Malformed/incomplete catalogs raise before output; never fall back
to ordinary vector output. Complete Item-shaped frames without the marker
also raise with guidance to construct catalog metadata explicitly. Features
with incidental id/assets columns remain ordinary vectors. collections= is
valid only for marked catalogs. Sampled catalogs remain transient and cannot
be serialized as unique Item tables.

The reader restores the file's STAC metadata even for selected columns. Only
resolve asset hrefs when the selected frame has the needed Item fields. This
preserves Collection definitions and prevents partial catalog rewrites from
silently losing their encoding intent. Recognizing columns is not validation
of format compliance.

Keep Collection copying/extents, relative asset hrefs, bbox reconstruction,
Item fields and resource/storage semantics. Shared writer options such as
compression and Parquet options reach the STAC encoder. STAC encoding keeps
WKB geometry, required bbox and no serialized pandas index; conflicting
options are rejected explicitly. Supported STAC GeoParquet schema versions
remain those supported by the installed dependency (1.0.0 and 1.1.0).

The alternative separate public method made persistence policy visible but
duplicated the caller's choice of file format. Select internally instead.
No accessor subclass dispatch, new wrappers/dependencies, notebook edits,
or broader raster/ModelSpec refactoring.
