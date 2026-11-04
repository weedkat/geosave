"""Generate literal, IDE-readable opener docstrings from typed option references."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
IO = ROOT / "src/geosave/geodata/io"


def options(path: Path, name: str) -> str:
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    doc = ast.get_docstring(node) or ""
    result = doc.split("Attributes:\n", 1)[1].rstrip()
    for field in node.body:
        if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
            name = field.target.id
            annotation = ast.unparse(field.annotation)
            result = result.replace(f"    {name}:", f"    {name} ({annotation}):")
    return result


def generate() -> str:
    source = (IO / "openers.py").read_text()
    storage = (ROOT / "scripts/references/storage-options.txt").read_text().rstrip()
    raster = """Open one raster file, preserving its stored group hierarchy.

Args:
    source: Path, URL, or UPath ending in .tif, .tiff, .jp2, .png, .zarr,
        .nc, .nc4, or .cdf. Signed URL queries are preserved.
    chunks: Dask chunk sizes: {} (default) uses backend-preferred chunks,
        "auto" uses Dask's target sizes, an integer or dimension mapping
        requests sizes, and None uses backend lazy indexing without Dask.
        Coordinate indexes may still be read eagerly.
    mask_and_scale: False (default) retains stored numbers; True decodes
        missing values, scale_factor and add_offset into physical values.
    storage_options: Filesystem configuration listed below. None uses the
        UPath's options and backend credential discovery.
    **options: Format-specific opening options listed below. Omitted keys
        use the installed backend's defaults unless explicitly stated.

Returns:
    Dataset for a banded file or root-only store; DataTree when a Zarr or
    NetCDF store has child groups, even one child. Close the returned object
    to release handles and temporary downloads. Opening does not reproject.

Raises:
    ValueError: Unsupported format or incompatible backend option.
    TypeError: A sequence is supplied instead of one location.
    NotImplementedError: SAFE product directories are not supported.
"""
    vector = """Open one vector file as a native GeoDataFrame.

Args:
    source: Path, URL, or UPath ending in .geojson, .json, .gpkg, .parquet,
        or .geoparquet. Parquet must contain GeoParquet geometry metadata.
    columns: Attribute names to read; None (default) reads all. Geometry is
        retained even if omitted; [] reads only geometry.
    bbox: Optional (minx, miny, maxx, maxy) filter in the file's CRS.
        Selects intersecting features without clipping. GeoParquet requires
        spatial covering metadata; an unsupported filter raises instead of
        silently loading the entire table. Cannot combine with mask.
    storage_options: Filesystem configuration listed below. None uses the
        UPath's options and backend credential discovery.
    **options: Format-specific options listed below. Omitted keys use the
        installed backend's defaults unless explicitly stated.

Returns:
    GeoDataFrame retaining the file's CRS. STAC GeoParquet footer metadata
    is restored in attrs and catalog asset hrefs become absolute.

Raises:
    ValueError: Unsupported format, ambiguous GeoPackage layer, or invalid filter.
    TypeError: Unsupported or duplicated backend option.
"""
    for title, path, name, extra in [
        (
            "GeoTIFF / COG, JPEG2000 and PNG",
            IO / "rasterio.py",
            "RasterioOpenOptions",
            "",
        ),
        ("Zarr", IO / "formats/zarr.py", "ZarrOpenOptions", ""),
        (
            "NetCDF",
            IO / "formats/netcdf.py",
            "NetCDFOpenOptions",
            '    engine: "netcdf4" (default) or "h5netcdf".\n',
        ),
    ]:
        raster += f"\n{title} options:\n{extra}{options(path, name)}\n"
    for title, mod, name, extra in [
        (
            "GeoJSON",
            "geojson",
            "GeoJSONOpenOptions",
            '    engine: "pyogrio" (default) or "fiona".\n',
        ),
        (
            "GeoPackage",
            "geopackage",
            "GeoPackageOpenOptions",
            '    engine: "pyogrio" (default) or "fiona".\n    layer: Layer name; None requires exactly one layer.\n',
        ),
        ("GeoParquet / STAC GeoParquet", "geoparquet", "GeoParquetOpenOptions", ""),
    ]:
        vector += f"\n{title} options:\n{extra}{options(IO / ('formats/' + mod + '.py'), name)}\n"
    docs = {"open_raster": raster, "open_vector": vector}
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.FunctionDef) and node.name in docs:
            doc = node.body[0]
            literal = docs[node.name].rstrip() + "\n\n" + storage + "\n"
            replacement = (
                '    """' + textwrap.indent(literal, "    ").lstrip() + '    """\n'
            )
            lines[doc.lineno - 1 : doc.end_lineno] = [replacement]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = IO / "openers.py"
    result = generate()
    if args.check:
        if target.read_text() != result:
            raise SystemExit(
                "Opener docstrings are stale; run scripts/opener_docstrings.py"
            )
    else:
        target.write_text(result)
