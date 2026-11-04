# Hugging Face remote I/O validation

Tested against the authorized `fatmur/test` bucket on 2026-11-03. The Hugging
Face connector also confirmed `hf://buckets/fatmur/test` exists. No installed
Hugging Face skill was found; actual GeoSave round trips used its Python APIs,
s3fs and Rasterio. Credentials were loaded from a temporary permission-0600
file, never written to the repository or the user's AWS configuration.

## Code change

`io.write_cog` and `io.write_geotiff` now explicitly accept `storage_options`
and merge them into the destination UPath before writing. Previously this
keyword leaked into GDAL creation options, while storage saw no credentials.
A live COG upload raised `NoCredentialsError`; two regression tests reproduced
the lost options for both writers before the fix and pass afterward.

## Live results

| Check | Result |
| --- | --- |
| COG with explicit S3 options | Passed: raw/decoded values, nodata, acquisition time, grid and lazy pixels |
| Plain GeoTIFF with explicit S3 options | Passed: upload, grid and pixel values |
| Zarr Dataset and DataTree | Passed: values, dtype, nodata, dates, grid, variable order and groups |
| NetCDF Dataset and DataTree | Passed: the same raster contracts |
| GeoJSON, GeoPackage, GeoParquet | Passed: geometry, CRS, attribute selection and bbox filtering |
| STAC GeoParquet | Passed: publish two dated COGs, persist/reopen Items, check hrefs, read actual remote assets |
| Native GDAL COG through `hf://buckets/…` | Passed using S3 credentials, including a selected window computation |
| Presigned HTTPS with namespace endpoint | Failed: gateway returns `SignatureDoesNotMatch`, including direct HTTP GET |
| Presigned HTTPS with root endpoint, default GDAL | Failed: signed GET works, but GDAL's HEAD request returns 403 |
| Presigned HTTPS with root endpoint and HEAD disabled | Passed: actual COG values and grid |
| Presigned HTTPS with root endpoint and explicit HTTP storage options | Passed: temporary-download path with `encoded=True` |
| Native Hugging Face filesystem: COG, GeoParquet, Zarr | Blocked: cached local Hub OAuth token expired; all three requests returned 401 before writes |

Authenticated operations used the supplied namespace endpoint
`https://s3.hf.co/fatmur` and bucket name `test`, with path addressing and
checksums set to `when_required`.

For presigning, the alternative root-endpoint addressing form worked:

```python
import rasterio
import s3fs
import geosave as gs

# storage_options contains credentials supplied outside the source code.
signer = s3fs.S3FileSystem(
    **{**storage_options, "endpoint_url": "https://s3.hf.co"}
)
url = signer.sign("fatmur/test/path/to/scene.tif", expiration=300)
with rasterio.Env(
    CPL_VSIL_CURL_USE_HEAD="NO",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
):
    with gs.open_raster(url) as raster:
        pixels = raster.red.isel(y=slice(0, 1), x=slice(0, 1)).compute()
```

The root-endpoint signed request returned HTTP 206 and exactly 16 bytes for
`Range: bytes=0-15`. HEAD returned 403. Both endpoint addressing forms are
covered by [Hugging Face's S3 documentation](https://huggingface.co/docs/hub/storage-buckets-s3).
The explicit HEAD setting follows [GDAL's configuration reference](https://gdal.org/en/stable/user/configoptions.html).
GeoSave's global GDAL defaults were not changed to accommodate this endpoint.

## Checks and limits

- Writer/opener regressions: **70 passed, 2 deselected**.
- I/O suite: **300 passed, 14 deselected, 36 warnings**.
- Ruff, opener-docstring freshness and public-API reference freshness passed.
- The main corrected live matrix passed 10 of 11 checks; its remaining check
  was the namespace-endpoint presigned URL failure isolated above. Separate
  tests verified plain TIFF and the two working signed-URL configurations.
- Tiny artifacts validate correctness, not throughput, large multipart uploads,
  interrupted transfers or the amount GDAL downloads for a realistic window.
- Native `HfFileSystem` still needs a refreshed local Hub login, independently
  of the working S3 credentials and connected Hugging Face tool.
- All writes used six UUID-qualified prefixes beneath `geosave-tests/`.
  Cleanup removed 96 created objects in total; the failed native-HF attempt
  created none and was independently checked using S3 credentials.
- Final S3 verification found all six test prefixes empty. The temporary
  credential file was deleted after verification; existing bucket data and
  local AWS configuration were left untouched.
