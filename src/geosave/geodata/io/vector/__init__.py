"""Vector persistence: a GeoDataFrame to one file, and back.

GeoPandas writes and reads each format. Where a format has a slot for it,
the frame's `attrs` travel with the file, so what is written reads back as it
was; STAC Item tables go through the stac-geoparquet codec.
"""
