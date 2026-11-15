"""Raster persistence: an xarray Dataset or tree to one file or store, and back.

Each format is a library writer and reader with the codecs translating
GeoSave's in-memory conventions to the format's own, so what is written reads
back as it was.
"""
