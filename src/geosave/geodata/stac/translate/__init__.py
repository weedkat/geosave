"""Translate native objects at STAC boundaries.

Item and Asset builders return PySTAC objects; their callers assemble the
hierarchy with PySTAC. Band metadata composes the extension-specific translators.
Collection translation reads discovery metadata as plain raster attributes.
Publication preparation belongs to the I/O writers.
Table conversion delegates the Item/GeoDataFrame representation to Arrow/rustac.

These modules do not load pixels, write files, search catalogs or align rasters.
"""
