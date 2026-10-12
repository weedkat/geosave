"""The `gs.stac` namespace."""


def test_the_stac_namespace_exposes_item_helpers_and_clients() -> None:
    import geosave as gs

    assert sorted(gs.stac.__all__) == [
        "StacClient",
        "StacProvider",
        "StacQuery",
        "StacSource",
        "StacSourceConfig",
        "cog_files",
        "create_collection",
        "create_item",
        "create_items",
        "get_collections",
        "instant",
    ]
    assert not hasattr(gs.stac, "table")
    assert not hasattr(gs.stac, "StacTableClient")
