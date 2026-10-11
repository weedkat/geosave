"""The `gs.stac` namespace."""


def test_the_stac_namespace_exposes_builders_and_clients_only() -> None:
    import geosave_engine as gs

    assert sorted(gs.stac.__all__) == [
        "StacClient",
        "StacProvider",
        "StacQuery",
        "StacSource",
        "StacSourceConfig",
        "create_collection",
        "create_item",
        "create_items",
        "get_collections",
    ]
    assert not hasattr(gs.stac, "table")
    assert not hasattr(gs.stac, "StacTableClient")
