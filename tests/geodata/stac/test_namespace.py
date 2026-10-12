"""The `gs.stac` namespace."""


def test_the_stac_namespace_exposes_item_helpers_and_clients() -> None:
    import geosave as gs

    assert sorted(gs.stac.__all__) == [
        "StacClient",
        "StacProvider",
        "StacQuery",
        "StacSource",
        "StacSourceConfig",
        "StacTable",
        "create_collection",
        "create_item",
            "from_item",
        "get_collections",
    ]
    assert not hasattr(gs.stac, "table")
    assert not hasattr(gs.stac, "StacTableClient")
