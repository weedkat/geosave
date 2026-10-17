"""The `gs.stac` namespace."""


def test_the_stac_namespace_exposes_item_helpers_and_clients() -> None:
    import geosave as gs

    assert sorted(gs.stac.__all__) == [
        "StacClient",
        "StacProvider",
        "StacQuery",
        "StacSource",
        "StacSourceConfig",
        "create_catalog",
        "create_collection",
        "create_items",
        "read_collections",
    ]
    assert not hasattr(gs.stac, "table")
    assert not hasattr(gs.stac, "StacTableClient")


def test_stac_client_and_query_are_available_at_the_root() -> None:
    import geosave as gs

    assert gs.StacClient is gs.stac.StacClient
    assert gs.StacQuery is gs.stac.StacQuery
