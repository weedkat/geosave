from geosave.geodata.utils import geocode


def test_reverse_geocode_uses_rounded_coordinates(monkeypatch) -> None:
    seen: list[tuple[float, float]] = []

    def request_address(latitude: float, longitude: float) -> dict[str, str]:
        seen.append((latitude, longitude))
        return {"city": "Malang"}

    monkeypatch.setattr(geocode, "_request_nominatim_address", request_address)

    assert geocode.reverse_geocode(-8.051, 112.149) == {"city": "Malang"}
    assert seen == [(-8.05, 112.15)]


def test_search_places_returns_wgs84_extents_by_name(monkeypatch) -> None:
    def request_places(query: str, limit: int):
        assert (query, limit) == ("Malang", 5)
        return (("Malang, East Java, Indonesia", (-8.05, -7.91, 112.57, 112.69)),)

    monkeypatch.setattr(geocode, "_request_nominatim_places", request_places)

    places = geocode.search_places("  Malang ")
    extent = places["Malang, East Java, Indonesia"]

    assert (extent.left, extent.bottom, extent.right, extent.top) == (
        112.57,
        -8.05,
        112.69,
        -7.91,
    )
    assert extent.crs == "EPSG:4326"
    assert geocode.search_places("   ") == {}
