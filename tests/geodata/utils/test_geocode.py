from geosave.geodata.utils import geocode


def test_reverse_geocode_uses_rounded_coordinates(monkeypatch) -> None:
    seen: list[tuple[float, float]] = []

    def request_address(latitude: float, longitude: float) -> dict[str, str]:
        seen.append((latitude, longitude))
        return {"city": "Malang"}

    monkeypatch.setattr(geocode, "_request_nominatim_address", request_address)

    assert geocode.reverse_geocode(-8.051, 112.149) == {"city": "Malang"}
    assert seen == [(-8.05, 112.15)]
