import os

from geosave import configure_gdal


def test_gdal_configuration_sets_supplied_values(monkeypatch) -> None:
    monkeypatch.delenv("GDAL_HTTP_MAX_RETRY", raising=False)
    monkeypatch.delenv("AWS_NO_SIGN_REQUEST", raising=False)

    configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"
    assert os.environ["GDAL_HTTP_MAX_RETRY"] == "3"


def test_gdal_configuration_keeps_omitted_settings(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    monkeypatch.delenv("AWS_HTTPS", raising=False)
    monkeypatch.delenv("VSI_CACHE", raising=False)
    configure_gdal(aws_https=False, vsi_cache=False)
    assert os.environ["AWS_DEFAULT_REGION"] == "ap-southeast-1"
    assert os.environ["AWS_HTTPS"] == "FALSE"
    assert os.environ["VSI_CACHE"] == "FALSE"
