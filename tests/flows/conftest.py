from prefect.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness
import pytest


@pytest.fixture(scope="module")
def prefect_server():
    with (
        temporary_settings({"server.analytics_enabled": False}),
        prefect_test_harness(),
    ):
        yield
