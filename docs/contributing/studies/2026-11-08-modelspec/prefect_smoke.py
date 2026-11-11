"""Run a genuine local Prefect flow against a temporary server/database."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

import operations
from geosave.model.spec import ModelSpec


@flow(persist_result=False)
def predict_file(path: str, output: str, batch_size: int = 2) -> str:
    spec = ModelSpec.load(Path(__file__).with_name("model_spec.yaml"))
    return spec.run(
        entry="from_path",
        output="saved",
        path=path,
        destination=output,
        batch_size=batch_size,
    )


def main():
    schema = predict_file.parameters.model_dump()
    assert schema["required"] == ["path", "output"]
    assert schema["properties"]["batch_size"]["type"] == "integer"
    with TemporaryDirectory(prefix="geosave-workflow-prefect-") as directory:
        folder = Path(directory)
        source = operations.save(operations.scene(-2.0), str(folder / "source.tif"))
        with prefect_test_harness(server_startup_timeout=20):
            saved = predict_file(path=source, output=str(folder / "prediction.tif"))
        raster = operations.open_raster(saved)
        np.testing.assert_array_equal(
            raster.label.values, np.ones((4, 4), dtype="uint8")
        )
        print("PASS: actual Prefect flow execution and COG value/grid check")
        assert raster.gs.geobox == operations.scene().gs.geobox


if __name__ == "__main__":
    main()
