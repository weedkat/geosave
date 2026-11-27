"""Save predicted variables as a project-selected data product."""

from geosave.geodata.io.storage import Location


def persist(product, destination: Location):
    """Write the final product to its destination."""
    raise NotImplementedError(
        "Choose an output representation for predicted variables."
    )
