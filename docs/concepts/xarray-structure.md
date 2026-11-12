# DataArray, Dataset, and DataTree

## DataArray: one array with named axes

```python
import numpy as np
import xarray as xr

red = xr.DataArray(
    [[1, 2, 3], [4, 5, 6]],
    dims=("y", "x"),
    coords={
        "y": [20, 10],
        "x": [100, 110, 120],
        "quality": ("x", [0, 1, 0]),
        "time": np.datetime64("2025-06-01"),
    },
    name="red",
    attrs={"units": "1"},
)
print(red)
```

```text
<xarray.DataArray 'red' (y: 2, x: 3)> Size: 48B
array([[1, 2, 3],
       [4, 5, 6]])
Coordinates:
  * y        (y) int64 16B 20 10
  * x        (x) int64 24B 100 110 120
    quality  (x) int64 24B 0 1 0
    time     datetime64[s] 8B 2025-06-01
Attributes:
    units:    1
```

The **data** is a 2 × 3 array. **Dimensions** name its axes: `y` has two
positions and `x` has three. **Coordinates** describe those positions:
`y` and `x` label the axes, `quality` describes each column, and scalar `time`
describes the whole array. **Attrs** describe the object without adding axes
or labels.

**Indexes** connect coordinate labels to positions. A one-dimensional
coordinate with the same name as its dimension gets an index by default:
`y(y)` and `x(x)` have one; `quality(x)` does not. The stars mark indexed
coordinates in this printout. A dimension can also exist without coordinates.

```python
print(red.indexes)
print(red.sel(y=10).values)   # Select by label.
print(red.isel(y=1).values)   # Select by position.
```

```text
Indexes:
    y        Index([20, 10], dtype='int64', name='y')
    x        Index([100, 110, 120], dtype='int64', name='x')
[4 5 6]
[4 5 6]
```

The data can be a lazy array. Dimensions, labels, and attrs still describe it
without loading all its values.

## Dataset: several variables sharing dimensions and coordinates

```python
scene = xr.Dataset(
    {"red": red, "cloud": ("x", [False, True, False])},
    attrs={"title": "Scene"},
)
print(scene)
```

```text
<xarray.Dataset> Size: 123B
Dimensions:  (y: 2, x: 3)
Coordinates:
  * y        (y) int64 16B 20 10
  * x        (x) int64 24B 100 110 120
    quality  (x) int64 24B 0 1 0
    time     datetime64[s] 8B 2025-06-01
Data variables:
    red      (y, x) int64 48B 1 2 3 4 5 6
    cloud    (x) bool 3B False True False
Attributes:
    title:    Scene
```

A Dataset replaces the single data array with named **data variables**.
Each variable chooses its dimensions: `red` uses `(y, x)`, while `cloud`
uses only `x`. A shared dimension has one size and shared labels within the
Dataset. Indexes let xarray align labelled inputs when constructing or
combining them.

`scene["red"]` is a DataArray with the coordinates that fit its dimensions.
It keeps its own `units` attr; the Dataset's `title` belongs to the Dataset.

## DataTree: Dataset nodes with coordinate inheritance

```python
tree = xr.DataTree.from_dict(
    {
        "/": xr.Dataset(
            coords={"y": [20, 10], "x": [100, 110, 120]},
            attrs={"title": "Survey"},
        ),
        "/scene": xr.Dataset(coords={"time": np.datetime64("2025-06-01")}),
        "/scene/image": xr.Dataset(
            {"red": (("y", "x"), [[1, 2, 3], [4, 5, 6]])}
        ),
    }
)
print(tree)
```

```text
<xarray.DataTree>
Group: /
│   Dimensions:  (y: 2, x: 3)
│   Coordinates:
│     * y        (y) int64 16B 20 10
│     * x        (x) int64 24B 100 110 120
│   Attributes:
│       title:    Survey
└── Group: /scene
    │   Dimensions:  ()
    │   Coordinates:
    │       time     datetime64[s] 8B 2025-06-01
    └── Group: /scene/image
            Dimensions:  (y: 2, x: 3)
            Data variables:
                red      (y, x) int64 48B 1 2 3 4 5 6
```

Every node holds a Dataset, including nodes with children and nodes without
data variables. Paths identify nodes in this tree; they do not make parent
nodes mere folders. Children can use their parents' coordinates. Here, the
root supplies the grid labels, `scene` supplies the date, and `image` supplies
the pixels. A child's indexed dimensions must align with those of its parents.

```python
image = tree["scene/image"]
print(image.to_dataset(inherit="all_coords"))
```

```text
<xarray.Dataset> Size: 96B
Dimensions:  (y: 2, x: 3)
Coordinates:
  * y        (y) int64 16B 20 10
  * x        (x) int64 24B 100 110 120
    time     datetime64[s] 8B 2025-06-01
Data variables:
    red      (y, x) int64 48B 1 2 3 4 5 6
```

| Conversion | Coordinates included |
| --- | --- |
| `image.to_dataset(inherit=False)` | Only those stored on `image` |
| `image.to_dataset()` | Also inherited indexed coordinates: `y`, `x` |
| `image.to_dataset(inherit="all_coords")` | Also inherited non-indexed coordinates: `time` |

Data variables and attrs belong to their own node; they do not inherit.
The result above has no `title`. Converting `tree["scene"]` gives its date
and inherited grid labels, with no data variables: `red` belongs to its child.
Combining children into one Dataset is a separate, explicit operation.

Printouts checked with xarray 2026.4.0.
