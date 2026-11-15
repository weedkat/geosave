# Encoders, decoders, and heads

Use [the registry](model-chain.md) to construct stages by name. Encoders may
require downloaded weights, particular bands, and time or location inputs.
Check each constructor before selecting a model.

## Encoders

::: geosave.ml.components.encoder.DINOv3

::: geosave.ml.components.encoder.Prithvi
    options:
      inherited_members:
        - __init__
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.ml.components.encoder.PrithviTL
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.ml.components.encoder.Clay

## Time and place from the grid entries

Encoders pretrained on location or dates read them from the batch's
`transform`, `crs` and `time`, as `DataTree.gs.to_tensor` writes them, and
derive their own inputs in `forward_pyramid`. Other encoders ignore them.

::: geosave.ml.components.encoder.grid

::: geosave.ml.components.encoder.prithvi.temporal_coords

::: geosave.ml.components.encoder.clay.week_and_hour

## Decoders

::: geosave.ml.components.decoder

## Heads

::: geosave.ml.components.head.ClassificationHead

::: geosave.ml.components.head.DenseHead

::: geosave.ml.components.head.RegressionHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

::: geosave.ml.components.head.SegmentationHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

## Complete models

::: geosave.ml.components.monolith
