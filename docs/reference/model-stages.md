# Encoders, decoders, and heads

Use [the registry](model-chain.md) to construct stages by name. Encoders may
require downloaded weights, particular bands, and time or location inputs.
Check each constructor before selecting a model.

## Encoders

::: geosave.model.stages.encoder.DINOv3

::: geosave.model.stages.encoder.Prithvi
    options:
      inherited_members:
        - __init__
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.model.stages.encoder.PrithviTL
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.model.stages.encoder.Clay

## Prithvi context

::: geosave.model.stages.encoder.prithvi
    options:
      members:
        - temporal_coords
        - location_coords
        - model_context

## Clay context

::: geosave.model.stages.encoder.clay
    options:
      members:
        - time
        - latlon
        - model_context

## Decoders

::: geosave.model.stages.decoder

## Heads

::: geosave.model.stages.head.ClassificationHead

::: geosave.model.stages.head.DenseHead

::: geosave.model.stages.head.RegressionHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

::: geosave.model.stages.head.SegmentationHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

## Complete models

::: geosave.model.stages.monolith
