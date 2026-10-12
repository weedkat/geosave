# Encoders, decoders, and heads

Use [the registry](model-chain.md) to construct stages by name. Encoders may
require downloaded weights, particular bands, and time or location inputs.
Check each constructor before selecting a model.

## Encoders

::: geosave_engine.model.encoder.DINOv3

::: geosave_engine.model.encoder.Prithvi
    options:
      inherited_members:
        - __init__
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave_engine.model.encoder.PrithviTL
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave_engine.model.encoder.Clay

## Prithvi context

::: geosave_engine.model.encoder.prithvi
    options:
      members:
        - temporal_coords
        - location_coords
        - model_context

## Clay context

::: geosave_engine.model.encoder.clay
    options:
      members:
        - time
        - latlon
        - model_context

## Decoders

::: geosave_engine.model.decoder

## Heads

::: geosave_engine.model.head.ClassificationHead

::: geosave_engine.model.head.DenseHead

::: geosave_engine.model.head.RegressionHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

::: geosave_engine.model.head.SegmentationHead
    options:
      inherited_members:
        - __init__
        - forward
        - forward_logits
        - feature_channels

## Complete models

::: geosave_engine.model.monolith
