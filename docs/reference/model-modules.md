# Encoders, decoders, and heads

Use [the registry](model-chain.md) to construct modules by name. Encoders may
require downloaded weights, particular bands, and time or location inputs.
Check each constructor before selecting a model.

## Encoders

::: geosave.ml.torch.modules.encoder.DINOv3

::: geosave.ml.torch.modules.encoder.Prithvi
    options:
      inherited_members:
        - __init__
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.ml.torch.modules.encoder.PrithviTL
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.ml.torch.modules.encoder.Clay
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

::: geosave.ml.torch.modules.encoder.ClayTL
    options:
      inherited_members:
        - forward
        - pyramid_channels
        - pyramid_strides
        - input_size

## Time and place as context entries

Encoders pretrained on location or dates take them by name, such as
`centroid` and `day_of_year`, as `.gs.to_tensor(context={...})` returns them.
Other encoders ignore them.

::: geosave.geodata.core.raster.utils.context

## Decoders

::: geosave.ml.torch.modules.decoder

## Heads

::: geosave.ml.torch.modules.head.ClassificationHead

::: geosave.ml.torch.modules.head.DenseHead

::: geosave.ml.torch.modules.head.RegressionHead
    options:
      inherited_members:
        - __init__
        - forward
        - feature_channels

::: geosave.ml.torch.modules.head.SegmentationHead
    options:
      inherited_members:
        - __init__
        - forward
        - feature_channels

## Complete models

::: geosave.ml.torch.modules.monolith
