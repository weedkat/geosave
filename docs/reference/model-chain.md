# Model chains and registry

A ModelChain is a PyTorch module that passes named values between stages.
See [Build a model](../guides/models.md) for examples before writing custom stages.

## Composition

::: geosave_engine.model.chain

## Construction

::: geosave_engine.model.registry
    options:
      members:
        - build_model
        - list_models
        - register_model

::: geosave_engine.model.factory.BuildSpec
