# Lightning boilerplate

Train a small binary classifier on deterministic synthetic features. This example runs locally with the generated configuration.

```bash
python main.py fit --config configs/boilerplate.yaml --trainer.max_epochs 1
```

Edit the sample in `training/datasets.py`, the batching in `training/datamodule.py`, and the model and loss in `training/module.py`. The config selects the model and data classes and sets the run options.
