# Models, inputs, and releases

Start with a PyTorch model, give it tensors, and use its predictions. GeoSave
adds named stage composition, a declaration of the raster inputs, and a way to
save a model for reuse.

Follow these guides in order:

1. [Build and use a model](models.md): ordinary `nn.Module`, named stages, and
   registered model construction.
2. [Declare and prepare model inputs](model-spec.md): choose bands, prepare
   physical values, and bind prepared rasters to model arguments.
3. [Save and load a release](release.md): preserve the construction recipe and
   weights, then optionally publish to Hugging Face.

`ModelChain` owns the tensor computation. `model_spec.yaml` owns the raster
requirements and input preparation. A release saves the chain's recipe and
weights, with the spec beside them when supplied.

[Training](training.md) shows how the model fits into a LightningModule and
DataModule. [Model input contracts](../concepts/model-inputs.md) explains the
details of references, metadata requirements, tensor transforms, and geospatial
context once the basic workflow is familiar.
