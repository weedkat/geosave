# Recipe processing paused

ModelSpec is being redesigned. Recipe-based DataModules, raster Datasets and
preparation scripts raise `NotImplementedError`. Model stages and training
Modules remain usable with caller-owned DataLoaders supplying their sample
contract. Chain and weight releases do not contain a processing recipe.
