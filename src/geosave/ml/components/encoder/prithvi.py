from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.registry import BACKBONE_REGISTRY

from geosave.ml.components.registry import ENCODERS
from geosave.ml.chain import chain_step
from geosave.ml.components.published import Published


class _Prithvi(nn.Module):
    """Shared Prithvi construction and geodata tensor layout."""

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    input_size: Published[int | tuple[int, int]]

    # Sample each transformer depth at even quarters.
    MODEL_SPECS: dict[str, dict] = {
        "prithvi_eo_v1_100": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_300": {"out_indices": (5, 11, 17, 23)},
        "prithvi_eo_v2_600": {"out_indices": (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
        pretrained: bool = False,
        bands: list[str] | None = None,
        input_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        drop_path_rate: float = 0.0,
        out_indices: list[int] | None = None,
        ckpt_path: str | None = None,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        """Build a terratorch Prithvi-EO backbone.

        Args:
            model_name: terratorch backbone registry name. Must be a key of `MODEL_SPECS`.
            pretrained: load pretrained weights from HuggingFace hub (or `ckpt_path`).
            bands: input bands in channel order, named as TerraTorch's `HLSBands`.
                They size and initialise only the patch embedding, one weight
                slice per input band; no pixel is changed. The checkpoint was
                pretrained on `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`
                (Sentinel-2 B02, B03, B04, B8A, B11, B12; B08 is `NIR_BROAD`),
                and those names keep their pretrained slice. Any other
                `HLSBands` name, or a misspelt one, gets a random slice with
                no error. `None` expects exactly those six, in that order.
            input_size: size the positional embedding buffer is built for, in pixels;
                tuple for non-square. Sizes `model_name`'s patch size does not divide
                are border-cropped. Call-time tensors may differ — the embedding is
                re-interpolated to whatever shape arrives.
            num_frames: number of timesteps in the input (temporal stacking); `1` for
                single-timestep imagery.
            drop_path_rate: stochastic depth rate.
            out_indices: which of the model's blocks to return features from. `None`
                uses this `model_name`'s default (see `MODEL_SPECS`).
            ckpt_path: matching local upstream checkpoint, used only with
                `pretrained=True`; `None` selects the public Hub checkpoint.
            vpt: add learnable Visual Prompt Tuning tokens per block.
                Backbone freezing is a separate training choice.
            vpt_n_tokens: prompt tokens per block. Required if `vpt` is True.
            vpt_dropout: dropout on VPT prompt tokens.

        Raises:
            ValueError: `model_name` not in `MODEL_SPECS`.
        """
        super().__init__()
        model_specs = type(self).MODEL_SPECS
        if model_name not in model_specs:
            raise ValueError(
                f"{model_name!r} not in {type(self).__name__}.MODEL_SPECS; must be one of {list(model_specs)}"
            )

        self.out_indices: list[int] = (
            out_indices
            if out_indices is not None
            else list(model_specs[model_name]["out_indices"])
        )
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=pretrained,
            bands=bands,
            img_size=input_size,
            num_frames=num_frames,
            drop_path=drop_path_rate,
            out_indices=self.out_indices,
            ckpt_path=ckpt_path,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )
        self.model: PrithviViT = cast(PrithviViT, model)

        self.pyramid_channels = [self.model.out_channels[i] for i in self.out_indices]
        model_patch_size = self.model.patch_embed.patch_size  # (t, h, w)
        self.pyramid_strides = [model_patch_size[-1]] * len(self.out_indices)

        self.input_size = input_size

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Run the backbone on geodata-ordered image tensors.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.

        Returns:
            Per-block token tensors from the wrapped backbone.
        """
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        return self.model(image, temporal_coords=None, location_coords=None)


@ENCODERS.register("prithvi")
class Prithvi(_Prithvi):
    """Prithvi encoder accepting prepared images in geodata axis order.

    Inputs have shape (B, C, H, W) or (B, T, C, H, W). Constructor options
    configure the shared Prithvi backbone.
    """

    @chain_step()
    def forward_pyramid(
        self, image: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.
        """
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.model.forward_features(
            image
        )  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0
        features = [
            features[i] for i in self.out_indices
        ]  # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens: list[torch.Tensor | None] = [
            f[:, :1, :] for f in features
        ]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(
            features
        )  # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens


@ENCODERS.register("prithvi_tl")
class PrithviTL(_Prithvi):
    """Prithvi encoder conditioned on acquisition time and geographic location.

    It reads them from the batch's `centroid`, `year` and `day_of_year`, as
    `DataTree.gs.to_tensor(context=["centroid", "year", "day_of_year"])`
    writes them, after DataLoader collation.
    """

    MODEL_SPECS: dict[str, dict] = {
        "prithvi_eo_v2_tiny_tl": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_100_tl": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_300_tl": {"out_indices": (5, 11, 17, 23)},
        "prithvi_eo_v2_600_tl": {"out_indices": (7, 15, 23, 31)},
    }

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300_tl",
        pretrained: bool = False,
        bands: list[str] | None = None,
        input_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        drop_path_rate: float = 0.0,
        out_indices: list[int] | None = None,
        ckpt_path: str | None = None,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        """Build a Prithvi backbone with temporal and location embeddings.

        Args:
            model_name: TerraTorch registry name from this class's `MODEL_SPECS`.
            pretrained: Load upstream encoder weights. False initializes a new
                backbone; True reads `ckpt_path` or downloads from Hugging Face.
            bands: Input bands in channel order, using TerraTorch `HLSBands`
                names. None uses BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2.
                Matching names preserve pretrained patch-embedding slices;
                other names receive random slices. This does not change pixels.
            input_size: Spatial size used to construct position embeddings,
                as a side length or `(height, width)`. The backbone interpolates
                embeddings for call-time grids; patch remainders are cropped.
            num_frames: Number of temporal frames supplied to the encoder.
            drop_path_rate: Backbone stochastic-depth probability.
            out_indices: Transformer blocks returned as pyramid levels. None
                uses the selected model's configured depth samples.
            ckpt_path: Matching local upstream checkpoint, used only when
                `pretrained` is True. None selects the public Hub checkpoint.
            vpt: Add learnable prompt tokens per block; freezing is explicit.
            vpt_n_tokens: Number of prompt tokens per block; required for VPT.
            vpt_dropout: Dropout probability applied to the prompt tokens.

        Raises:
            ValueError: The model name is unsupported, or VPT lacks its token
                count. Upstream checkpoint-loading errors propagate.
        """
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            bands=bands,
            input_size=input_size,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            ckpt_path=ckpt_path,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )

    @chain_step()
    def forward_pyramid(
        self,
        image: torch.Tensor,
        centroid: torch.Tensor,
        year: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """Extract multi-scale intermediate features, conditioned on time/location.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.
            centroid: `(B, 2)` latitude and longitude of each grid's centre.
            year: `(B, T)` year of each frame.
            day_of_year: `(B, T)` day of year of each frame, from 1.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.

        Examples:
            >>> tree = xr.DataTree.from_dict({"image": chip})
            >>> pixels, context = tree.gs.to_tensor(context=["centroid", "year", "day_of_year"])
            >>> sample = {**pixels, **context}
            >>> batch = default_collate([sample])
            >>> pyramid, _ = PrithviTL().forward_pyramid(**batch)
        """
        # The pretrained model counts days from zero.
        temporal = torch.stack([year, day_of_year - 1], dim=-1).float()
        location = centroid.float()
        if image.ndim == 5:
            image = image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.model.forward_features(  # list[depth] of (B, 1+N_patches, embed_dim), CLS at idx 0 # type: ignore
            image, temporal_coords=temporal, location_coords=location
        )
        features = [features[i] for i in self.out_indices]  # pyright: ignore[reportGeneralTypeIssues] # list[len(out_indices)] of (B, 1+N_patches, embed_dim)
        prefix_tokens: list[torch.Tensor | None] = [
            f[:, :1, :] for f in features
        ]  # list[len(out_indices)] of (B, 1, embed_dim) -- CLS only
        pyramid = self.model.prepare_features_for_image_model(features)  # type: ignore # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens
