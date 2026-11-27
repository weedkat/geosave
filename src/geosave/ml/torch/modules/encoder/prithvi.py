from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn as nn

from geosave.ml.torch.chain.registry import MODULES
from geosave.ml.torch.chain.published import Published
from geosave.ml.torch.typing import (
    Centroid,
    DayOfYear,
    FeatureMap,
    Image,
    ImageSeries,
    PrefixTokens,
    Year,
)

if TYPE_CHECKING:
    from terratorch.models.backbones.prithvi_mae import PrithviViT


class BasePrithvi(nn.Module):
    """Prithvi-EO backbone construction and pyramid extraction, shared by `Prithvi` and `PrithviTL`.

    The variants differ only in what their forward reads, so they are
    siblings: each declares its checkpoints in `MODEL_SPECS` and its own
    forward.
    """

    returns = ("pyramid", "prefix_tokens")

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    input_size: Published[int | tuple[int, int]]

    MODEL_SPECS: ClassVar[dict[str, dict]]

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
        pretrained: bool = False,
        bands: list[str] | None = None,
        input_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        transpose_time_band: bool = True,
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
            transpose_time_band: swap the time and band axes of an image
                series, from `(B, T, C, H, W)` as `gs.to_tensor` stacks it to
                `(B, C, T, H, W)` as Prithvi reads it. False passes series
                already in Prithvi's order. A single image, `(B, C, H, W)`,
                has no time axis to swap.
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
        # TerraTorch loads Lightning, so registering this class must not import it.
        from terratorch.registry import BACKBONE_REGISTRY

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
        self.model: PrithviViT = cast("PrithviViT", model)

        self.pyramid_channels = [self.model.out_channels[i] for i in self.out_indices]
        model_patch_size = self.model.patch_embed.patch_size  # (t, h, w)
        self.pyramid_strides = [model_patch_size[-1]] * len(self.out_indices)

        self.input_size = input_size
        self.transpose_time_band = transpose_time_band

    def _transpose_time_band(self, image: Image | ImageSeries) -> torch.Tensor:
        """Return the image in Prithvi's axis order, swapping a series' time and band axes when `transpose_time_band` is set."""
        if self.transpose_time_band and image.ndim == 5:
            return image.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
        return image

    def _extract_pyramid(
        self, features: list[torch.Tensor]
    ) -> tuple[list[FeatureMap], list[PrefixTokens]]:
        """Return the pyramid and CLS tokens of the `out_indices` blocks of the backbone's features.

        Args:
            features: Per-block `(B, 1+N_patches, embed_dim)` tokens, CLS first,
                as `forward_features` returns them.
        """
        selected = []
        for index in self.out_indices:
            selected.append(features[index])

        prefix_tokens: list[torch.Tensor | None] = []
        for level in selected:
            prefix_tokens.append(level[:, :1, :])  # (B, 1, embed_dim), CLS only
        pyramid = self.model.prepare_features_for_image_model(
            selected
        )  # list of (B, embed_dim, H/patch, W/patch)
        return pyramid, prefix_tokens


@MODULES.register("prithvi")
class Prithvi(BasePrithvi):
    """Prithvi encoder accepting prepared images in geodata axis order.

    Inputs have shape (B, C, H, W) or (B, T, C, H, W). `PrithviTL` also reads
    acquisition time and location.
    """

    # Sample each transformer depth at even quarters.
    MODEL_SPECS: ClassVar[dict[str, dict]] = {
        "prithvi_eo_v1_100": {"out_indices": (2, 5, 8, 11)},
        "prithvi_eo_v2_300": {"out_indices": (5, 11, 17, 23)},
        "prithvi_eo_v2_600": {"out_indices": (7, 15, 23, 31)},
    }

    def forward(
        self, image: Image | ImageSeries
    ) -> tuple[list[FeatureMap], list[PrefixTokens]]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) or (B, T, C, H, W) prepared pixels.

        Returns:
            (pyramid, prefix_tokens) — list of per-level (B, C, H, W) feature
            maps, list of per-level (B, 1, C) CLS tokens.
        """
        features = self.model.forward_features(self._transpose_time_band(image))
        return self._extract_pyramid(features)


@MODULES.register("prithvi_tl")
class PrithviTL(BasePrithvi):
    """Prithvi encoder conditioned on acquisition time and geographic location.

    It reads them from the batch's `centroid`, `year` and `day_of_year`, as
    `DataTree.gs.to_tensor(context=...)` with the readers of the same names
    writes them, after DataLoader collation.
    """

    MODEL_SPECS: ClassVar[dict[str, dict]] = {
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
        transpose_time_band: bool = True,
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
            transpose_time_band: Swap an image series' time and band axes,
                from `(B, T, C, H, W)` to Prithvi's `(B, C, T, H, W)`, as for
                `Prithvi`.
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
            transpose_time_band=transpose_time_band,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            ckpt_path=ckpt_path,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )

    def forward(
        self,
        image: Image | ImageSeries,
        centroid: Centroid,
        year: Year,
        day_of_year: DayOfYear,
    ) -> tuple[list[FeatureMap], list[PrefixTokens]]:
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
            >>> pixels, context = tree.gs.to_tensor(
            ...     context={"centroid": "centroid", "year": "year", "day_of_year": "day_of_year"}
            ... )
            >>> sample = {**pixels, **context}
            >>> batch = default_collate([sample])
            >>> pyramid, _ = PrithviTL()(**batch)
        """
        # The pretrained model counts days from zero.
        temporal = torch.stack([year, day_of_year - 1], dim=-1).float()
        location = centroid.float()
        features = self.model.forward_features(
            self._transpose_time_band(image),
            temporal_coords=temporal,
            location_coords=location,
        )
        return self._extract_pyramid(features)
