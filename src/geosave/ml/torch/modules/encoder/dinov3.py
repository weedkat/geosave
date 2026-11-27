import timm
import torch.nn as nn

from typing import cast
from timm.models.eva import Eva

from geosave.ml.torch.chain.registry import MODULES
from geosave.ml.torch.chain.published import Published
from geosave.ml.torch.typing import Embedding, FeatureMap, Image, PrefixTokens

# Sample each model's block depth at even quarters.
MODEL_SPECS: dict[str, dict] = {
    "vit_small_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_plus_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_small_plus_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_base_patch16_dinov3.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_base_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (2, 5, 8, 11)},
    "vit_large_patch16_dinov3.lvd1689m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3.sat493m": {"out_indices": (5, 11, 17, 23)},
    "vit_large_patch16_dinov3_qkvb.sat493m": {"out_indices": (5, 11, 17, 23)},
    "vit_huge_plus_patch16_dinov3.lvd1689m": {"out_indices": (7, 15, 23, 31)},
    "vit_huge_plus_patch16_dinov3_qkvb.lvd1689m": {"out_indices": (7, 15, 23, 31)},
    "vit_7b_patch16_dinov3.lvd1689m": {"out_indices": (9, 19, 29, 39)},
    "vit_7b_patch16_dinov3.sat493m": {"out_indices": (9, 19, 29, 39)},
}


class BaseDINOv3(nn.Module):
    """timm DINOv3 backbone construction, shared by `DINOv3` and `DINOv3Embedding`.

    The variants differ only in which of timm's forwards they run and what
    they return, so they are siblings with one construction.
    """

    input_size: Published[int | tuple[int, int]]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        in_channels: int = 3,
        input_size: int | tuple[int, int] = 224,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """Build a timm DINOv3 backbone.

        Args:
            model_name: timm model name. Must be a key of `MODEL_SPECS`.
            pretrained: load pretrained weights from timm hub.
            in_channels: input image channel count. It sizes and initialises
                only the patch embedding; no pixel is changed. The checkpoint
                is RGB, so with `pretrained` the first three channels keep their
                kernel slices only when they are red, green and blue, in that
                order. timm repeats those three slices over further channels
                and scales every slice by `3 / in_channels`.
            input_size: input spatial size; tuple for non-square.
            drop_path_rate: stochastic depth rate.
            proj_drop_rate: dropout on attention output projection.
            attn_drop_rate: dropout on attention weights.
            init_values: initial layer-scale value; DINOv3 pretraining uses 1e-5.
                Only the init — a real checkpoint's trained values override this at
                load time, so it never affects `pretrained=True` weight compatibility.
            dynamic_img_size: interpolate positional embeddings for variable input sizes.
            dynamic_img_pad: pad input to nearest patch multiple when dynamic sizing.

        Raises:
            ValueError: `model_name` is not in `MODEL_SPECS`.
        """
        super().__init__()
        if model_name not in MODEL_SPECS:
            raise ValueError(
                f"{model_name!r} not in MODEL_SPECS; must be one of {list(MODEL_SPECS)}"
            )

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            img_size=input_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            # Shapes the register-token embedding, so an off-spec count breaks loading.
            num_reg_tokens=4,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.model = cast(Eva, model)
        self.input_size = input_size


@MODULES.register("dinov3")
class DINOv3(BaseDINOv3):
    """A DINOv3 encoder returning a feature pyramid from intermediate blocks, with their prefix tokens."""

    returns = ("pyramid", "prefix_tokens")

    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        in_channels: int = 3,
        input_size: int | tuple[int, int] = 224,
        out_indices: list[int] | None = None,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """Build a timm DINOv3 backbone and choose its pyramid levels.

        Args:
            model_name: As for `BaseDINOv3`.
            pretrained: As for `BaseDINOv3`.
            in_channels: As for `BaseDINOv3`.
            input_size: As for `BaseDINOv3`.
            out_indices: Transformer blocks to return as pyramid levels.
                None or an empty list uses this model's four depth samples.
            drop_path_rate: As for `BaseDINOv3`.
            proj_drop_rate: As for `BaseDINOv3`.
            attn_drop_rate: As for `BaseDINOv3`.
            init_values: As for `BaseDINOv3`.
            dynamic_img_size: As for `BaseDINOv3`.
            dynamic_img_pad: As for `BaseDINOv3`.

        Raises:
            ValueError: `model_name` is not in `MODEL_SPECS`.
        """
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            input_size=input_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.out_indices: list[int] = out_indices or list(
            MODEL_SPECS[model_name]["out_indices"]
        )

        feature_info: list = self.model.feature_info
        self.pyramid_channels = [
            int(feature_info[i]["num_chs"]) for i in self.out_indices
        ]
        self.pyramid_strides = [  # [16, 16, 16, 16]
            int(feature_info[i]["reduction"]) for i in self.out_indices
        ]

    def forward(self, image: Image) -> tuple[list[FeatureMap], list[PrefixTokens]]:
        """Extract multi-scale intermediate features from the ViT.

        Args:
            image: (B, C, H, W) input tensor.

        Returns:
            (pyramid, prefix_tokens) — list of per-level feature maps, list
            of per-level prefix tokens.
        """
        pairs = self.model.forward_intermediates(
            image,
            indices=self.out_indices,
            intermediates_only=True,
            return_prefix_tokens=True,
            output_fmt="NCHW",
        )
        pyramid, prefix_tokens = map(list, zip(*pairs))
        return pyramid, prefix_tokens


@MODULES.register("dinov3_embedding")
class DINOv3Embedding(BaseDINOv3):
    """A DINOv3 encoder returning one pooled embedding per image, as image feature extraction does.

    The embedding is timm's pre-logits output: the final tokens, normalized
    and pooled with the model's `global_pool`.
    """

    returns = ("embedding",)

    embedding_channels: Published[int]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        in_channels: int = 3,
        input_size: int | tuple[int, int] = 224,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_values: float | None = 1e-5,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        """Build a timm DINOv3 backbone; arguments as for `BaseDINOv3`.

        Raises:
            ValueError: `model_name` is not in `MODEL_SPECS`.
        """
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            input_size=input_size,
            drop_path_rate=drop_path_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.embedding_channels = int(self.model.num_features)

    def forward(self, image: Image) -> Embedding:
        """Return each image's pooled embedding.

        Args:
            image: (B, C, H, W) input tensor.

        Returns:
            (B, embedding_channels) embeddings.
        """
        tokens = self.model.forward_features(image)
        return self.model.forward_head(tokens, pre_logits=True)
