
import torch
from harpvla.adapters import DINOAdapterInjector, SigLIPAdapterInjector

import copy
from functools import partial
from typing import Dict, List, Tuple, Any, Callable


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


def inject_adapter(
        vision_backbone,
        dinov2_adapter_path=None,
        siglip_adapter_path=None,
        device_id=None,
        dtype=None,
    ):
    """inject adapters into the vision backbone: prism-dinosiglip-224px+7b"""

    # in PrismaticVLM, dino_featurizer and siglip_featurizer are used;
    # while in HF, featurizer and fused_featureizer are used
    assert hasattr(vision_backbone, "dino_featurizer") or hasattr(vision_backbone, "featurizer")
    assert hasattr(vision_backbone, "siglip_featurizer") or hasattr(vision_backbone, "fused_featurizer")

    if hasattr(vision_backbone, "dino_featurizer") and hasattr(vision_backbone, "siglip_featurizer"):
        hf_style = False
    elif hasattr(vision_backbone, "featurizer") and hasattr(vision_backbone, "fused_featurizer"):
        hf_style = True
    else:
        raise ValueError("Vision backbone must have either (dino_featurizer and siglip_featurizer) or (featurizer and fused_featurizer)")

    # inject adapters into dinov2
    dino_featurizer = vision_backbone.featurizer if hf_style else vision_backbone.dino_featurizer
    dinov2_inj = DINOAdapterInjector(
        copy.deepcopy(dino_featurizer),
        which_layers="all",
        reduction=16,
        dropout=0.0,
        scale=1,
        where=("attn","mlp"),
        token_cfg = {
            "has_cls": True,
            "cls": False,
            "reg": True,
            "patch": True,
            "num_reg": 4,
        }
    )
    dinov2 = dinov2_inj.inject()
    if dinov2_adapter_path is not None:
        dinov2_state = torch.load(dinov2_adapter_path, map_location="cpu")
        dinov2.load_state_dict(dinov2_state, strict=False)
    if device_id is not None:
        dinov2 = dinov2.to(device_id)
    if dtype is not None:
        dinov2 = dinov2.to(dtype)

    # inject adapters into siglip
    siglip_featurizer = vision_backbone.fused_featurizer if hf_style else vision_backbone.siglip_featurizer
    siglip_inj = SigLIPAdapterInjector(
        copy.deepcopy(siglip_featurizer),
        which_layers="all",
        reduction=32,
        dropout=0.0,
        scale=1,
        where=("attn","mlp"),
        token_cfg = {
            "has_cls": False,
            "patch": True,
            "num_reg": 0,
        }
    )
    siglip = siglip_inj.inject()
    if siglip_adapter_path is not None:
        siglip_state = torch.load(siglip_adapter_path, map_location="cpu")
        siglip.load_state_dict(siglip_state, strict=False)
    if device_id is not None:
        siglip = siglip.to(device_id)
    if dtype is not None:
        siglip = siglip.to(dtype)

    # replace and only use features from the L-2 layer
    if hf_style:
        vision_backbone.featurizer = dinov2
        vision_backbone.fused_featurizer = siglip
        vision_backbone.featurizer.forward = unpack_tuple(
            partial(
                vision_backbone.featurizer.get_intermediate_layers,
                n={len(vision_backbone.featurizer.blocks) - 2},
            )
        )
        vision_backbone.fused_featurizer.forward = unpack_tuple(
            partial(
                vision_backbone.fused_featurizer.get_intermediate_layers,
                n={len(vision_backbone.fused_featurizer.blocks) - 2},
            )
        )
    else:
        vision_backbone.dino_featurizer = dinov2
        vision_backbone.siglip_featurizer = siglip
        vision_backbone.dino_featurizer.forward = unpack_tuple(
            partial(
                vision_backbone.dino_featurizer.get_intermediate_layers,
                n={len(vision_backbone.dino_featurizer.blocks) - 2},
            )
        )
        vision_backbone.siglip_featurizer.forward = unpack_tuple(
            partial(
                vision_backbone.siglip_featurizer.get_intermediate_layers,
                n={len(vision_backbone.siglip_featurizer.blocks) - 2},
            )
        )
