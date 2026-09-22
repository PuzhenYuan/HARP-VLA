"""Load the merged HARP-VLA policy and its continuous-action components."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from harpvla.vision import inject_adapter
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector


class HARPForActionPrediction(OpenVLAForActionPrediction):
    def __init__(self, config):
        super().__init__(config)
        # Install adapters before loading weights, so all parameters load once.
        inject_adapter(self.vision_backbone)


def resolve_checkpoint(checkpoint):
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path
    if str(checkpoint) != "ypz21/HARP_VLA_calvin":
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    return Path(snapshot_download(repo_id=str(checkpoint)))


def load_component(module, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    module.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.bfloat16).eval()


def load_policy(checkpoint="ypz21/HARP_VLA_calvin", device="cuda:0"):
    path = resolve_checkpoint(checkpoint)
    config = OpenVLAConfig.from_pretrained(path)
    vla, info = HARPForActionPrediction.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, output_loading_info=True,
    )
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint does not match the HARP-VLA architecture: {info}")
    vla.vision_backbone.set_num_images_in_input(2)
    with (path / "normalization.json").open() as stream:
        vla.norm_stats = json.load(stream)
    vla = vla.to(device).eval()
    processor = PrismaticProcessor(
        image_processor=PrismaticImageProcessor.from_pretrained(path),
        tokenizer=AutoTokenizer.from_pretrained(path, trust_remote_code=False),
    )
    action_head = load_component(
        L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=7),
        path / "action_head.pt", device,
    )
    proprio_projector = load_component(
        ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=7),
        path / "proprio_projector.pt", device,
    )
    return SimpleNamespace(vla=vla, processor=processor, action_head=action_head,
                           proprio_projector=proprio_projector)
