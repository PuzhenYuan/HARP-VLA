import torch
import torch.nn as nn
import torch.nn.functional as F




class SimpleAdapter(nn.Module):
    def __init__(self, dim, reduction=16, dropout=0.0, scale=1.0):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = scale
        # 常见做法：只训 adapter，故初始化要小（近似 0）
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Linear(d_model -> r) → GELU → Linear(r -> d_model) + residual
        residual = x
        x = self.down(x)
        x = F.gelu(x)
        x = self.up(x)
        x = self.dropout(x)
        return residual + self.scale * x

class TokenAdapterHook:
    """
    Token-aware adapter hook.

    token_cfg example:

    DINOv2 ViT-L (reg4):
    {
        "has_cls": True,
        "cls": False,
        "reg": True,
        "patch": True,
        "num_reg": 4,
    }

    SigLIP So400M:
    {
        "has_cls": False,
        "patch": True,
        "num_reg": 0,
    }
    """

    def __init__(self, adapter, token_cfg):
        self.adapter = adapter
        self.token_cfg = token_cfg

    def __call__(self, module, inputs, output):
        """
        output: Tensor [B, T, D]
        """
        x = output
        B, T, D = x.shape

        idx = 0
        chunks = []

        # ---- CLS token (only if exists) ----
        if self.token_cfg.get("has_cls", False):
            cls = x[:, idx:idx+1]
            if self.token_cfg.get("cls", False):
                cls = self.adapter(cls)
            chunks.append(cls)
            idx += 1

        # ---- REG tokens (only if exist) ----
        num_reg = self.token_cfg.get("num_reg", 0)
        if num_reg > 0:
            reg = x[:, idx:idx+num_reg]
            if self.token_cfg.get("reg", False):
                reg = self.adapter(reg)
            chunks.append(reg)
            idx += num_reg

        # ---- PATCH tokens (always remaining) ----
        if idx < T:
            patch = x[:, idx:]
            if self.token_cfg.get("patch", True):
                patch = self.adapter(patch)
            chunks.append(patch)
        # Checking
        expected = ((1 if self.token_cfg.get("has_cls", False) else 0) + self.token_cfg.get("num_reg", 0))
        assert T >= expected, f"Token layout mismatch: T={T}, expected at least {expected}"

        return torch.cat(chunks, dim=1)


class BaseAdapterInjector:
    """
    Unified adapter injector.
    Subclasses must implement:
      - _get_layers() -> list-like
      - _get_dim(layer) -> int
      - _get_target_submodule(layer, where: str) -> nn.Module
    for DINO:
        token_cfg = {
            "cls": False,
            "reg": True,
            "patch": True,
            "num_reg": 4,
        }
    for SigLIP
    token_cfg = {
        "cls": False,     # 通常不动 CLS
        "patch": True,
        "num_reg": 0,
    }
    """

    SUPPORTED_WHERE = ("attn", "mlp")

    def __init__(
        self,
        model,
        which_layers="all",
        reduction=32,   # for ViT-L:32 / ViT-B: 16
        dropout=0.0,
        scale=1.0,
        where=("attn", "mlp"),
        adapter_cls=SimpleAdapter,
        token_cfg=None,
    ):
        self.model = model
        self.handles = []

        self.reduction = reduction
        self.dropout = dropout
        self.scale = scale
        self.adapter_cls = adapter_cls
        self.token_cfg = token_cfg or {}

        # normalize where
        where = tuple(where)
        for w in where:
            if w not in self.SUPPORTED_WHERE:
                raise ValueError(f"Unsupported where='{w}'. Supported: {self.SUPPORTED_WHERE}")
        self.where = where

        layers = self._get_layers()
        if which_layers == "all":
            self.layer_ids = list(range(len(layers)))
        else:
            self.layer_ids = list(which_layers)

    # --- methods subclass must provide ---
    def _get_layers(self):
        raise NotImplementedError

    def _get_dim(self, layer):
        raise NotImplementedError

    def _get_target_submodule(self, layer, where: str) -> nn.Module:
        raise NotImplementedError

    # --- common hook logic ---
    def _register_adapter_on_submodule_output(self, layer, where: str, dim: int):
        adapter = self.adapter_cls(dim, self.reduction, self.dropout, self.scale)
        setattr(layer, f"{where}_adapter", adapter)

        submod = self._get_target_submodule(layer, where)
        hook_fn = TokenAdapterHook(adapter, self.token_cfg)

        handle = submod.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def inject(self):
        layers = self._get_layers()
        for i in self.layer_ids:
            layer = layers[i]
            dim = self._get_dim(layer)

            for w in self.where:
                submod = None
                try:
                    submod = self._get_target_submodule(layer, w)
                except AttributeError:
                    submod = None
                if submod is not None:
                    self._register_adapter_on_submodule_output(layer, w, dim)

        return self.model  # keep your DINO-style "inj.inject() -> model"

    def adapters(self):
        layers = self._get_layers()
        for i in self.layer_ids:
            layer = layers[i]
            for w in self.where:
                name = f"{w}_adapter"
                if hasattr(layer, name):
                    yield getattr(layer, name)

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# -------------------------
# DINO injector
# -------------------------
class DINOAdapterInjector(BaseAdapterInjector):
    def _get_layers(self):
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError("DINO-like ViT expected: model.blocks not found.")
        return blocks

    def _get_dim(self, block):
        # prefer norm1 dim
        if hasattr(block, "norm1"):
            return block.norm1.normalized_shape[0]
        # fallback: mlp.fc1.in_features
        if hasattr(block, "mlp") and hasattr(block.mlp, "fc1"):
            return block.mlp.fc1.in_features
        raise ValueError("Cannot infer hidden dim for DINO block.")

    def _get_target_submodule(self, block, where: str) -> nn.Module:
        # DINO blocks usually have .attn and .mlp
        return getattr(block, where)


class SigLIPAdapterInjector(BaseAdapterInjector):
    """
    Adapter injector for timm SigLIP ViT models.

    Assumptions (timm vit_so400m_patch14_siglip_224):
      - model.blocks exists
      - block.attn / block.mlp exist
      - forward_features returns PATCH tokens only:
            [B, N_patches, D]
        (NO CLS, NO REG)
    """

    def _get_layers(self):
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise ValueError("SigLIP ViT expected: model.blocks not found.")
        return blocks

    def _get_dim(self, block):
        # SigLIP ViT-L / So400M: dim = 768
        if hasattr(block, "norm1"):
            return block.norm1.normalized_shape[0]
        if hasattr(block, "mlp") and hasattr(block.mlp, "fc1"):
            return block.mlp.fc1.in_features
        raise ValueError("Cannot infer hidden dim for SigLIP block.")

    def _get_target_submodule(self, block, where: str):
        # timm ViT blocks expose .attn and .mlp
        return getattr(block, where)
