# hooks.py
import torch
from typing import Dict, List

class ResidualStreamCaptureLastTok:
    """
    Captures residual stream at the INPUT of each transformer block,
    but stores ONLY the last-token vector: (d,).
    """
    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self):
        self.cache = {}

    def add_hooks(self, model):
        self.remove_hooks()
        layers = model.model.layers
        for i, block in enumerate(layers):
            h = block.register_forward_pre_hook(self._make_prehook(i))
            self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _make_prehook(self, layer_idx: int):
        def prehook(module, inputs):
            hidden_states = inputs[0]  # (B, T, d)
            # store ONLY last token vec, on CPU, fp32
            self.cache[layer_idx] = hidden_states[0, -1, :].detach().to("cpu", dtype=torch.float32)
        return prehook
