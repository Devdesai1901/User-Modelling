# steering.py
import torch
from typing import Optional

class ResidualSteererLastTok:
    """
    Adds alpha * v to the residual stream at the INPUT of a given layer.
    Applies to ALL tokens (stronger causal effect).
    """
    def __init__(self, layer_idx: int, v: torch.Tensor, alpha: float):
        self.layer_idx = layer_idx
        self.v = v          # (d,) tensor on GPU
        self.alpha = float(alpha)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def add(self, model):
        self.remove()
        block = model.model.layers[self.layer_idx]

        def prehook(module, inputs):
            hidden_states = inputs[0]  # (B, T, d)
            hidden_states = hidden_states + self.alpha * self.v
            return (hidden_states,) + inputs[1:]

        self.handle = block.register_forward_pre_hook(prehook)

    def remove(self):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None
