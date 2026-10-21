"""Local, reversible additive projection interventions; imports torch lazily."""
from contextlib import contextmanager


@contextmanager
def contribution_hook(component, position, vector, mean_vector, alpha=0., coordinate=None):
    """Replace a fixed-baseline group contribution at one token only.

    Hooking the component (not the target linear) ensures the perturbation is
    applied after ComponentModel's replacement, before the MLP nonlinearity.
    All other outputs and the existing delta contribution remain unchanged.
    """
    import torch
    if alpha not in (0, 1, 2) or position < 0:
        raise ValueError("invalid signed perturbation")
    difference = (alpha - 1) * (vector - mean_vector)
    if coordinate is not None:
        if not 0 <= coordinate < difference.shape[-1]:
            raise ValueError("coordinate outside projection output")
        only = torch.zeros_like(difference)
        only[..., coordinate] = difference[..., coordinate]
        difference = only
    def hook(module, args, output):
        if position >= output.shape[-2] or difference.shape[-1] != output.shape[-1]:
            raise ValueError("intervention shape/position mismatch")
        result = output.clone()
        result[:, position] += difference.to(device=output.device, dtype=output.dtype)
        return result
    handle = component.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
