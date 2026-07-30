"""Answering "is this value from an optional framework?" without importing that framework.

recordstream's core is numpy. A handful of places still need to recognise a torch tensor —
storage converting a payload before writing, image conversion detaching one — and doing that
with a module-level ``import torch`` made a 2GB framework a hard dependency of a package whose
own work is arrays.

The trick is that the question answers itself: **a torch tensor cannot exist unless torch has
already been imported.** So consulting ``sys.modules`` is exact, not a heuristic — if the module
is absent the value is provably not one of its types, and if it is present we do a real
``isinstance`` with no import of our own.

This is the same instinct as the op-family matchers in :mod:`recordstream.core`, which identify
an albumentations or torchvision transform by its MRO module name rather than importing either.
"""

import sys
from typing import Any

__all__ = ["is_torch_tensor"]


def is_torch_tensor(value: Any) -> bool:
    """True when ``value`` is a ``torch.Tensor``, without importing torch.

    Exact rather than duck-typed: when ``torch`` is already loaded this is a real
    ``isinstance`` check; when it is not, no torch tensor can exist in the process, so the
    answer is ``False``.

    Example::

        payload = value.detach().cpu().numpy() if is_torch_tensor(value) else np.asarray(value)
    """
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)
