"""Adapters that run external augmentation libraries as typed-bag transforms.

These are deliberately NOT imported by :mod:`sampleflux.bag`'s top-level ``__init__`` — each
lazy-imports its library inside method bodies, so ``import sampleflux.bag`` stays safe on a
host without torchvision. Import an adapter directly::

    from sampleflux.bag.adapters import TorchvisionV2Adapter, AlbumentationsAdapter
"""

from sampleflux.bag.adapters.albumentations import AlbumentationsAdapter
from sampleflux.bag.adapters.torchvision import TorchvisionV2Adapter

__all__ = ["TorchvisionV2Adapter", "AlbumentationsAdapter"]
