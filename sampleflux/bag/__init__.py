"""``sampleflux.bag`` — the typed-bag data model with type-dispatched transforms.

A sample is a NAMED BAG of TYPED ITEMS (:class:`TypedSample`), each item owning its own
metadata; ``input``/``target`` are ROLE TAGS on fields, not tuple positions. Transforms
dispatch on item TYPE via a kernel registry, sampling their parameters once per sample so
multi-field consistency (flip image + mask + boxes together) is automatic. External libraries
(torchvision ``transforms.v2``, albumentations) drop into a :class:`Pipeline` bare — a
registered adapter wraps each — and user/domain packages register their own item types,
kernels, adapters, and storage codecs from outside (``register_item`` / ``@Transform.kernel``
/ ``register_adapter`` / ``register_io``).

This is THE sampleflux data model (the legacy ``Sample`` triple is being migrated out; it
survives only until every consumer has flipped). Import the public surface from the PACKAGE
TOP LEVEL (``from sampleflux import TypedSample, Image, Transform, ...``) — the ``bag``
module path is a transitional home. See ``docs/typed-model.md`` (usage) and
``docs/architecture.md`` (rationale).
"""

# Import the adapters for their SIDE EFFECT: each registers a coercion matcher so a bare
# torchvision v2 / albumentations transform can be dropped straight into a Pipeline. This does
# NOT import torchvision/albumentations (the adapters lazy-import their library inside method
# bodies), so `import sampleflux.bag` stays library-free — pinned by test_bag_pipeline.py.
from sampleflux.bag import adapters as _adapters  # noqa: F401,E402  (registration side effect)
from sampleflux.bag.dispatch import dispatch, register_kernel, registered_kernels
from sampleflux.bag.io import (
    EncodedField,
    EncodedItem,
    decode_item,
    decode_sample,
    encode_item,
    encode_sample,
    register_io,
)
from sampleflux.bag.items import (
    Image,
    Label,
    Mask,
    NDArrayItem,
    Regions,
    get_item_type,
    is_item,
    item_data,
    item_type_names,
    item_types,
    register_item,
    with_data,
)
from sampleflux.bag.sample import ROLES, Role, TypedSample, primary
from sampleflux.bag.transform import (
    FunctionTransform,
    Pipeline,
    Transform,
    as_transform,
    coerce_transform,
    register_adapter,
)

__all__ = [
    # data model
    "TypedSample",
    "Role",
    "ROLES",
    "primary",
    # items
    "NDArrayItem",
    "Image",
    "Mask",
    "Regions",
    "Label",
    "register_item",
    "item_types",
    "item_type_names",
    "get_item_type",
    "is_item",
    "item_data",
    "with_data",
    # transforms
    "Transform",
    "Pipeline",
    "FunctionTransform",
    "as_transform",
    "register_adapter",
    "coerce_transform",
    # dispatch
    "dispatch",
    "register_kernel",
    "registered_kernels",
    # storage codec
    "EncodedItem",
    "EncodedField",
    "register_io",
    "encode_item",
    "decode_item",
    "encode_sample",
    "decode_sample",
]
