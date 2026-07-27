"""The kernel registry — exact + MRO dispatch, transform-MRO inheritance, override, cache."""

from typing import Any, Dict

from recordstream import Image, Label, Mask, Regions, Transform
from recordstream.dispatch import dispatch, get_kernel, register_kernel, registered_kernels
from tests._fixtures import FixtureFlip


class TestDispatch:
    def test_exact_hit(self) -> None:
        assert get_kernel(FixtureFlip, Image) is not None
        assert dispatch(FixtureFlip, Image) is get_kernel(FixtureFlip, Image)

    def test_miss_returns_none(self) -> None:
        # FixtureFlip has no Label kernel — a Label value passes through.
        assert dispatch(FixtureFlip, Label) is None
        assert get_kernel(FixtureFlip, Label) is None

    def test_item_mro_walk(self) -> None:
        class SubMask(Mask):
            pass

        # No kernel for SubMask, but its base Mask has one — the MRO walk resolves it.
        assert get_kernel(FixtureFlip, SubMask) is None
        assert dispatch(FixtureFlip, SubMask) is get_kernel(FixtureFlip, Mask)

    def test_transform_mro_inheritance(self) -> None:
        class TunedFlip(FixtureFlip):
            pass

        # A subclass transform inherits its base's kernels until it overrides them.
        assert dispatch(TunedFlip, Image) is get_kernel(FixtureFlip, Image)

    def test_override_wins_and_invalidates_cache(self) -> None:
        class T(Transform):
            pass

        assert dispatch(T, Image) is None  # populate the cache with a miss

        @register_kernel(T, Image)
        def _kernel(item: Any, params: Dict[str, Any]) -> Any:
            return item

        assert dispatch(T, Image) is _kernel  # cache was cleared on registration

    def test_plain_value_type_misses(self) -> None:
        # A plain (non-item) value type — e.g. float — has no kernel: the op passes it through.
        assert dispatch(FixtureFlip, float) is None

    def test_registered_kernels_lists_pairs(self) -> None:
        pairs = registered_kernels()
        assert ("FixtureFlip", "Image") in pairs
        assert ("FixtureFlip", "Regions") in pairs
        assert dispatch(FixtureFlip, Label) is None  # FixtureFlip does not handle Label
        assert dispatch(FixtureFlip, Regions) is get_kernel(FixtureFlip, Regions)
