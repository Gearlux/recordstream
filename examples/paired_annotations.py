"""PairedSource walkthrough: binary-first, annotation-first, broadcast, slicing.

Runs standalone with no external data. Demonstrates the four scenarios the
``PairedSource`` primitive is designed for, using a tiny in-memory primary
source and a dict-shaped annotation store.
"""

from typing import Any, Dict, Iterator, Optional

import confluid  # type: ignore[import-not-found]

from dataflux.paired import PairedSource
from dataflux.sample import Sample


@confluid.configurable
class WindowedSource:
    """Primary source yielding N windows over a mock pack."""

    def __init__(self, pack_id: str = "demo:pack1", n_windows: int = 6, samples_per_window: int = 100) -> None:
        self.pack_id = pack_id
        self.n_windows = n_windows
        self.samples_per_window = samples_per_window

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            input=f"iq_window_{idx}",
            target=None,
            metadata={
                "pack_id": self.pack_id,
                "window_start_sample": idx * self.samples_per_window,
                "window_end_sample": (idx + 1) * self.samples_per_window,
                "samplerate": 1_000_000.0,
            },
        )

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n_windows):
            yield self[i]


@confluid.configurable
class DictStore:
    """Mapping-shaped secondary for demos."""

    def __init__(self, records: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.records = records or {}

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.records[key]

    def keys(self) -> Any:
        return self.records.keys()


# Module-level callables so they survive Confluid YAML round-trip via
# dataflux.discovery.resolve_callable("examples.paired_annotations:<name>").
def window_key(sample: Sample) -> str:
    return f"{sample.metadata['pack_id']}:win{sample.metadata['window_start_sample']:08d}"


def pack_key(sample: Sample) -> str:
    return str(sample.metadata["pack_id"])


def resolve_by_window_key(key: str, primary: WindowedSource) -> Sample:
    for item in primary:
        if window_key(item) == key:
            return item
    raise KeyError(key)


def slice_intervals(record: Dict[str, Any], sample: Sample) -> Optional[Dict[str, Any]]:
    """Trim per-pack time intervals down to each window's range."""
    samplerate = sample.metadata["samplerate"]
    win_start = sample.metadata["window_start_sample"] / samplerate
    win_end = sample.metadata["window_end_sample"] / samplerate
    trimmed = []
    for iv in record.get("intervals", []):
        s, e = max(iv["start_s"], win_start), min(iv["end_s"], win_end)
        if e > s:
            trimmed.append({**iv, "start_s": s, "end_s": e})
    if not trimmed:
        return None
    out = {k: v for k, v in record.items() if k != "intervals"}
    out["intervals"] = trimmed
    return out


def scenario_a_binary_first() -> None:
    """Scenario A: iterate all samples; attach annotation when available."""
    print("\n=== Scenario A: binary-first, annotations optional ===")
    primary = WindowedSource(n_windows=4)
    store = DictStore({"demo:pack1:win00000100": {"label": "dji_mavic", "score": 0.92}})

    paired = PairedSource(primary=primary, secondary=store, key_fn=window_key)

    for s in paired:
        flag = "ANNOTATED" if s.metadata["annotated"] else "        -"
        label = s.metadata.get("label", "")
        print(f"  [{flag}] window_start={s.metadata['window_start_sample']:>4}  label={label!r}")


def scenario_b_annotation_first() -> None:
    """Scenario B: only emit samples that have an annotation."""
    print("\n=== Scenario B: annotation-first, curated labeled subset ===")
    primary = WindowedSource(n_windows=6)
    store = DictStore(
        {
            "demo:pack1:win00000000": {"label": "wifi"},
            "demo:pack1:win00000300": {"label": "lora"},
        }
    )

    paired = PairedSource(primary=primary, secondary=store, key_fn=window_key, policy="inner")

    for s in paired:
        print(f"  key={s.metadata['annotation_key']:<30}  label={s.metadata['label']!r}")
    print(f"  -> {len(list(paired))} samples (out of {len(primary)} in primary)")


def scenario_c1_broadcast() -> None:
    """Scenario C1: one annotation per pack, broadcast to every window."""
    print("\n=== Scenario C1: pack-level broadcast ===")
    primary = WindowedSource(n_windows=4)
    store = DictStore({"demo:pack1": {"drone": "DJI Mavic 3 Pro", "operator": "alice"}})

    paired = PairedSource(primary=primary, secondary=store, key_fn=pack_key)

    for s in paired:
        print(
            f"  win={s.metadata['window_start_sample']:>4}  "
            f"drone={s.metadata['drone']!r}  operator={s.metadata['operator']!r}"
        )


def scenario_c2_slicing() -> None:
    """Scenario C2: pack-level time-ranged annotation, sliced per window."""
    print("\n=== Scenario C2: pack-level time intervals, sliced per window ===")
    primary = WindowedSource(n_windows=6, samples_per_window=100)
    # Samplerate is 1 MHz and windows are 100 samples = 100 us each, so:
    # win0 = [0, 100us], win1 = [100us, 200us], ..., win5 = [500us, 600us].
    # An interval at [150us, 470us] overlaps windows 1, 2, 3, 4.
    store = DictStore(
        {
            "demo:pack1": {
                "drone": "mavic",
                "intervals": [{"start_s": 150e-6, "end_s": 470e-6, "label": "active_emission"}],
            }
        }
    )

    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=pack_key,
        extract_fn=slice_intervals,
    )

    for s in paired:
        win = s.metadata["window_start_sample"]
        if s.metadata["annotated"]:
            iv = s.metadata["intervals"][0]
            print(
                f"  win_start={win:>4}  drone={s.metadata['drone']!r}  "
                f"active=[{iv['start_s']*1e6:.1f}us, {iv['end_s']*1e6:.1f}us]"
            )
        else:
            print(f"  win_start={win:>4}  no overlap")


def scenario_d_right_driven() -> None:
    """Scenario D: iterate the annotation store, resolve primary on demand."""
    print("\n=== Scenario D: right-driven (sparse labels, large primary) ===")
    primary = WindowedSource(n_windows=1000)  # pretend this is huge
    store = DictStore(
        {
            "demo:pack1:win00000000": {"label": "wifi"},
            "demo:pack1:win00050000": {"label": "lora"},
            "demo:pack1:win00099900": {"label": "dji_ocusync"},
        }
    )

    paired = PairedSource(
        primary=primary,
        secondary=store,
        key_fn=window_key,
        policy="right_driven",
        primary_resolver=resolve_by_window_key,
    )

    for s in paired:
        print(f"  key={s.metadata['annotation_key']:<30}  label={s.metadata['label']!r}")


def scenario_e_confluid_roundtrip() -> None:
    """Show that the pipeline survives YAML serialization via Confluid."""
    print("\n=== Scenario E: Confluid YAML round-trip ===")
    primary = WindowedSource(n_windows=3)
    store = DictStore({"demo:pack1:win00000000": {"label": "wifi"}})
    paired = PairedSource(primary=primary, secondary=store, key_fn=window_key)

    yaml_state = confluid.dump(paired)
    print(yaml_state)
    restored = confluid.load(yaml_state)
    print(f"  restored.policy = {restored.policy!r}")
    print(f"  restored.key_fn = {restored.key_fn!r}")


def main() -> None:
    scenario_a_binary_first()
    scenario_b_annotation_first()
    scenario_c1_broadcast()
    scenario_c2_slicing()
    scenario_d_right_driven()
    scenario_e_confluid_roundtrip()
    print("\nOK — paired_annotations.py finished.")


if __name__ == "__main__":
    main()
