from typing import Any, Dict

from dataflux.hf_core import HFFlux
from dataflux.sample import Sample


def test_hfflux_from_list() -> None:
    raw_data = [
        Sample(input="data1", target=0, metadata={"id": "s1"}),
        Sample(input="data2", target=1, metadata={"id": "s2"}),
    ]
    flux = HFFlux.from_source(raw_data)
    assert len(flux) == 2
    assert flux[0].input == "data1"
    assert flux[0].metadata["id"] == "s1"


def test_hfflux_map() -> None:
    raw_data = [Sample(input=1), Sample(input=2)]
    flux = HFFlux.from_source(raw_data)

    # HF style map expects a dict and returns a dict
    def double(example: Dict[str, Any]) -> Dict[str, Any]:
        example["input"] = example["input"] * 2
        return example

    mapped = flux.map(double)
    assert mapped[0].input == 2
    assert mapped[1].input == 4


def test_hfflux_filter() -> None:
    raw_data = [Sample(input=1), Sample(input=2), Sample(input=3)]
    flux = HFFlux.from_source(raw_data)

    filtered = flux.filter(lambda x: x["input"] > 1)
    assert len(filtered) == 2
    assert filtered[0].input == 2
