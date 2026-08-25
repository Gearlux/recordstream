"""RecordContract — the pass-through interface op.

One class serves BOTH boundary roles by position: the first op in a chain states what the
host must feed (input contract), the last op states what the pipeline guarantees to
deliver (output contract). These tests pin that dual use, the pass-through identity, and
every error branch (missing entry, wrong item type, plain value, unknown declared type).
"""

import numpy as np
import pytest

from recordstream import Image, Label, Mask, Record, Stream
from recordstream.ops.contract import ContractError, RecordContract


def classification_record() -> Record:
    return {"image": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "class": Label(value=3)}


class TestPassThrough:
    def test_zero_arg_contract_passes_any_record_through(self) -> None:
        record = {"anything": 1.5}
        assert RecordContract()(record) is record

    def test_satisfied_contract_returns_the_same_record_object(self) -> None:
        contract = RecordContract(fields={"image": "Image", "class": "Label"}, name="classification output")
        record = classification_record()
        assert contract(record) is record
        assert list(record) == ["image", "class"]  # nothing added, nothing dropped

    def test_a_subclass_of_the_declared_item_type_satisfies_it(self) -> None:
        class DomainImage(Image):  # deliberately NOT register_item-ed — isinstance is the rule
            pass

        contract = RecordContract(fields={"image": "Image"})
        record = {"image": DomainImage(np.zeros((2, 2)))}
        assert contract(record) is record


class TestWildcard:
    def test_wildcard_requires_presence_only(self) -> None:
        contract = RecordContract(fields={"image": "*"})
        record = {"image": np.zeros((2, 2))}  # a plain value — no item type demanded
        assert contract(record) is record

    def test_wildcard_still_fails_on_an_absent_entry(self) -> None:
        contract = RecordContract(fields={"image": "*"}, name="probe")
        with pytest.raises(ContractError, match=r"probe: record #0 has no entry 'image'"):
            contract({"other": 1})


class TestViolations:
    def test_missing_entry_names_contract_field_expected_and_present(self) -> None:
        contract = RecordContract(fields={"image": "Image", "class": "Label"}, name="classification output")
        record = {"image": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "samplerate": 30.72e6}
        with pytest.raises(ContractError) as excinfo:
            contract(record)
        message = str(excinfo.value)
        assert "classification output" in message
        assert "has no entry 'class' (expected Label)" in message
        assert "image[Image]" in message  # what IS present, with its type
        assert "samplerate[float]" in message

    def test_wrong_item_type_names_got_and_expected(self) -> None:
        contract = RecordContract(fields={"class": "Label"}, name="classification output")
        record = {"class": Mask(np.zeros((2, 2)))}
        with pytest.raises(
            ContractError, match=r"classification output: record #0 entry 'class' is a Mask, expected Label"
        ):
            contract(record)

    def test_plain_value_under_a_typed_declaration_names_its_python_type(self) -> None:
        contract = RecordContract(fields={"class": "Label"})
        with pytest.raises(ContractError, match=r"record contract: record #0 entry 'class' is a int, expected Label"):
            contract({"class": 3})

    def test_unknown_declared_type_name_raises_lazily_naming_the_known_types(self) -> None:
        contract = RecordContract(fields={"image": "Picture"})  # construction stays cheap and silent
        with pytest.raises(ContractError, match=r"no item type registered as 'Picture'") as excinfo:
            contract({"image": 1})
        assert "Image" in str(excinfo.value)  # the known-types list travels with the error

    def test_the_record_ordinal_counts_per_instance(self) -> None:
        contract = RecordContract(fields={"class": "Label"}, name="probe")
        contract(classification_record())
        with pytest.raises(ContractError, match=r"probe: record #1 "):
            contract({"other": 1})


class TestInputAndOutputPositions:
    """The precondition question: ONE class, both boundary roles, decided by position."""

    @staticmethod
    def _wrap_raw_as_image(record: Record) -> Record:
        record["image"] = Image(record.pop("raw"))
        return record

    def test_one_contract_class_serves_input_and_output_positions(self) -> None:
        stream = Stream(
            source=[{"raw": np.zeros((2, 2)), "class": Label(value=1)}],
            ops=[
                RecordContract(fields={"raw": "*", "class": "Label"}, name="classification input"),
                self._wrap_raw_as_image,
                RecordContract(fields={"image": "Image", "class": "Label"}, name="classification output"),
            ],
        )
        records = list(stream)
        assert len(records) == 1
        assert isinstance(records[0]["image"], Image)

    def test_a_bad_source_fails_at_the_input_contract_by_name(self) -> None:
        stream = Stream(
            source=[{"class": Label(value=1)}],  # no "raw" — the INPUT contract must be the one that fires
            ops=[
                RecordContract(fields={"raw": "*", "class": "Label"}, name="classification input"),
                self._wrap_raw_as_image,
                RecordContract(fields={"image": "Image", "class": "Label"}, name="classification output"),
            ],
        )
        with pytest.raises(ContractError, match=r"classification input: record #0 has no entry 'raw'"):
            list(stream)

    def test_a_bad_pipeline_fails_at_the_output_contract_by_name(self) -> None:
        stream = Stream(
            source=[{"raw": np.zeros((2, 2)), "class": Label(value=1)}],
            ops=[
                RecordContract(fields={"raw": "*"}, name="classification input"),
                RecordContract(fields={"image": "Image"}, name="classification output"),  # nothing made an image
            ],
        )
        with pytest.raises(ContractError, match=r"classification output: record #0 has no entry 'image'"):
            list(stream)
