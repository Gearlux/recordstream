"""RecordContract — the pass-through interface op.

One class serves BOTH boundary roles by position: the first op in a chain states what the
host must feed (input contract), the last op states what the pipeline guarantees to
deliver (output contract). These tests pin that dual use, the pass-through identity, and
every error branch (missing entry, wrong item type, plain value, unknown declared type).
"""

import numpy as np
import pytest

from recordstream import Image, Label, Mask, Record, Stream
from recordstream.ops.contract import ClassNamesOutput, ClassNamesScan, ContractError, RecordContract


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


class TestClassNamesOutput:
    """The graph's OUTPUT declaration. It READS what it was given and derives nothing.

    Three routes, each chosen explicitly by the graph author: type the names, connect something
    that carries a vocabulary, or connect a walker. This class picks none of them on their
    behalf -- no scanning fallback, because a walk is seconds per thousand records and would run
    on every graph open.
    """

    def test_route_one_the_names_are_typed_in(self) -> None:
        assert ClassNamesOutput(names=["cat", "dog"]).class_names == ["cat", "dog"]
        assert ClassNamesOutput(names=["cat", "dog"]).num_classes == 2

    def test_route_two_a_wired_vocabulary_arrives_as_the_list(self) -> None:
        """A wire is FOLDED into the document as a literal, so the node only ever sees a list."""
        assert ClassNamesOutput(names=ClassNamesScan(source=[{"class": Label(value="x")}]).class_names).class_names == [
            "x"
        ]

    def test_nothing_given_is_empty_never_a_guess(self) -> None:
        assert ClassNamesOutput().class_names == [] and ClassNamesOutput().num_classes == 0

    def test_zero_arg_construction_is_legal(self) -> None:
        """A visual editor PLACES it unfilled, so it must build empty."""
        assert ClassNamesOutput().names == []

    def test_it_holds_only_the_list_and_never_scans(self) -> None:
        """The CON case that defines this class: it has nothing to walk with."""
        import inspect

        params = list(inspect.signature(ClassNamesOutput).parameters)
        assert params == ["names"], "a second input is a second way to be wrong"


class TestClassNamesScan:
    """The walker an author PLACES when a source declares no vocabulary of its own."""

    def test_it_reports_sorted_unique_names(self) -> None:
        records = [{"class": Label(value="dog")}, {"class": Label(value="cat")}, {"class": Label(value="dog")}]
        assert ClassNamesScan(source=records).class_names == ["cat", "dog"]

    def test_encoded_labels_report_their_ids_as_strings(self) -> None:
        """A walk over encoded labels can only honestly report the ids it saw -- which is also
        what the dataset's own metadata says (MNIST's names really are '0'...'9')."""
        records = [{"class": Label(value=2)}, {"class": Label(value=0)}, {"class": Label(value=1)}]
        assert ClassNamesScan(source=records).class_names == ["0", "1", "2"]

    def test_the_label_key_is_configurable(self) -> None:
        assert ClassNamesScan(source=[{"target": Label(value="x")}], key="target").class_names == ["x"]

    def test_records_without_a_label_are_skipped_not_fatal(self) -> None:
        assert ClassNamesScan(source=[{"class": Label(value="dog")}, {"image": 1}]).class_names == ["dog"]

    def test_no_source_is_an_empty_vocabulary(self) -> None:
        assert ClassNamesScan().class_names == [] and ClassNamesScan().num_classes == 0

    def test_the_walk_goes_through_the_projection_protocol(self) -> None:
        """Key-restricted, so a projection-aware source is asked only for the label column --
        that is what makes a walk affordable at all."""
        import inspect

        import recordstream.ops.contract as module

        assert "iter_key" in inspect.getsource(module.ClassNamesScan)
