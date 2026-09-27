"""GraphContract — what ONE graph must deliver, declared as data and checked in one place.

A consuming workspace opens several graphs (a source graph, a files graph, a sink graph, …) and
each must deliver something different: a stream of records carrying an image and a label, a
sink, a vocabulary. One class states that per graph — `outputs` (every entry required),
`records` (what each record of a delivered stream carries) — and `check()` is the single
verification the editor and the workspace both call, so a graph refused in one is refused in
the other with the same words.
"""

import re
from typing import List

import numpy as np
import pytest

from recordstream import Image, Label, Record, Stream
from recordstream.ops.contract import GRAPH_SLOT_KINDS, ContractError, GraphContract, RecordContract


def _records() -> List[Record]:
    return [
        {"input": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "target": Label(value=1)},
        {"input": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "target": Label(value=0)},
    ]


def classification() -> GraphContract:
    return GraphContract(
        name="classification source",
        outputs={"stream": "Stream", "class_names": "List[str]"},
        records={"input": "Image", "target": "Label"},
    )


class TestDeclaration:
    def test_zero_arg_construction_declares_nothing_and_checks_clean(self) -> None:
        contract = GraphContract()
        assert contract.outputs == {} and contract.records == {} and contract.inputs == {}
        contract.check()  # nothing declared, nothing to refuse

    def test_every_output_is_required_there_is_no_second_list(self) -> None:
        """What is in `outputs` is required — an optional output would not be a contract."""
        assert classification().missing() == ["stream", "class_names"]

    def test_the_slot_vocabulary_is_a_closed_set_plus_the_item_registry(self) -> None:
        assert "Stream" in GRAPH_SLOT_KINDS and "List[str]" in GRAPH_SLOT_KINDS
        with pytest.raises(ContractError, match=r"output 'x' declares the type 'Widget', which is neither"):
            GraphContract(outputs={"x": "Widget"}).check()

    def test_an_item_type_is_a_legal_slot_kind(self) -> None:
        GraphContract(outputs={"picture": "Image"}, delivered={"picture": Image(np.zeros((2, 2)))}).check()


class TestCheck:
    def test_a_missing_output_is_refused_by_name_and_lists_what_the_graph_delivers(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records())}
        with pytest.raises(ContractError) as refusal:
            contract.check()
        assert str(refusal.value) == (
            "classification source: 'class_names' is not delivered — wire it in the graph "
            "(this graph delivers: stream, class_names)"
        )

    def test_a_stream_without_a_source_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(), "class_names": ["cat", "dog"]}
        with pytest.raises(ContractError, match=r"the Stream delivered as 'stream' has no source — connect a Source"):
            contract.check()

    def test_a_source_where_a_stream_is_expected_is_refused_with_the_fix(self) -> None:
        contract = classification()
        contract.delivered = {"stream": _records(), "class_names": ["cat", "dog"]}
        with pytest.raises(ContractError, match=r"'stream' is a list, not a Stream — put a Stream between them"):
            contract.check()

    def test_an_empty_vocabulary_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "class_names": []}
        with pytest.raises(ContractError, match=r"'class_names' resolved to \[\] — wire a class_names output"):
            contract.check()

    def test_a_vocabulary_that_is_not_a_list_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "class_names": "cat,dog"}
        with pytest.raises(ContractError, match=r"'class_names' is a str, not a list of names"):
            contract.check()

    def test_a_satisfied_contract_checks_clean(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "class_names": ["cat", "dog"]}
        contract.check()
        assert contract.missing() == []


class TestWhatItDelivers:
    def test_the_stream_carries_the_record_contract_as_its_last_op(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "class_names": ["cat", "dog"]}
        stream = contract.stream()
        assert isinstance(stream.ops[-1], RecordContract)
        assert stream.ops[-1].fields == {"input": "Image", "target": "Label"}
        assert stream.ops[-1].name == "classification source"
        assert len(list(stream)) == 2  # both records pass

    def test_the_delivered_stream_object_is_not_mutated(self) -> None:
        delivered = Stream(source=_records(), ops=[])
        contract = classification()
        contract.delivered = {"stream": delivered, "class_names": ["cat", "dog"]}
        contract.stream()
        assert delivered.ops == []

    def test_a_record_violating_the_contract_is_refused_where_the_stream_is_read(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=[{"note": "no image, no label"}]), "class_names": ["a"]}
        with pytest.raises(ContractError) as refusal:
            list(contract.stream())
        assert str(refusal.value) == (
            "classification source: record #0 has no entry 'input' (expected Image); present: note[str]"
        )

    def test_a_contract_without_records_appends_nothing(self) -> None:
        contract = GraphContract(outputs={"stream": "Stream"}, delivered={"stream": Stream(source=_records())})
        assert contract.stream().ops == []

    def test_class_names_are_strings_in_the_delivered_order(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "class_names": [0, 1, 2]}
        assert contract.class_names == ["0", "1", "2"]

    def test_asking_for_a_stream_the_graph_does_not_deliver_is_refused(self) -> None:
        with pytest.raises(ContractError, match=r"delivers no stream"):
            GraphContract(name="sink", outputs={"sink": "Sink"}).stream()


class TestHostInputs:
    def test_inputs_are_declared_beside_outputs(self) -> None:
        contract = GraphContract(inputs={"files": "List[str]"}, outputs={"stream": "Stream"})
        assert contract.inputs == {"files": "List[str]"}

    def test_an_unknown_input_kind_is_refused_like_an_output(self) -> None:
        with pytest.raises(ContractError, match=re.escape("input 'files' declares the type 'Paths'")):
            GraphContract(inputs={"files": "Paths"}).check()
