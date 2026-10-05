"""GraphContract — what ONE graph must deliver, declared as data and checked in one place.

A consuming workspace opens several graphs (a source graph, a files graph, a sink graph, …) and
each must deliver something different: a stream of records carrying an image and a label, a
sink, a vocabulary. One class states that per graph — `outputs` (every entry required),
`records` (what each record of a delivered stream carries) — and `check()` is the single
verification the editor and the workspace both call, so a graph refused in one is refused in
the other with the same words.

The class vocabulary slot is named `classes`, never `class_names`: confluid broadcasts a key
under `delivered:` into every sibling whose constructor takes a parameter of that name, and
`class_names` IS a parameter of `Stream`. Measured 2026-09-27 and pinned below: a producer wired
under `class_names:` fails to LOAD; a literal list under it is silently pushed into the Stream.
"""

import re
from typing import List

import numpy as np
import pytest
from confluid import ConstructionError, input_specs, load

from recordstream import Image, Label, Record, Stream
from recordstream.ops.contract import (
    CLASS_VOCABULARY_SLOT,
    GRAPH_SLOT_KINDS,
    ClassNamesOutput,
    ContractError,
    GraphContract,
    RecordContract,
)
from recordstream.sources.files import FilesSource


def _records() -> List[Record]:
    return [
        {"input": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "target": Label(value=1)},
        {"input": Image(np.zeros((4, 4, 3), dtype=np.uint8)), "target": Label(value=0)},
    ]


def classification() -> GraphContract:
    return GraphContract(
        name="classification source",
        outputs={"stream": "Stream", "classes": "List[str]"},
        records={"input": "Image", "target": "Label"},
    )


class _NamesByMethod:
    """A producer that declares its vocabulary as a zero-arg METHOD rather than a property."""

    def class_names(self) -> List[str]:
        return ["bird", "fish"]


class TestDeclaration:
    def test_zero_arg_construction_declares_nothing_and_checks_clean(self) -> None:
        contract = GraphContract()
        assert contract.outputs == {} and contract.records == {} and contract.inputs == {}
        contract.check()  # nothing declared, nothing to refuse

    def test_every_output_is_required_there_is_no_second_list(self) -> None:
        """What is in `outputs` is required — an optional output would not be a contract."""
        assert classification().missing() == ["stream", "classes"]

    def test_the_slot_vocabulary_is_a_closed_set_plus_the_item_registry(self) -> None:
        assert "Stream" in GRAPH_SLOT_KINDS and "List[str]" in GRAPH_SLOT_KINDS
        with pytest.raises(ContractError, match=r"output 'x' declares the type 'Widget', which is neither"):
            GraphContract(outputs={"x": "Widget"}).check()

    def test_an_item_type_is_a_legal_slot_kind(self) -> None:
        GraphContract(outputs={"picture": "Image"}, delivered={"picture": Image(np.zeros((2, 2)))}).check()

    def test_the_class_vocabulary_slot_is_named_classes(self) -> None:
        assert CLASS_VOCABULARY_SLOT == "classes"


class TestCheck:
    def test_a_missing_output_is_refused_by_name_and_lists_what_the_graph_delivers(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records())}
        with pytest.raises(ContractError) as refusal:
            contract.check()
        assert str(refusal.value) == (
            "classification source: 'classes' is not delivered — wire it in the graph "
            "(this graph delivers: stream, classes)"
        )

    def test_a_stream_without_a_source_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(), "classes": ["cat", "dog"]}
        with pytest.raises(ContractError, match=r"the Stream delivered as 'stream' has no source — connect a Source"):
            contract.check()

    def test_a_source_where_a_stream_is_expected_is_refused_with_the_fix(self) -> None:
        contract = classification()
        contract.delivered = {"stream": _records(), "classes": ["cat", "dog"]}
        with pytest.raises(ContractError, match=r"'stream' is a list, not a Stream — put a Stream between them"):
            contract.check()

    def test_an_empty_vocabulary_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": []}
        with pytest.raises(ContractError, match=r"'classes' resolved to \[\] — wire a class_names output"):
            contract.check()

    def test_a_vocabulary_that_is_not_a_list_is_refused(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": "cat,dog"}
        with pytest.raises(ContractError, match=r"'classes' is a str, not a list of names"):
            contract.check()

    def test_a_satisfied_contract_checks_clean(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ["cat", "dog"]}
        contract.check()
        assert contract.missing() == []


class TestTheClassVocabulary:
    """`classes` takes a typed list OR a wired producer that declares its own `class_names`."""

    def test_a_typed_list_is_read_as_strings_in_its_order(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": [0, 1, 2]}
        contract.check()
        assert contract.class_names == ["0", "1", "2"]

    def test_a_wired_producer_is_accepted_and_its_declared_names_are_read(self) -> None:
        """The document keeps the PRODUCER — where the names came from stays visible."""
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ClassNamesOutput(names=["cat", "dog"])}
        contract.check()
        assert contract.class_names == ["cat", "dog"]

    def test_a_producer_may_declare_its_names_as_a_zero_arg_method(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": _NamesByMethod()}
        contract.check()
        assert contract.class_names == ["bird", "fish"]

    def test_a_producer_that_declares_no_class_names_is_refused_by_name(self) -> None:
        """A FilesSource has no vocabulary: the refusal names the class and both ways out."""
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": FilesSource(files=["a.png"])}
        with pytest.raises(ContractError) as refusal:
            contract.check()
        assert str(refusal.value) == (
            "classification source: 'classes' is delivered by a FilesSource, which declares no class names — "
            "wire a source that does (a HuggingFace source has one) or type the names as a list"
        )

    def test_a_producer_whose_declared_names_are_empty_is_refused_the_same_way(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ClassNamesOutput()}
        with pytest.raises(
            ContractError, match=r"'classes' is delivered by a ClassNamesOutput, which declares no class"
        ):
            contract.check()

    def test_a_graph_delivering_no_vocabulary_reads_as_an_empty_list(self) -> None:
        contract = GraphContract(outputs={"stream": "Stream"}, delivered={"stream": Stream(source=_records())})
        assert contract.class_names == []


class TestSlotNamesAreNotParameters:
    """An output slot may not share its name with a constructor parameter of any delivered object.

    The rule exists because of confluid's broadcast: a key under `delivered:` is pushed into every
    sibling whose constructor takes that name. The two measurements below are the WHY, pinned so
    a change in confluid's rule is noticed here.
    """

    PRODUCER_UNDER_CLASS_NAMES = """\
names: !class:recordstream.ops.contract.ClassNamesOutput {names: [cat, dog]}
stream: !class:recordstream.core.stream.Stream
  source: [{a: 1}]
source: !class:recordstream.ops.contract.GraphContract
  name: classification source
  outputs: {stream: Stream, class_names: "List[str]"}
  delivered:
    stream: !ref:stream
    class_names: !ref:names
"""

    def test_why_a_producer_wired_under_class_names_fails_to_load(self) -> None:
        """Measured 2026-09-27: the ClassNamesOutput lands in Stream(class_names=…) and the Stream refuses it."""
        with pytest.raises(ConstructionError, match=r"Failed to construct Stream at <unicode string>:2:9"):
            load(self.PRODUCER_UNDER_CLASS_NAMES)

    def test_why_a_literal_list_under_class_names_is_silently_pushed_into_the_stream(self) -> None:
        """The silent half: it 'worked' only because a list happens to be what Stream.class_names takes."""
        document = self.PRODUCER_UNDER_CLASS_NAMES.replace("class_names: !ref:names", "class_names: [cat, dog]")
        contract = load(document)["source"]
        assert contract.delivered["stream"].class_names == ["cat", "dog"]  # the Stream never declared them

    def test_the_same_producer_wired_under_classes_loads_and_checks_clean(self) -> None:
        document = self.PRODUCER_UNDER_CLASS_NAMES.replace("class_names", "classes")
        contract = load(document)["source"]
        contract.check()
        assert type(contract.delivered["classes"]).__name__ == "ClassNamesOutput"
        assert contract.class_names == ["cat", "dog"]
        assert contract.delivered["stream"].class_names is None  # nothing was pushed into the Stream

    def test_the_colliding_slot_is_refused_at_declaration_naming_the_parameter_and_the_fix(self) -> None:
        """CON case: `class_names` beside a delivered Stream — refused even with a well-typed list."""
        contract = GraphContract(
            name="classification source",
            outputs={"stream": "Stream", "class_names": "List[str]"},
            delivered={"stream": Stream(source=_records()), "class_names": ["cat", "dog"]},
        )
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "classification source: output 'class_names' shares its name with a parameter of Stream "
            "(delivered as 'stream'); confluid would push the slot's value into that parameter — "
            "rename the slot (the class vocabulary slot is 'classes')"
        )
        with pytest.raises(ContractError, match=r"output 'class_names' shares its name"):
            contract.check()  # check() runs the declaration check first

    def test_the_renamed_slot_passes(self) -> None:
        """PRO case: the same graph with `classes` — no parameter of a delivered object is named that."""
        contract = GraphContract(
            name="classification source",
            outputs={"stream": "Stream", "classes": "List[str]"},
            delivered={"stream": Stream(source=_records()), "classes": ["cat", "dog"]},
        )
        contract.check_declaration()
        contract.check()

    def test_the_rule_covers_every_parameter_of_every_delivered_object(self) -> None:
        """`source` is a Stream parameter too; the hint about `classes` is reserved for `class_names`."""
        contract = GraphContract(
            name="files",
            outputs={"stream": "Stream", "source": "Source"},
            delivered={"stream": Stream(source=_records()), "source": FilesSource(files=["a.png"])},
        )
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "files: output 'source' shares its name with a parameter of Stream (delivered as 'stream'); "
            "confluid would push the slot's value into that parameter — rename the slot"
        )

    def test_a_declaration_with_nothing_delivered_yet_is_not_refused(self) -> None:
        """What the editor checks before it draws: no objects yet, so no parameters to collide with."""
        GraphContract(outputs={"stream": "Stream", "class_names": "List[str]"}).check_declaration()

    def test_plain_values_have_no_parameters(self) -> None:
        """A list, a dict, a str delivered into a slot contribute no constructor to collide with."""
        GraphContract(
            outputs={"stream": "Stream", "files": "Source", "names": "List[str]", "note": "str"},
            delivered={"stream": Stream(source=_records()), "files": _records(), "names": ["a"], "note": "x"},
        ).check_declaration()


class TestWhatItDelivers:
    def test_the_stream_carries_the_record_contract_as_its_last_op(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ["cat", "dog"]}
        stream = contract.stream()
        assert isinstance(stream.ops[-1], RecordContract)
        assert stream.ops[-1].fields == {"input": "Image", "target": "Label"}
        assert stream.ops[-1].name == "classification source"
        assert len(list(stream)) == 2  # both records pass

    def test_the_delivered_stream_object_is_not_mutated(self) -> None:
        delivered = Stream(source=_records(), ops=[])
        contract = classification()
        contract.delivered = {"stream": delivered, "classes": ["cat", "dog"]}
        contract.stream()
        assert delivered.ops == []

    def test_a_record_violating_the_contract_is_refused_where_the_stream_is_read(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=[{"note": "no image, no label"}]), "classes": ["a"]}
        with pytest.raises(ContractError) as refusal:
            list(contract.stream())
        assert str(refusal.value) == (
            "classification source: record #0 has no entry 'input' (expected Image); present: note[str]"
        )

    def test_a_contract_without_records_appends_nothing(self) -> None:
        contract = GraphContract(outputs={"stream": "Stream"}, delivered={"stream": Stream(source=_records())})
        assert contract.stream().ops == []

    def test_the_stream_carries_the_contracts_vocabulary(self) -> None:
        """Typed or wired, the delivered stream answers `class_names` with what the graph delivered."""
        for classes in (["cat", "dog"], ClassNamesOutput(names=["cat", "dog"])):
            contract = classification()
            contract.delivered = {"stream": Stream(source=_records()), "classes": classes}
            assert contract.stream().class_names == ["cat", "dog"]

    def test_a_stream_keeps_its_own_vocabulary_when_the_graph_delivers_none(self) -> None:
        contract = GraphContract(
            outputs={"stream": "Stream"}, delivered={"stream": Stream(source=_records(), class_names=["x", "y"])}
        )
        assert contract.stream().class_names == ["x", "y"]

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


def _source_spelled_records() -> List[Record]:
    """Records the way a dataset source spells them — `image` and `class`, not the root's `input` / `target`."""
    return [
        {
            "image": Image(np.zeros((4, 4, 3), dtype=np.uint8)),
            "class": Label(value=1),
            "hf_split": Label(value="train"),
        },
        {
            "image": Image(np.zeros((4, 4, 3), dtype=np.uint8)),
            "class": Label(value=0),
            "hf_split": Label(value="train"),
        },
    ]


class TestARecordEntryWiredIntoTheRoot:
    """A `records` entry of the root is WIRED from the entry a source writes: `delivered: {input: image}`.

    The editor draws each declared record entry as an input of the root; a wire from a source's `image`
    output into `input` is saved as `input: image` under `delivered:` — the entry key the root reads it
    from. The delivered stream then carries the entry under the root's name, so a host reading `input`
    and `target` takes a source that writes `image` and `class` as it is, with no rename step drawn.
    """

    WIRED = """\
stream: !class:recordstream.core.stream.Stream
  source: [{a: 1}]
contract: !class:recordstream.ops.contract.GraphContract
  name: classification source
  outputs: {stream: Stream, classes: "List[str]"}
  records: {input: Image, target: Label}
  delivered:
    stream: !ref:stream
    classes: [cat, dog]
    input: image
    target: class
"""

    def test_wired_entries_leave_the_stream_under_the_roots_names(self) -> None:
        contract = classification()
        contract.delivered = {
            "stream": Stream(source=_source_spelled_records()),
            "classes": ["cat", "dog"],
            "input": "image",
            "target": "class",
        }
        contract.check()
        records = list(contract.stream())
        assert [list(record) for record in records] == [["input", "target", "hf_split"]] * 2
        assert contract.wired_entries() == {"input": "image", "target": "class"}
        assert isinstance(contract.stream().ops[-1], RecordContract)  # the boundary check stays the last op
        assert contract.stream().ops[-1].fields == {"input": "Image", "target": "Label"}

    def test_the_saved_spelling_loads_as_plain_entry_keys_and_reaches_no_sibling(self) -> None:
        """confluid pushes a `delivered:` key into a sibling that takes a parameter of that name — the Stream
        takes neither `input` nor `target`, so the keys stay where they are written."""
        contract = load(self.WIRED)["contract"]
        assert contract.delivered["input"] == "image" and contract.delivered["target"] == "class"
        contract.check()

    def test_an_unwired_entry_is_read_under_its_own_name(self) -> None:
        """PRO, unchanged: a chain that already writes `input` / `target` needs no wire."""
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ["cat", "dog"]}
        assert contract.wired_entries() == {}
        assert [type(op).__name__ for op in contract.stream().ops] == ["RecordContract"]  # no step added
        assert len(list(contract.stream())) == 2

    def test_a_wired_entry_the_records_lack_is_refused_naming_both(self) -> None:
        """CON: wired to `image`, but the source writes `picture`."""
        contract = classification()
        contract.delivered = {
            "stream": Stream(source=[{"picture": Image(np.zeros((2, 2))), "class": Label(value=0)}]),
            "classes": ["a"],
            "input": "image",
            "target": "class",
        }
        with pytest.raises(ContractError) as refusal:
            list(contract.stream())
        assert str(refusal.value) == (
            "classification source: record #0 has no entry 'image' (wired into 'input', expected Image); "
            "present: picture[Image], class[Label]"
        )

    def test_a_wired_entry_of_the_wrong_type_names_both_keys(self) -> None:
        contract = classification()
        contract.delivered = {
            "stream": Stream(source=[{"class": Label(value=1)}]),
            "classes": ["a"],
            "input": "class",
            "target": "class",
        }
        with pytest.raises(ContractError) as refusal:
            list(contract.stream())
        assert str(refusal.value) == (
            "classification source: record #0 entry 'class' (wired into 'input') is a Label, expected Image; "
            "present: class[Label]"
        )

    def test_the_wired_entry_replaces_one_already_named_like_the_roots(self) -> None:
        image = Image(np.zeros((2, 2)))
        contract = classification()
        contract.delivered = {
            "stream": Stream(source=[{"input": image, "target": Label(value="stale"), "class": Label(value="digit")}]),
            "classes": ["a"],
            "target": "class",
        }
        (record,) = list(contract.stream())
        assert record == {"input": image, "target": Label(value="digit")}

    def test_two_entries_may_swap_keys(self) -> None:
        contract = GraphContract(
            name="g",
            outputs={"stream": "Stream"},
            records={"a": "*", "b": "*"},
            delivered={"stream": Stream(source=[{"a": 1, "b": 2}]), "a": "b", "b": "a"},
        )
        assert list(contract.stream()) == [{"b": 1, "a": 2}]

    def test_the_record_contract_every_pane_carries_is_unchanged(self) -> None:
        """The move lives in a step the root builds — RecordContract keeps its two settings and its pass-through."""
        assert [spec["name"] for spec in input_specs(RecordContract)] == ["fields", "name"]
        record = {"input": Image(np.zeros((2, 2)))}
        assert RecordContract(fields={"input": "Image"})(record) is record

    def test_a_wired_entry_must_be_named_by_a_string(self) -> None:
        contract = classification()
        contract.delivered = {"stream": Stream(source=_records()), "classes": ["a"], "input": 3}
        with pytest.raises(ContractError) as refusal:
            contract.check()
        assert str(refusal.value) == (
            "classification source: the record entry 'input' is wired to 3 — it names the entry it is read from, "
            "a word like 'image'"
        )

    def test_a_record_entry_named_like_an_output_is_refused(self) -> None:
        contract = GraphContract(name="g", outputs={"stream": "Stream"}, records={"stream": "Image"})
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "g: 'stream' is declared both as an output and as a record entry — they are wired into the same place, "
            "so give them different names"
        )

    def test_a_record_entry_named_like_a_delivered_objects_parameter_is_refused(self) -> None:
        """The broadcast rule bites a wired record entry exactly as it bites an output slot."""
        contract = GraphContract(
            name="g",
            outputs={"stream": "Stream"},
            records={"source": "Image"},
            delivered={"stream": Stream(source=_records()), "source": "image"},
        )
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "g: record entry 'source' shares its name with a parameter of Stream (delivered as 'stream'); "
            "confluid would push the slot's value into that parameter — rename the slot"
        )


class TestAnOptionalSlot:
    """``optional`` names a slot a graph MAY deliver: the host has its own answer when it is left unwired.

    A graph of dropped files may deliver its own class list — or leave it to the host, which then uses the
    vocabulary of the workspace's source graph. Every entry of ``outputs`` stays REQUIRED; an optional slot is
    declared apart, so a reader sees at once which slots the host can do without.
    """

    def files(self, **delivered: object) -> GraphContract:
        return GraphContract(
            name="classification files",
            inputs={"files": "List[str]"},
            outputs={"stream": "Stream"},
            optional={"classes": "List[str]"},
            records={"input": "Image"},
            delivered={"stream": Stream(source=_records()), **delivered},
        )

    def test_left_unwired_it_is_not_missing_and_the_graph_checks_clean(self) -> None:
        contract = self.files()
        assert contract.missing() == []
        contract.check()
        assert contract.class_names == []

    def test_wired_it_is_read_like_an_output(self) -> None:
        for classes in (["cat", "dog"], ClassNamesOutput(names=["cat", "dog"])):
            contract = self.files(classes=classes)
            contract.check()
            assert contract.class_names == ["cat", "dog"]
            assert contract.stream().class_names == ["cat", "dog"]

    def test_wired_wrongly_it_is_refused_like_an_output(self) -> None:
        """CON: an empty list typed into it is a mistake, not "left to the host"."""
        with pytest.raises(ContractError) as refusal:
            self.files(classes=[]).check()
        assert str(refusal.value) == (
            "classification files: 'classes' resolved to [] — wire a class_names output (a HuggingFace source has "
            "one) or a Value list into it"
        )

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ContractError, match=re.escape("optional 'classes' declares the type 'Names'")):
            GraphContract(name="g", optional={"classes": "Names"}).check_declaration()

    def test_a_slot_declared_both_required_and_optional_is_refused(self) -> None:
        contract = GraphContract(name="g", outputs={"classes": "List[str]"}, optional={"classes": "List[str]"})
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "g: 'classes' is declared both as an output and as an optional slot — give it one place"
        )

    def test_an_optional_slot_named_like_a_delivered_objects_parameter_is_refused(self) -> None:
        contract = GraphContract(
            name="g",
            outputs={"stream": "Stream"},
            optional={"source": "Source"},
            delivered={"stream": Stream(source=_records())},
        )
        with pytest.raises(ContractError) as refusal:
            contract.check_declaration()
        assert str(refusal.value) == (
            "g: optional 'source' shares its name with a parameter of Stream (delivered as 'stream'); "
            "confluid would push the slot's value into that parameter — rename the slot"
        )

    def test_the_saved_spelling_loads(self) -> None:
        document = """\
class_list: !class:recordstream.ops.contract.ClassNamesOutput {names: [cat, dog]}
stream: !class:recordstream.core.stream.Stream
  source: [{a: 1}]
contract: !class:recordstream.ops.contract.GraphContract
  name: classification files
  outputs: {stream: Stream}
  optional: {classes: "List[str]"}
  delivered:
    stream: !ref:stream
    classes: !ref:class_list
"""
        contract = load(document)["contract"]
        contract.check()
        assert contract.optional == {"classes": "List[str]"}
        assert contract.class_names == ["cat", "dog"]


class TestAWiredStreamSavesAndLoadsBack:
    """The stream a root hands on is SAVED by a host (a workspace file dumps the stream it lists), so the step that
    reads a wired entry must round-trip through confluid like every other op.

    Measured 2026-10-05 before the fix: ``dump: a live recordstream.ops.contract._WiredEntries has no reconstructible
    document spelling`` on save, then ``Failed to construct _WiredEntries … missing 3 required positional arguments``
    when the saved file was opened again.
    """

    def test_the_dumped_stream_loads_and_yields_the_same_records(self) -> None:
        from confluid import dump

        contract = classification()
        contract.delivered = {
            "stream": Stream(source=[{"image": "pixels", "class": "label", "hf_split": "train"}]),
            "classes": ["cat", "dog"],
            "input": "image",
            "target": "class",
        }
        contract.records = {"input": "*", "target": "*"}
        text = dump(contract.stream())
        assert "WiredEntries" in text and "_WiredEntries" not in text
        reloaded = load(text)
        assert (
            list(reloaded) == list(contract.stream()) == [{"input": "pixels", "target": "label", "hf_split": "train"}]
        )

    def test_the_step_is_no_node_of_the_palette(self) -> None:
        """Configurable so it round-trips, but no category: no editor offers it as a node to place."""
        from confluid import marks

        from recordstream.ops.contract import WiredEntries

        assert marks(WiredEntries).category is None
        assert list(WiredEntries()({"a": 1}).items()) == [("a", 1)]  # zero-arg: moves nothing
