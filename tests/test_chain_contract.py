"""``check_chain``: an op DECLARES what it consumes, produces and reports, and a chain is
checked before it runs.

``RecordContract`` states what a record carries at a graph BOUNDARY — you place a node and it
asserts. That answers "what must reach here?" but not "does this chain hold together?", which is
the question an analysis chain of a dozen small ops actually raises: a node reading a measurement
an earlier node was supposed to write fails as an EMPTY result, not as an error, because a missing
record key just means the op returns early.

So a node may declare its interface as class attributes, and ``check_chain`` reads them off the
op list once, before the first record::

    class MeasureSymbolClock:
        consumes = {"signal": "Signal", "inst_freq": "InstFreq"}
        produces = {"symbol_clock": "SymbolClock"}
        reports  = "clock"
        flags    = ()

Declaring is OPT-IN — an op that declares nothing is checked for nothing, so every chain that
existed before this keeps working untouched.
"""

from typing import Dict, Optional

import pytest

from recordstream.items import Record
from recordstream.ops.contract import ChainContractError, check_chain


class _Node:
    """A declaring op — the shape the analysis nodes take."""

    consumes: Dict[str, str] = {}
    produces: Dict[str, str] = {}
    reports: str = ""
    flags: tuple = ()

    def __init__(self, requires: str = "") -> None:
        self.requires = requires

    def __call__(self, record: Record) -> Optional[Record]:
        return record


class Windowed(_Node):
    consumes = {"signal": "Signal"}
    produces = {"signal": "Signal"}


class Measure(_Node):
    consumes = {"signal": "Signal"}
    produces = {"clock": "Signal"}
    reports = "clock"


class Classify(_Node):
    consumes = {"clock": "Signal"}
    reports = "classify"
    flags = ("ble", "bredr")


class DecodeBle(_Node):
    consumes = {"clock": "Signal"}
    reports = "ble"


class Opaque:
    """An op that declares nothing — every op written before this mechanism existed."""

    def __call__(self, record: Record) -> Optional[Record]:
        return record


class TestAWellFormedChainPasses:
    def test_a_chain_whose_needs_are_all_met_is_accepted(self) -> None:
        check_chain([Windowed(), Measure(), Classify()], provided={"signal"})

    def test_a_gate_naming_a_flag_an_earlier_node_raises_is_accepted(self) -> None:
        check_chain([Measure(), Classify(), DecodeBle(requires="ble")], provided={"signal"})

    def test_an_undeclared_chain_is_checked_for_nothing(self) -> None:
        """Opt-in: every graph that predates the mechanism keeps working."""
        check_chain([Opaque(), Opaque()])

    def test_a_declaring_node_after_an_undeclared_one_still_checks_what_it_can(self) -> None:
        check_chain([Opaque(), Measure(), Classify()], provided={"signal"})


class TestAnUnsatisfiableChainIsRefused:
    def test_a_need_nothing_produces_names_the_node_and_the_key(self) -> None:
        with pytest.raises(ChainContractError) as exc:
            check_chain([Classify()], provided={"signal"})
        message = str(exc.value)
        assert "Classify" in message and "clock" in message

    def test_the_message_says_where_the_chain_stands(self) -> None:
        """A located refusal: the reader must not have to guess which node position broke."""
        with pytest.raises(ChainContractError) as exc:
            check_chain([Measure(), Classify()], provided=set(), where="view_bte.yaml")
        message = str(exc.value)
        assert "view_bte.yaml" in message
        assert "signal" in message, "the FIRST unmet need is the one reported"

    def test_a_producer_further_down_the_chain_does_not_count(self) -> None:
        """Order matters — a need must be met by something EARLIER, not later."""
        with pytest.raises(ChainContractError):
            check_chain([Classify(), Measure()], provided={"signal"})


class TestFlags:
    def test_a_gate_on_a_flag_nobody_raises_is_refused(self) -> None:
        with pytest.raises(ChainContractError) as exc:
            check_chain([Measure(), Classify(), DecodeBle(requires="nosuch")], provided={"signal"})
        message = str(exc.value)
        assert "nosuch" in message
        assert "ble" in message and "bredr" in message, "the message lists the flags that exist"

    def test_a_gate_on_a_flag_raised_later_is_refused(self) -> None:
        with pytest.raises(ChainContractError):
            check_chain([Measure(), DecodeBle(requires="ble"), Classify()], provided={"signal"})

    def test_two_nodes_raising_one_flag_is_ambiguous(self) -> None:
        """A flag must have exactly ONE producer — that is what makes 'which node decided this?'
        answerable, for the report and for a visual editor drawing the gate as a wire."""

        class Twin(_Node):
            flags = ("ble",)

        with pytest.raises(ChainContractError) as exc:
            check_chain([Measure(), Classify(), Twin()], provided={"signal"})
        message = str(exc.value)
        assert "ble" in message and "Classify" in message and "Twin" in message

    def test_the_flag_producer_map_is_total(self) -> None:
        """The property a visual fork conversion needs: every gate resolves to exactly one
        producing node, so the wire to draw is derivable from the chain alone."""
        from recordstream.ops.contract import flag_producers

        ops = [Measure(), Classify(), DecodeBle(requires="ble")]
        producers = flag_producers(ops)
        assert producers == {"ble": 1, "bredr": 1}, "index of the node that raises each flag"
        gated = [(index, op.requires) for index, op in enumerate(ops) if getattr(op, "requires", "")]
        assert all(requires in producers for _, requires in gated)


class TestReports:
    def test_two_nodes_reporting_under_one_name_is_ambiguous(self) -> None:
        """The report is keyed by the reporting name, so a duplicate would overwrite a finding."""

        class Twin(_Node):
            reports = "clock"

        with pytest.raises(ChainContractError) as exc:
            check_chain([Measure(), Twin()], provided={"signal"})
        assert "clock" in str(exc.value)

    def test_a_transform_reports_nothing_and_may_repeat(self) -> None:
        """`reports = ""` marks a transform, not an analysis — several may sit in one chain."""
        check_chain([Windowed(), Windowed()], provided={"signal"})
