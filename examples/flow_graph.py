"""Flow-document graphs: author a named-step graph, run it on BOTH engines, convert both ways.

Demonstrates the graph execution model:
- a ``flow`` mapping (step-name -> op) with fan-out (two readers of one step), fan-in
  (``target_from``), and a per-sample parameter bind;
- native execution on :class:`sampleflux.FlowGraph`;
- lowering to a flat context-ops list (:func:`sampleflux.to_ops`) executed by the plain
  serial :class:`sampleflux.Flux` engine — with identical results;
- lifting a flat list back into a flow (:func:`sampleflux.from_ops`).

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

from confluid import configurable

from sampleflux import FlowGraph, Flux, Sample, from_ops, to_ops
from sampleflux.ops.swap import SwapInputTargetOp


@configurable
class AddOp:
    """Add a constant to the input.

    Args:
        amount: Value added to ``sample.input``.
    """

    def __init__(self, amount: float = 1.0) -> None:
        self.amount = amount

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input + self.amount)


@configurable
class ScaleOp:
    """Multiply the input by a factor.

    Args:
        factor: Multiplier applied to ``sample.input``.
    """

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def __call__(self, sample: Sample) -> Sample:
        return sample._replace(input=sample.input * self.factor)


def build_flow() -> dict:
    """The graph: fork `a`; branch A swaps v+1 into target; branch B computes (v+1)*2 + bind."""
    return {
        "a": AddOp(amount=1.0),  # input: the source sample
        "branch_a": {"op": SwapInputTargetOp(), "from": "a"},  # A: park v+1 in target
        "branch_b": {"op": ScaleOp(factor=2.0), "from": "a"},  # B: (v+1)*2   (2nd read of a = fan-out)
        "shifted": {"op": AddOp(), "from": "branch_b", "bind": {"amount": "a"}},  # per-sample param
        "out": {"from": "shifted", "target_from": "branch_a"},  # fan-in
    }


def main() -> None:
    source = [Sample(input=float(i), target=i, metadata={"idx": i}) for i in range(4)]

    # 1. Native FlowGraph execution
    native = [(s.input, s.target) for s in FlowGraph(source=source, flow=build_flow())]
    print(f"FlowGraph (native): {native}")

    # 2. Lower to the flat context-ops list -> serial Flux engine
    ops = to_ops(build_flow())
    print(f"Lowered ops: {[type(o).__name__ for o in ops]}")
    serial = [(s.input, s.target) for s in Flux(source=source, ops=ops)]
    print(f"Flux (lowered):     {serial}")
    assert native == serial, "engine parity is a hard contract"

    # 3. Lift the flat list back into a flow document
    lifted, outputs = from_ops(to_ops(build_flow()))
    relifted = [(s.input, s.target) for s in FlowGraph(source=source, flow=lifted, outputs=outputs)]
    assert relifted == native
    print(f"Lifted flow steps:  {list(lifted)} (outputs={outputs!r})")
    print("flow -> ops -> flow round-trip: parity holds ✓")


if __name__ == "__main__":
    main()
