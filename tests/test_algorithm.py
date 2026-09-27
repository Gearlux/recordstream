"""An ``Algorithm`` declares its settings, inputs and outputs; the recordstream op is derived.

The author writes three kinds of slot and one method::

    class BackgroundLevel(Algorithm):
        percentile: float = Param(default=25.0, doc="...")   # a setting (constructor argument)
        image: Image = Input(doc="...")                       # read from the record entry "image"
        background: float = Output(doc="...")                 # written to the record entry "background"

        def compute(self):
            return {"background": ...}

and never sees a record: reading the inputs out of one, checking them, and writing the outputs back
is the base class's job, done once for every algorithm. Everything else a tool needs — the
constructor, the settings schema, the output sockets, the chain checker's ``consumes`` /
``produces`` — is derived from the same three declarations.
"""

import inspect
import os
import pickle
import re
from typing import Any, Dict, List, Literal, Optional

import confluid
import numpy as np
import pytest
from annotated_types import Interval
from confluid import configurable, output_specs, parse_param_docs, to_pydantic
from typing_extensions import Annotated

import recordstream
from recordstream import Boxes, FlowGraph, Image, Mask, Stream
from recordstream.algorithm import Algorithm, Input, Output, Param, algorithm_spec
from recordstream.ops.contract import ANY_TYPE, ChainContractError, check_chain
from recordstream.ops.numpy import Threshold

Percent = Annotated[float, Interval(ge=0.0, le=100.0)]


@configurable(category="op", group="test")
class BackgroundLevel(Algorithm):
    """Estimate an image's background level: a low percentile of its per-row medians."""

    percentile: Percent = Param(default=25.0, doc="Percentile rank in [0, 100] across the per-row medians.")
    image: Image = Input(doc="The image to read.")
    background: float = Output(doc="The background level, in the image's own units.")

    def compute(self) -> Dict[str, Any]:
        per_row = np.median(np.asarray(self.image, dtype=np.float64), axis=1)
        return {"background": float(np.percentile(per_row, self.percentile))}


@configurable(category="op", group="test")
class KeepCentredBoxes(Algorithm):
    """Keep the boxes whose centre pixel is ON in a mask."""

    mask: Mask = Input(doc="The mask to test against.")
    boxes: Boxes = Input(doc="The boxes to filter.")
    kept: Boxes = Output(doc="The boxes whose centre is ON.")

    def compute(self) -> Dict[str, Any]:
        return {"kept": _centred(self.mask, self.boxes)}


@configurable(category="op", group="test")
class DropOffMaskBoxes(Algorithm):
    """The same filter, written back over the boxes it read — ``replaces`` declared once, in the class."""

    mask: Mask = Input(doc="The mask to test against.")
    boxes: Boxes = Input(doc="The boxes to filter.")
    kept: Boxes = Output(replaces="boxes", doc="The boxes whose centre is ON.")

    def compute(self) -> Dict[str, Any]:
        return {"kept": _centred(self.mask, self.boxes)}


class WhereWasIRun(Algorithm):
    """Report the process that computed the record — proves a record crossed into a worker."""

    tag: int = Input(doc="Any value; the record needs one input.")
    pid: int = Output(doc="The computing process's id.")

    def compute(self) -> Dict[str, Any]:
        return {"pid": os.getpid()}


class ScaleWithOptionalMask(Algorithm):
    """Multiply an image; where a mask is given, only under the mask."""

    factor: float = Param(default=2.0, doc="The multiplier.")
    image: Image = Input(doc="The image.")
    mask: Optional[Mask] = Input(default=None, doc="Where to apply it; absent = everywhere.")
    scaled: np.ndarray = Output(doc="The scaled pixels.")

    def compute(self) -> Dict[str, Any]:
        pixels = np.asarray(self.image, dtype=np.float64)
        where = np.ones(pixels.shape, bool) if self.mask is None else np.asarray(self.mask, bool)
        return {"scaled": np.where(where, pixels * self.factor, pixels)}


def _centred(mask: Mask, boxes: Boxes) -> Boxes:
    keep = [b for b in boxes.boxes if mask[int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)]]
    return Boxes(boxes=keep, canvas=boxes.canvas)


def _image() -> Image:
    rng = np.random.default_rng(0)
    pixels = 10.0 + rng.standard_normal((32, 48))
    pixels[4:8, :] += 50.0  # a bright band the low percentile must ignore
    return Image(pixels, layout="HWC")


def _mask_and_boxes() -> Dict[str, Any]:
    mask = Mask(np.zeros((10, 10), bool))
    mask[0:5, 0:5] = True
    return {"mask": mask, "boxes": Boxes(boxes=[[0, 0, 4, 4], [6, 6, 9, 9]], canvas=(10, 10))}


def _expected_background(image: Image, percentile: float) -> float:
    return float(np.percentile(np.median(np.asarray(image, dtype=np.float64), axis=1), percentile))


# --------------------------------------------------------------------------------------------
# standalone: no record anywhere
# --------------------------------------------------------------------------------------------


class TestStandalone:
    def test_run_returns_every_output_by_name(self) -> None:
        image = _image()
        assert BackgroundLevel(percentile=30.0).run(image=image) == {"background": _expected_background(image, 30.0)}

    def test_the_output_is_readable_after_a_run_and_none_before(self) -> None:
        algo = BackgroundLevel()
        assert algo.background is None
        algo.run(image=_image())
        assert algo.background == _expected_background(_image(), 25.0)

    def test_run_refuses_an_input_that_is_not_declared(self) -> None:
        with pytest.raises(
            TypeError, match=r"BackgroundLevel\.run\(\): unknown input\(s\) \['picture'\]; inputs are \['image'\]"
        ):
            BackgroundLevel().run(picture=_image())

    def test_run_refuses_a_missing_required_input(self) -> None:
        with pytest.raises(TypeError, match=r"BackgroundLevel\.run\(\): missing input 'image'"):
            BackgroundLevel().run()

    def test_an_optional_input_left_out_gets_its_default(self) -> None:
        image = Image(np.ones((2, 2)), layout="HWC")
        assert ScaleWithOptionalMask().run(image=image)["scaled"].tolist() == [[2.0, 2.0], [2.0, 2.0]]
        mask = Mask(np.array([[True, False], [False, False]]))
        assert ScaleWithOptionalMask().run(image=image, mask=mask)["scaled"].tolist() == [[2.0, 1.0], [1.0, 1.0]]

    def test_the_shared_instance_keeps_no_input_values(self) -> None:
        """Inputs live on a per-run copy, so the configured object never holds a record's arrays."""
        algo = BackgroundLevel()
        algo.run(image=_image())
        algo({"image": _image()})
        assert "image" not in vars(algo)
        with pytest.raises(AttributeError):
            algo.image  # noqa: B018 — the read IS the assertion


class TestCompute:
    def test_forgetting_an_output_is_refused(self) -> None:
        class Two(Algorithm):
            x: float = Input()
            a: float = Output()
            b: float = Output()

            def compute(self) -> Dict[str, Any]:
                return {"a": self.x}

        with pytest.raises(TypeError, match=r"Two\.compute\(\) returned \['a'\]; it must return exactly \['a', 'b'\]"):
            Two().run(x=1.0)

    def test_an_undeclared_output_is_refused(self) -> None:
        class One(Algorithm):
            x: float = Input()
            a: float = Output()

            def compute(self) -> Dict[str, Any]:
                return {"a": self.x, "extra": 1}

        with pytest.raises(
            TypeError, match=r"One\.compute\(\) returned \['a', 'extra'\]; it must return exactly \['a'\]"
        ):
            One().run(x=1.0)

    def test_returning_something_other_than_a_mapping_is_refused(self) -> None:
        class Bare(Algorithm):
            x: float = Input()
            a: float = Output()

            def compute(self) -> Any:
                return self.x

        with pytest.raises(TypeError, match=r"Bare\.compute\(\) returned float; it must return exactly \['a'\]"):
            Bare().run(x=1.0)

    def test_an_algorithm_without_compute_says_so(self) -> None:
        class Unwritten(Algorithm):
            a: float = Output()

        with pytest.raises(NotImplementedError, match=r"Unwritten must implement compute\(\)"):
            Unwritten().run()


# --------------------------------------------------------------------------------------------
# settings: the generated constructor
# --------------------------------------------------------------------------------------------


class TestSettings:
    def test_params_are_keyword_constructor_arguments_with_their_defaults(self) -> None:
        assert BackgroundLevel().percentile == 25.0
        assert BackgroundLevel(percentile=30.0).percentile == 30.0
        assert list(inspect.signature(BackgroundLevel).parameters) == ["percentile", "keys"]
        assert all(
            p.kind is inspect.Parameter.KEYWORD_ONLY for p in inspect.signature(BackgroundLevel).parameters.values()
        )

    def test_a_param_without_default_is_required(self) -> None:
        class NeedsPath(Algorithm):
            path: str = Param(doc="Where to read from.")
            out: str = Output()

            def compute(self) -> Dict[str, Any]:
                return {"out": self.path}

        assert NeedsPath(path="x").run() == {"out": "x"}
        with pytest.raises(TypeError, match=r"NeedsPath\(\) missing parameter 'path'"):
            NeedsPath()  # type: ignore[call-arg]

    def test_a_mutable_default_is_not_shared_between_instances(self) -> None:
        class Collects(Algorithm):
            names: List[str] = Param(default=[], doc="Names.")
            out: int = Output()

            def compute(self) -> Dict[str, Any]:
                return {"out": len(self.names)}

        first, second = Collects(), Collects()
        first.names.append("a")
        assert second.names == []

    def test_an_unknown_parameter_is_refused(self) -> None:
        class Plain(Algorithm):
            level: float = Param(default=1.0)
            out: float = Output()

            def compute(self) -> Dict[str, Any]:
                return {"out": self.level}

        with pytest.raises(TypeError, match=r"Plain\(\) got unknown parameter\(s\) \['levle'\]"):
            Plain(levle=2.0)  # type: ignore[call-arg]

    def test_an_unknown_parameter_is_refused_by_confluid_too(self) -> None:
        with pytest.raises(ValueError, match="percentil"):
            BackgroundLevel(percentil=30.0)  # type: ignore[call-arg]

    def test_an_input_is_not_a_constructor_argument(self) -> None:
        with pytest.raises(ValueError, match="image"):
            BackgroundLevel(image=_image())  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------------
# as a recordstream op
# --------------------------------------------------------------------------------------------


class TestAsAnOp:
    def test_it_reads_inputs_by_name_and_writes_outputs_by_name(self) -> None:
        image = _image()
        record = {"image": image, "samplerate": 1.0}
        out = BackgroundLevel(percentile=30.0)(record)
        assert out["background"] == _expected_background(image, 30.0)
        assert out["image"] is image and out["samplerate"] == 1.0
        assert "background" not in record  # the incoming record is not mutated

    def test_the_op_and_run_give_the_same_answer(self) -> None:
        image = _image()
        assert BackgroundLevel()({"image": image})["background"] == BackgroundLevel().run(image=image)["background"]

    def test_keys_remaps_an_input_and_an_output(self) -> None:
        image = _image()
        out = BackgroundLevel(keys={"image": "photo", "background": "level"})({"photo": image})
        assert set(out) == {"photo", "level"}
        assert out["level"] == _expected_background(image, 25.0)

    def test_two_inputs_one_output(self) -> None:
        out = KeepCentredBoxes()(_mask_and_boxes())
        assert out["kept"].boxes == [[0, 0, 4, 4]]
        assert out["boxes"].boxes == [[0, 0, 4, 4], [6, 6, 9, 9]]

    def test_an_optional_input_absent_from_the_record_gets_its_default(self) -> None:
        out = ScaleWithOptionalMask()({"image": Image(np.ones((2, 2)), layout="HWC")})
        assert out["scaled"].tolist() == [[2.0, 2.0], [2.0, 2.0]]

    def test_a_value_that_is_not_an_item_type_is_passed_as_is(self) -> None:
        """Only a registered ITEM type is checked; a float input accepts a numpy float32."""

        class Double(Algorithm):
            x: float = Input()
            y: float = Output()

            def compute(self) -> Dict[str, Any]:
                return {"y": self.x * 2}

        assert Double()({"x": np.float32(1.5)})["y"] == 3.0


class TestReplacingOutputs:
    def test_a_replacing_output_writes_back_where_its_input_was_read(self) -> None:
        out = DropOffMaskBoxes()(_mask_and_boxes())
        assert set(out) == {"mask", "boxes"}
        assert out["boxes"].boxes == [[0, 0, 4, 4]]

    def test_it_follows_the_input_when_keys_moves_the_input(self) -> None:
        record = _mask_and_boxes()
        record = {"mask": record["mask"], "predictions": record["boxes"]}
        out = DropOffMaskBoxes(keys={"boxes": "predictions"})(record)
        assert set(out) == {"mask", "predictions"}
        assert out["predictions"].boxes == [[0, 0, 4, 4]]

    def test_keys_on_the_output_itself_still_wins(self) -> None:
        out = DropOffMaskBoxes(keys={"kept": "filtered"})(_mask_and_boxes())
        assert out["filtered"].boxes == [[0, 0, 4, 4]]
        assert out["boxes"].boxes == [[0, 0, 4, 4], [6, 6, 9, 9]]

    def test_replaces_must_name_an_input(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"Wrong: the output 'kept' replaces 'box', which is not an input; the inputs are \['boxes'\]",
        ):

            class Wrong(Algorithm):
                boxes: Boxes = Input()
                kept: Boxes = Output(replaces="box")


class TestWhatTheUserReadsWhenTheRecordIsWrong:
    def test_the_entry_is_stored_under_another_name(self) -> None:
        message = (
            "BackgroundLevel needs the input 'image' (Image) from the record entry 'image', which this record "
            "does not carry (it has: photo) — if it is stored under another name, set keys: {image: <entry>}"
        )
        with pytest.raises(ValueError, match=f"^{_escaped(message)}$"):
            BackgroundLevel()({"photo": _image()})

    def test_keys_names_an_entry_the_record_does_not_carry(self) -> None:
        message = (
            "BackgroundLevel needs the input 'image' (Image) from the record entry 'picture', which this record "
            "does not carry (it has: photo)"
        )
        with pytest.raises(ValueError, match=f"^{_escaped(message)}$"):
            BackgroundLevel(keys={"image": "picture"})({"photo": _image()})

    def test_the_entry_has_the_wrong_item_type(self) -> None:
        message = (
            "BackgroundLevel: the input 'image' must be of type Image, but the record entry 'image' is of type ndarray"
        )
        with pytest.raises(ValueError, match=f"^{_escaped(message)}$"):
            BackgroundLevel()({"image": np.zeros((4, 4))})

    def test_keys_names_a_slot_that_does_not_exist(self) -> None:
        message = (
            "BackgroundLevel: keys names ['imgae'], which are not inputs or outputs; those are ['background', 'image']"
        )
        with pytest.raises(ValueError, match=f"^{_escaped(message)}$"):
            BackgroundLevel(keys={"imgae": "photo"})({"photo": _image()})

    def test_keys_must_be_a_mapping(self) -> None:
        """confluid refuses it when the object is built; the op refuses it too when it arrives later."""
        with pytest.raises(ValueError, match="keys"):
            BackgroundLevel(keys=["photo"])  # type: ignore[arg-type]
        algo = BackgroundLevel()
        algo.keys = ["photo"]  # type: ignore[assignment]
        message = "BackgroundLevel: keys must map an input or output name to a record entry; got a value of type list"
        with pytest.raises(ValueError, match=f"^{_escaped(message)}$"):
            algo({"photo": _image()})


def _escaped(text: str) -> str:
    return re.escape(text)


# --------------------------------------------------------------------------------------------
# declaring: what a class may and may not say
# --------------------------------------------------------------------------------------------


class TestDeclaring:
    def test_a_hand_written_constructor_is_refused(self) -> None:
        with pytest.raises(TypeError, match=r"Custom: an Algorithm's constructor is generated from its Param slots"):

            class Custom(Algorithm):
                level: float = Param(default=1.0)

                def __init__(self, level: float = 1.0) -> None:
                    self.level = level

    def test_a_slot_may_not_shadow_the_base_class(self) -> None:
        with pytest.raises(TypeError, match=r"Shadow: the slot name\(s\) \['run'\] are taken by Algorithm itself"):

            class Shadow(Algorithm):
                run: float = Input()  # type: ignore[assignment]

    def test_a_subclass_inherits_the_slots_and_may_add_more(self) -> None:
        class Tuned(BackgroundLevel):
            gain: float = Param(default=1.0, doc="Multiplier on the level.")

            def compute(self) -> Dict[str, Any]:
                return {"background": super().compute()["background"] * self.gain}

        image = _image()
        assert list(inspect.signature(Tuned).parameters) == ["percentile", "gain", "keys"]
        assert Tuned(gain=2.0)({"image": image})["background"] == 2.0 * _expected_background(image, 25.0)

    def test_a_subclass_without_new_slots_keeps_the_parent_constructor(self) -> None:
        """Regenerating it would drop confluid's validation wrapper around the parent's constructor."""

        class Renamed(BackgroundLevel):
            pass

        assert Renamed.__init__ is BackgroundLevel.__init__


# --------------------------------------------------------------------------------------------
# what every tool reads — derived, never written twice
# --------------------------------------------------------------------------------------------


class TestIntrospection:
    def test_algorithm_spec_lists_the_three_kinds_of_slot(self) -> None:
        spec = algorithm_spec(BackgroundLevel)
        assert [(s.name, s.doc, s.required) for s in spec.params] == [
            ("percentile", "Percentile rank in [0, 100] across the per-row medians.", False)
        ]
        assert spec.params[0].default == 25.0
        assert [(s.name, s.annotation, s.doc, s.required) for s in spec.inputs] == [
            ("image", Image, "The image to read.", True)
        ]
        assert [(s.name, s.annotation, s.doc) for s in spec.outputs] == [
            ("background", float, "The background level, in the image's own units.")
        ]

    def test_algorithm_spec_reads_an_instance_like_its_class(self) -> None:
        assert algorithm_spec(BackgroundLevel()) == algorithm_spec(BackgroundLevel)

    def test_algorithm_spec_names_optional_inputs_and_replacing_outputs(self) -> None:
        assert [(s.name, s.required, s.default) for s in algorithm_spec(ScaleWithOptionalMask).inputs] == [
            ("image", True, None),
            ("mask", False, None),
        ]
        assert [(s.name, s.replaces) for s in algorithm_spec(DropOffMaskBoxes).outputs] == [("kept", "boxes")]

    def test_confluid_sees_the_params_and_keys_as_settings_and_nothing_else(self) -> None:
        fields = to_pydantic(BackgroundLevel).model_json_schema()["properties"]
        assert set(fields) == {"percentile", "keys"}
        assert fields["percentile"]["minimum"] == 0.0 and fields["percentile"]["maximum"] == 100.0
        assert fields["percentile"]["default"] == 25.0
        assert fields["percentile"]["description"] == "Percentile rank in [0, 100] across the per-row medians."

    def test_confluid_sees_the_outputs_as_output_sockets(self) -> None:
        assert [(o["name"], o["type"]) for o in output_specs(BackgroundLevel)] == [("background", "float")]
        assert output_specs(BackgroundLevel)[0]["description"] == "The background level, in the image's own units."

    def test_the_param_docs_reach_the_args_block_every_gui_reads(self) -> None:
        docs = parse_param_docs(BackgroundLevel)
        assert docs["percentile"] == "Percentile rank in [0, 100] across the per-row medians."
        assert set(docs) == {"percentile", "keys"}

    def test_the_type_checker_is_told_which_members_are_constructor_arguments(self) -> None:
        """mypy/pyright read ``__dataclass_transform__`` and each field specifier's ``init`` default.

        Measured with mypy on the declaring spelling: ``BackgroundLevel(percentil=30.0)`` (typo),
        ``BackgroundLevel(percentile="high")`` (wrong type) and ``BackgroundLevel(image=...)`` (an input
        is not a setting) are errors, while ``BackgroundLevel()`` and ``BackgroundLevel(percentile=30.0)``
        are not. The spelling ``percentile: Param[float] = 25.0`` was rejected because mypy could not
        tell inputs from constructor arguments there and refused the correct call.
        """
        meta = Algorithm.__dataclass_transform__  # type: ignore[attr-defined]
        assert meta["kw_only_default"] is True
        assert set(meta["field_specifiers"]) == {Param, Input, Output}
        for specifier, init in ((Param, Literal[True]), (Input, Literal[False]), (Output, Literal[False])):
            parameter = inspect.signature(specifier).parameters["init"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.annotation == init


class TestChainContract:
    def test_consumes_names_the_record_entries_of_the_required_inputs(self) -> None:
        assert BackgroundLevel().consumes == {"image": "Image"}
        assert BackgroundLevel(keys={"image": "photo"}).consumes == {"photo": "Image"}
        assert ScaleWithOptionalMask().consumes == {"image": "Image"}

    def test_produces_names_the_record_entries_the_outputs_land_in(self) -> None:
        assert BackgroundLevel().produces == {"background": ANY_TYPE}
        assert KeepCentredBoxes().produces == {"kept": "Boxes"}
        assert DropOffMaskBoxes(keys={"boxes": "predictions"}).produces == {"predictions": "Boxes"}

    def test_the_chain_checker_refuses_an_input_nothing_provides(self) -> None:
        with pytest.raises(ChainContractError, match=r"pane\.yaml: BackgroundLevel needs the record entry 'image'"):
            check_chain([BackgroundLevel()], provided=["signal"], where="pane.yaml")

    def test_the_chain_checker_follows_keys_and_earlier_outputs(self) -> None:
        check_chain([BackgroundLevel(keys={"image": "photo"})], provided=["photo"])

        class Brighten(Algorithm):
            image: Image = Input()
            brighter: Image = Output()

            def compute(self) -> Dict[str, Any]:
                return {"brighter": Image(np.asarray(self.image) + 1.0, layout="HWC")}

        check_chain([Brighten(), BackgroundLevel(keys={"image": "brighter"})], provided=["image"])


# --------------------------------------------------------------------------------------------
# in the engine: YAML, flow graphs, spawn workers
# --------------------------------------------------------------------------------------------


def _class_tag(cls: type) -> str:
    return f"!class:{cls.__module__}.{cls.__qualname__}"


class TestInTheEngine:
    def test_it_sits_in_a_yaml_ops_list_and_dumps_back(self) -> None:
        text = f"""
stream: !class:recordstream.core.stream.Stream
  ops:
    - {_class_tag(BackgroundLevel)}
      percentile: 30
      keys: {{image: photo, background: level}}
"""
        op = confluid.load(text)["stream"].ops[0]
        image = _image()
        out = op({"photo": image})
        assert list(out) == ["photo", "level"]
        assert out["level"] == _expected_background(image, 30.0)
        rebuilt = confluid.load(confluid.dump(op))
        assert (rebuilt.percentile, rebuilt.keys) == (30, {"image": "photo", "background": "level"})

    def test_a_later_flow_step_binds_the_live_output(self) -> None:
        text = f"""
flow:
  level: {_class_tag(BackgroundLevel)} {{percentile: 50}}
  bright:
    op: !class:recordstream.ops.numpy.Threshold {{field: image, output: bright}}
    bind:
      low_level: level.background
"""
        image = _image()
        out = next(iter(FlowGraph.from_yaml(text, source=[{"image": image}])))
        level = _expected_background(image, 50.0)
        assert out["background"] == level
        expected = Threshold(field="image", output="bright", low_level=level)({"image": image})["bright"]
        assert np.array_equal(out["bright"], expected)

    def test_it_pickles_and_runs_in_spawn_workers(self) -> None:
        op = KeepCentredBoxes(keys={"kept": "boxes"})
        back = pickle.loads(pickle.dumps(op))
        assert back.keys == {"kept": "boxes"}
        records = []
        for i in range(6):
            mask = Mask(np.zeros((10, 10), bool))
            mask[: i + 1, : i + 1] = True
            records.append({"mask": mask, "boxes": Boxes(boxes=[[0, 0, 2, 2], [2, 2, 6, 6], [6, 6, 9, 9]])})
        serial = [len(r["boxes"].boxes) for r in Stream(source=records, ops=[op])]
        parallel = [len(r["boxes"].boxes) for r in Stream(source=records, ops=[op]).parallel(2)]
        assert serial == [0, 1, 1, 1, 2, 2]
        assert parallel == serial
        pids = {r["pid"] for r in Stream(source=[{"tag": i} for i in range(4)], ops=[WhereWasIRun()]).parallel(2)}
        assert os.getpid() not in pids, "the records were computed in this process, not in spawn workers"


def test_the_surface_is_exported_from_the_package() -> None:
    for name in ("Algorithm", "Param", "Input", "Output", "algorithm_spec", "AlgorithmSpec", "AlgorithmSlot"):
        assert getattr(recordstream, name) is getattr(recordstream.algorithm, name)
        assert name in recordstream.__all__
