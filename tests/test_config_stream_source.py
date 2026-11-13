"""``ConfigStreamSource``: the records of a stream another config holds — its source alone, or with its ops."""

from pathlib import Path

import pytest

from recordstream import Stream
from recordstream.sources import ConfigStreamSource

CONFIG = """
train_set: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.range.RangeSource
    source: !class:recordstream.sources.files.FilesSource
      files: [a.png, b.png, c.png]
  ops:
    - !class:recordstream.ops.structure.RenameField {src: file, dst: path}
val_set: !class:recordstream.core.stream.Stream
  source: !class:recordstream.sources.files.FilesSource
    files: [v.png]
small: !scope:size=small
  train_set: !class:recordstream.core.stream.Stream
    source: !class:recordstream.sources.files.FilesSource
      files: [only.png]
number: 3
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    file = tmp_path / "train.yaml"
    file.write_text(CONFIG)
    return file


def test_it_builds_with_no_argument_and_loads_nothing() -> None:
    source = ConfigStreamSource()
    assert source._records is None
    with pytest.raises(ValueError, match="name the config"):
        len(source)


def test_by_default_it_reads_the_streams_source_without_its_ops(config: Path) -> None:
    source = ConfigStreamSource(config=str(config))
    records = list(source)
    assert [r["file"] for r in records] == ["a.png", "b.png", "c.png"]
    assert len(source) == 3 and source[1]["file"] == "b.png"


def test_apply_ops_reads_the_stream_as_the_config_defines_it(config: Path) -> None:
    records = list(ConfigStreamSource(config=str(config), apply_ops=True))
    assert [r["path"] for r in records] == ["a.png", "b.png", "c.png"]


def test_the_key_and_the_scopes_reach_the_config(config: Path) -> None:
    assert [r["file"] for r in ConfigStreamSource(config=str(config), key="val_set")] == ["v.png"]
    assert [r["file"] for r in ConfigStreamSource(config=str(config), scopes=["size=small"])] == ["only.png"]


def test_it_feeds_a_stream_of_its_own(config: Path) -> None:
    from recordstream.ops.structure import RenameField

    stream = Stream(ConfigStreamSource(config=str(config)), ops=[RenameField(src="file", dst="name")])
    assert [r["name"] for r in stream] == ["a.png", "b.png", "c.png"]


@pytest.mark.parametrize(
    "settings, said",
    [
        ({"key": "test_set"}, "holds no 'test_set' — it holds: train_set, val_set"),
        ({"key": "number"}, "not a stream with a source"),
    ],
)
def test_a_key_or_setting_it_cannot_read_is_refused_by_name(config: Path, settings: dict, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        list(ConfigStreamSource(config=str(config), **settings))


def test_a_config_it_cannot_read_declares_nothing(tmp_path: Path) -> None:
    assert ConfigStreamSource(config=str(tmp_path / "missing.yaml")).produces == {}


def test_it_is_a_source_of_the_palette() -> None:
    assert ConfigStreamSource.__confluid_category__ == "source"
