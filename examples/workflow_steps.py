"""A workflow's steps around its runs — a check, a folder emptied once, two configs run in turn — and its dry run.

The shell script it replaces: refuse when an input folder is empty, delete the output folder unless it holds this
layout's marker, then run two conversions. As ONE document run by ``recordstream run``:

1. ``recordstream run workflow.yaml --dry_run`` prints every step, changes nothing, exits 0 (every step would run);
2. ``recordstream run workflow.yaml`` runs them: the output folder made with its marker, both conversions done;
3. with the second input emptied, the dry run exits 1 naming the folder and the command that fills it, and a real
   run refuses before its first step — the first conversion does not run again.

Standalone, zero-arg, exit 0 (CI runs every ``examples/*.py``).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

WORKFLOW = """
paths:
  output: out/converted
runnable: !class:recordstream.workflow.Sequence
  steps:
    - !class:recordstream.workflow.ReplaceFolder { path: "${paths.output}", marker: .converted_v1 }
    - !class:recordstream.workflow.RunConfig
      name: images
      config: convert.yaml
      settings: { input: images, output: "${paths.output}/images.h5" }
      hint: fetch the images first
    - !class:recordstream.workflow.RunConfig
      name: labels
      config: convert.yaml
      settings: { input: labels, output: "${paths.output}/labels.h5" }
      hint: fetch the labels first
"""

# One conversion: every JSON file of `input` read as a record, written into an HDF5 store at `output`.
CONVERT = """
input: images
output: out/converted/images.h5
runnable: !class:recordstream.processing.DatasetProcessor
  stream: !class:recordstream.core.stream.Stream
    source: !class:recordstream.sources.files.FilesSource
      root: !ref:input
      pattern: "*.json"
  sink: !class:recordstream.storage.hdf5.HDF5Sink
    path: !ref:output
"""


def recordstream_run(folder: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    """``recordstream run <args>`` in ``folder``, as a person would type it."""
    command: List[str] = [sys.executable, "-m", "recordstream.cli", "run", *args]
    return subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=300, check=False, env=os.environ)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "workflow.yaml").write_text(WORKFLOW)
        (folder / "convert.yaml").write_text(CONVERT)
        (folder / "out").mkdir()
        for name in ("images", "labels"):
            (folder / name).mkdir()
            for index in range(3):
                (folder / name / f"{name}_{index}.json").write_text("{}")

        planned = recordstream_run(folder, "workflow.yaml", "--dry_run")
        print(planned.stdout)
        assert planned.returncode == 0, planned.stdout + planned.stderr
        assert "records: 3, start: 0, to_build: 3, sink: HDF5Sink" in planned.stdout
        assert not (folder / "out" / "converted").exists(), "a dry run changes nothing"

        ran = recordstream_run(folder, "workflow.yaml")
        assert ran.returncode == 0, ran.stderr
        converted = folder / "out" / "converted"
        assert (converted / ".converted_v1").is_file()
        assert (converted / "images.h5").is_file() and (converted / "labels.h5").is_file()

        for emptied in (folder / "labels").iterdir():
            emptied.unlink()
        refused = recordstream_run(folder, "workflow.yaml", "--dry_run")
        print(refused.stdout)
        assert refused.returncode == 1
        assert "matched no files under 'labels'" in refused.stdout and "run: fetch the labels first" in refused.stdout

        images_written = (converted / "images.h5").stat().st_mtime_ns
        stopped = recordstream_run(folder, "workflow.yaml")
        print(stopped.stderr.strip().splitlines()[-1])
        assert stopped.returncode == 1 and "Error: [3/3] labels:" in stopped.stderr
        assert (converted / "images.h5").stat().st_mtime_ns == images_written, "nothing ran before the refusal"
    print("workflow steps: dry run, run and refusal all as documented")


if __name__ == "__main__":
    main()
