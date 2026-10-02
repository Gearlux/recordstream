"""A settings class written with postponed annotations: its signature hands a draw strings, not types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from confluid import configurable


@configurable
@dataclass(kw_only=True)
class ToyLater:
    width: Literal[4, 8] = 4
