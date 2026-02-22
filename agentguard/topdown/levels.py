"""Level definitions for top-down generation."""

from enum import StrEnum


class Level(StrEnum):
    """The four levels of top-down generation."""

    SKELETON = "skeleton"       # L1: file tree + responsibilities
    CONTRACTS = "contracts"     # L2: typed stubs with signatures
    WIRING = "wiring"           # L3: import graph + call chain
    LOGIC = "logic"             # L4: function implementation
