"""Result types for top-down generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentguard.llm.types import CostEstimate

if TYPE_CHECKING:
    from agentguard.tracing.trace import Trace


@dataclass
class FileEntry:
    """A single file in the skeleton."""

    path: str
    purpose: str


@dataclass
class SkeletonResult:
    """L1 output — file tree with responsibilities."""

    files: list[FileEntry] = field(default_factory=list)


@dataclass
class ContractsResult:
    """L2 output — typed stubs for each file."""

    # Map of file path → code content (typed stubs with NotImplementedError)
    files: dict[str, str] = field(default_factory=dict)
    skeleton: SkeletonResult = field(default_factory=SkeletonResult)


@dataclass
class WiringResult:
    """L3 output — files with imports wired up."""

    # Map of file path → code content (with imports, still NotImplementedError bodies)
    files: dict[str, str] = field(default_factory=dict)
    contracts: ContractsResult = field(default_factory=ContractsResult)


@dataclass
class LogicResult:
    """L4 output — files with function bodies implemented."""

    # Map of file path → code content (fully implemented)
    files: dict[str, str] = field(default_factory=dict)
    wiring: WiringResult = field(default_factory=WiringResult)


@dataclass
class GenerationResult:
    """Complete result of a top-down generation run."""

    skeleton: SkeletonResult
    contracts: ContractsResult
    wiring: WiringResult
    logic: LogicResult
    trace: Trace | None = None
    total_cost: CostEstimate = field(default_factory=CostEstimate.zero)
    validation_fixes: int = 0
    challenge_reworks: int = 0

    @property
    def files(self) -> dict[str, str]:
        """Final file contents (from L4 logic, falling back to L3 wiring)."""
        result = dict(self.wiring.files)
        result.update(self.logic.files)
        return result
