"""Dataset adapter protocol."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from nemotron3_fc.data.schema import CanonicalRecord


@runtime_checkable
class DatasetAdapter(Protocol):
    """Translate one raw source record into the canonical schema."""

    name: str

    def source_id(self, raw: Mapping[str, Any], index: int) -> str:
        """Return the stable canonical ID used for retained and rejected records."""
        ...

    def convert_record(self, raw: Mapping[str, Any], index: int) -> CanonicalRecord:
        """Convert and validate one raw record."""
        ...
