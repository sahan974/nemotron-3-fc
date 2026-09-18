"""Dataset adapter protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Protocol, runtime_checkable

from nemotron3_fc.data.schema import CanonicalRecord


@runtime_checkable
class DatasetAdapter(Protocol):
    """Translate a source dataset split into canonical records."""

    name: str

    def load_split(self, path: Path, options: Mapping[str, object] | None = None) -> Iterable[CanonicalRecord]:
        """Yield validated canonical records from one source split."""
        ...

