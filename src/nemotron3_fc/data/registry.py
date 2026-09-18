"""Dataset adapter registration and construction."""

from __future__ import annotations

from collections.abc import Callable

from nemotron3_fc.data.adapters.base import DatasetAdapter

AdapterFactory = Callable[[], DatasetAdapter]


class DatasetRegistry:
    """Resolve dataset adapters by configuration name without coupling consumers to sources."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Dataset adapter name must not be empty")
        if normalized in self._factories:
            raise ValueError(f"Dataset adapter is already registered: {normalized}")
        self._factories[normalized] = factory

    def create(self, name: str) -> DatasetAdapter:
        normalized = name.strip().lower()
        try:
            return self._factories[normalized]()
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown dataset adapter '{name}'. Available: {available}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

