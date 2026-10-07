import pytest


@pytest.fixture(autouse=True)
def _no_literature_network(monkeypatch):
    """Keep the offline suite offline: agents skip literature lookups unless a
    test re-enables them (and mocks HTTP) with monkeypatch.delenv."""
    monkeypatch.setenv("CO_SCIENTIST_DISABLE_LITERATURE", "1")
