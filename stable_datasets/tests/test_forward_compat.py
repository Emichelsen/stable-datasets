"""Compatibility with the current and legacy forward function names."""

from types import SimpleNamespace

import pytest

from benchmarks import models


@pytest.mark.parametrize("name", ["simclr", "barlow_twins"])
@pytest.mark.parametrize("api", ["current", "legacy", "both"])
def test_resolve_forward(monkeypatch, name, api):
    def current():
        pass

    def legacy():
        pass

    namespace = {}
    if api in {"current", "both"}:
        namespace[name] = current
    if api in {"legacy", "both"}:
        namespace[f"{name}_forward"] = legacy
    monkeypatch.setattr(models, "forward", SimpleNamespace(**namespace))
    assert models.resolve_forward(name) is (legacy if api == "legacy" else current)


@pytest.mark.parametrize("name", ["simclr", "barlow_twins"])
def test_missing_forward_has_actionable_error(monkeypatch, name):
    monkeypatch.setattr(models, "forward", SimpleNamespace())
    with pytest.raises(AttributeError, match="check the installed stable-pretraining version"):
        models.resolve_forward(name)
