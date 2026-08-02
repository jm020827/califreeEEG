from __future__ import annotations

from pathlib import Path

from cfeg.assets.hf import resolve_hf_hub_cache


def test_hf_hub_cache_environment_has_priority(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    monkeypatch.setenv("HF_HUB_CACHE", "/cache/custom-hub")
    assert resolve_hf_hub_cache() == Path("/cache/custom-hub")


def test_hf_home_defaults_to_hub_subdirectory(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    assert resolve_hf_hub_cache() == Path("/cache/hf/hub")


def test_explicit_cache_dir_has_priority(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/cache/custom-hub")
    assert resolve_hf_hub_cache("/cache/explicit") == Path("/cache/explicit")
