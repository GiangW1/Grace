import builtins
import importlib.metadata

from grace_gc import versions


def test_installed_distribution_version_does_not_import_gpu_package(monkeypatch):
    monkeypatch.setattr(versions, "cuda_environment", lambda: {})
    monkeypatch.setattr(versions, "git_info", lambda: {"head": "synthetic"})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3" if name == "vllm" else "0.9.0")
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"vllm", "math_verify"}:
            raise AssertionError("version collection must not import these packages")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    report = versions.collect_versions(extra_modules=("vllm", "math_verify"))
    assert report["packages"] == {"vllm": "1.2.3", "math_verify": "0.9.0"}
    assert report["package_version_sources"] == {"vllm": "distribution_metadata", "math_verify": "distribution_metadata"}


def test_module_only_install_and_missing_package_keep_explicit_sources(monkeypatch):
    monkeypatch.setattr(versions, "cuda_environment", lambda: {})
    monkeypatch.setattr(versions, "git_info", lambda: {"head": "synthetic"})
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", absent)
    result = versions.collect_versions(extra_modules=("json", "grace_missing_package"))
    assert result["packages"]["json"]
    assert result["package_version_sources"]["json"] == "module_attribute"
    assert result["packages"]["grace_missing_package"] is None
    assert "grace_missing_package" in result["missing"]
