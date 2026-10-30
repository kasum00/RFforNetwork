"""Dependency metadata only; imports no NumPy/sklearn or dataset."""
import importlib.metadata
import platform


def environment():
    python = tuple(map(int, platform.python_version_tuple()[:2]))
    expected = {"numpy": "1.26.4" if python < (3, 13) else "2.2.6",
                "pandas": "2.3.3", "scipy": "1.15.3", "scikit-learn": "1.7.2",
                "joblib": "1.5.3", "psutil": "5.9.5", "pyarrow": "23.0.1", "tqdm": "4.67.1"}
    versions, errors = {}, []
    for name, version in expected.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != version:
            errors.append(f"{name}: expected {version}, installed {versions[name]}")
    if not (3, 10) <= python < (3, 14):
        errors.append("Approved dependency profile supports Python 3.10..3.13")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "errors": errors}
