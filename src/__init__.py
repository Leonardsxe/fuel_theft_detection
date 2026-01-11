"""Top-level package for the fuel theft detection project."""

from importlib import import_module

# Lazily expose common subpackages so downstream code can `import src` once.
config = import_module("src.config")
data = import_module("src.data")
features = import_module("src.features")
detection = import_module("src.detection")
clustering = import_module("src.clustering")
models = import_module("src.models")
pipeline = import_module("src.pipeline")
utils = import_module("src.utils")

__all__ = [
    "config",
    "data",
    "features",
    "detection",
    "clustering",
    "models",
    "pipeline",
    "utils",
]
