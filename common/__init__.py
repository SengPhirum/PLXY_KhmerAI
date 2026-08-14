"""Shared, dependency-light utilities used by every component of the platform.

Nothing in this package may import FastAPI, torch, transformers or any other
heavy optional dependency at module import time.  That rule keeps the data
pipeline, the RAG core and the test suite runnable from `requirements/base.txt`
alone.
"""

from common.versions import PlatformVersions, load_versions

__all__ = ["PlatformVersions", "load_versions"]
