"""Compatibility entry point for the reusable gender visual analysis module."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.analysis.gender_visuals import revision_v2 as _impl
from src.pipelines.analysis.gender_visuals.revision_v2 import *  # noqa: F401,F403,E402


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))


if __name__ == "__main__":
    raise SystemExit(_impl.main())
