"""Helpers that let the data-dependent tests run on any checkout and platform.

The notebook feature caches name recordings by the absolute path of the machine that wrote them (a Linux home
directory); local_recording maps such a name to the same file under this checkout's Data_clean. Symbolic links
need a privilege on Windows that a normal account lacks, so link_or_copy falls back to a copy.
"""
import shutil
from pathlib import Path

GNB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GNB_DIR.parent
DATA_DIR = REPO_ROOT / "Data_clean"


def local_recording(path):
    """The recording `path` (as a cache or CSV names it, on any platform) under this checkout's Data_clean, as a
    POSIX-style string; a path without a Data_clean folder is returned unchanged."""
    parts = Path(str(path).replace("\\", "/")).as_posix().split("/")
    if "Data_clean" not in parts:
        return str(path)
    rel = parts[parts.index("Data_clean") + 1:]
    # POSIX form on every platform: the sorted order of recording names fixes the notebook's group codes and
    # therefore its split, so the separator must not differ from the machine that wrote the cache
    return DATA_DIR.joinpath(*rel).as_posix()


def link_or_copy(link, target):
    """Make `link` refer to the file `target`: a symbolic link where allowed, otherwise a copy."""
    link, target = Path(link), Path(target)
    try:
        link.symlink_to(target)
    except OSError:                                  # Windows without the symlink privilege (developer mode)
        shutil.copy2(target, link)
    return link
