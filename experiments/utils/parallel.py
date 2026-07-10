"""CUDA stream + thread-pool helpers."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    """Thread-safe print."""
    import builtins
    with _print_lock:
        builtins.print(*args, **kwargs)


@contextmanager
def cuda_stream(stream: Optional["object"] = None) -> Iterator[None]:
    """Run inside an explicit CUDA stream when available; no-op on CPU."""
    try:
        import torch
        if stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(stream):
                yield
            return
    except ImportError:
        pass
    yield


def make_cuda_stream() -> Optional["object"]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.Stream()
    except ImportError:
        pass
    return None


def clear_gpu() -> None:
    """gc + cuda cache empty + sync."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
