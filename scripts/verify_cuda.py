#!/usr/bin/env python
from __future__ import annotations

import json
import platform

from _bootstrap import add_src_to_path

add_src_to_path()

import torch

from cfeg.runtime import cuda_execution_probe


def main() -> None:
    expected = {
        "python_major_minor": "3.10",
        "torch": "2.2.2+cu121",
        "torch_cuda": "12.1",
    }
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if ".".join(observed["python"].split(".")[:2]) != expected["python_major_minor"]:
        raise RuntimeError(f"Expected Python 3.10, observed {observed['python']}.")
    if observed["torch"] != expected["torch"]:
        raise RuntimeError(f"Expected torch {expected['torch']}, observed {observed['torch']}.")
    if observed["torch_cuda"] != expected["torch_cuda"]:
        raise RuntimeError(
            f"Expected torch CUDA {expected['torch_cuda']}, observed {observed['torch_cuda']}."
        )
    observed["probe"] = cuda_execution_probe()
    print(json.dumps({"status": "ready", "expected": expected, "observed": observed}, sort_keys=True))


if __name__ == "__main__":
    main()
