"""Environment smoke test for the PCPA project.

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\check_env.py

Checks the interpreter version, the PyTorch / CUDA build, whether the
GPU actually supports the kernels compiled into torch (the RTX 5070 is
Blackwell sm_120), and that every pure-python dependency imports.
"""

from __future__ import annotations

import platform
import sys

OK = "OK  "
WARN = "WARN"
FAIL = "FAIL"

LIGHT_DEPS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("opencv", "cv2"),
    ("matplotlib", "matplotlib"),
    ("PyYAML", "yaml"),
    ("scikit-learn", "sklearn"),
    ("h5py", "h5py"),
    ("pandas", "pandas"),
    ("tqdm", "tqdm"),
    ("pillow", "PIL"),
]

TORCH_HINT = (
    "install it with:\n"
    "        .\\.venv\\Scripts\\python.exe -m pip install torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu128"
)


def line(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def check_python() -> int:
    line(OK, f"Python {platform.python_version()} on {platform.machine()}")
    line(OK, f"executable: {sys.executable}")
    major_minor = sys.version_info[:2]
    if major_minor == (3, 13):
        line(OK, "interpreter 3.13 --- the recommended version for this project")
        return 0
    if major_minor == (3, 14):
        line(WARN, "3.14 detected: torch>=2.10 ships cp314 wheels, but opencv "
                   "and scikit-learn wheels still lag; 3.13 is recommended")
        return 0
    line(WARN, f"unexpected interpreter {major_minor}; 3.13 is recommended")
    return 0


def check_torch() -> int:
    problems = 0
    try:
        import torch
    except ImportError as exc:  # noqa: BLE001
        line(FAIL, f"torch is not installed ({exc})")
        line(FAIL, TORCH_HINT)
        return 1

    line(OK, f"torch {torch.__version__}")
    line(OK, f"torch built with CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        line(FAIL, "torch.cuda.is_available() is False --- you need a CUDA build")
        line(FAIL, TORCH_HINT)
        return problems + 1

    count = torch.cuda.device_count()
    line(OK, f"CUDA available, {count} device(s)")
    arch_list = torch.cuda.get_arch_list()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        line(OK, f"  [{i}] {props.name}, {props.total_memory / 1024 ** 3:.1f} GiB, "
                 f"compute capability {props.major}.{props.minor}")
        want = f"sm_{props.major}{props.minor}"
        if want not in arch_list:
            line(WARN, f"device needs {want} but torch only provides {arch_list}")
            problems += 1

    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device)
    checksum = torch.mm(a, a).sum().item()
    line(OK, f"GPU matmul smoke test passed (checksum {checksum:.3f})")
    line(OK, f"torch arch list: {arch_list}")
    return problems


def check_light_deps() -> int:
    problems = 0
    for name, module in LIGHT_DEPS:
        try:
            mod = __import__(module)
        except ImportError:
            line(FAIL, f"{name} is missing")
            problems += 1
        else:
            line(OK, f"{name} {getattr(mod, '__version__', '?')}")
    return problems


def main() -> int:
    problems = 0
    problems += check_python()
    print()
    problems += check_torch()
    print()
    problems += check_light_deps()
    print()
    if problems:
        print(f"{problems} problem(s) found --- see FAIL/WARN lines above.")
        return 1
    print("Environment looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
