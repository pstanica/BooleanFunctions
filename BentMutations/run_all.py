"""Run all verification programs and print their results."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PYTHON_PROGRAMS = (
    "verify_seed_and_lift.py",
    "enumerate_isotropic_subspaces.py",
    "enumerate_circulant_graphs.py",
    "verify_su8.py",
)
CPP_PROGRAM = "enumerate_quartic_replacements.cpp"


def run_all():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or later is required.")
    if sys.flags.optimize:
        raise SystemExit("Run without Python optimization so assertions are enabled.")

    for program in PYTHON_PROGRAMS:
        print(f"== {program} ==", flush=True)
        subprocess.run([sys.executable, str(ROOT / program)], check=True)

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with TemporaryDirectory() as temporary:
        executable = Path(temporary) / "enumerate_quartic_replacements"
        subprocess.run(
            [
                *compiler, "-std=c++20", "-O3", "-Wall", "-Wextra", "-pedantic",
                str(ROOT / CPP_PROGRAM), "-o", str(executable),
            ],
            check=True,
        )
        print(f"== {CPP_PROGRAM} ==", flush=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    run_all()
