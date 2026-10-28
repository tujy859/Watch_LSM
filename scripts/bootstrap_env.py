"""Create an isolated local venv and install an explicit CPU/CUDA PyTorch profile.

Use a Python 3.10-3.12 interpreter to run this script. No environment is copied.
The selected torch release is pinned to the project's validated baseline.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(arguments, env):
    print("Running:", " ".join(map(str, arguments)), flush=True)
    subprocess.run(list(map(str, arguments)), cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "cpu", "cu124"], default="cpu")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--torch-wheel", type=Path, help="Optional previously downloaded and verified local wheel cache")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        parser.error("Use Python 3.10, 3.11 or 3.12 for the pinned torch baseline")
    env_path = args.venv.resolve()
    if env_path == Path(sys.prefix).resolve():
        parser.error("Run bootstrap with a base Python, not the venv being modified")
    backend = args.backend
    if backend == "auto":
        backend = "cpu"
        executable = shutil.which("nvidia-smi")
        if executable:
            probe = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, timeout=15)
            if probe.returncode == 0:
                backend = "cu124"
    if backend == "cu124" and platform.system() not in ["Windows", "Linux"]:
        parser.error("cu124 is supported only on Windows/Linux; select cpu on macOS")
    local = ROOT / ".local"
    temp = local / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    # Avoid global package/cache mutation and sandbox-dependent temporary paths.
    env = os.environ.copy()
    env.update({"TMP": str(temp), "TEMP": str(temp), "TMPDIR": str(temp),
                "PIP_CACHE_DIR": str(local / "pip-cache"), "PYTHONNOUSERSITE": "1",
                "PIP_DEFAULT_TIMEOUT": "30"})
    os.environ.update({k: env[k] for k in ["TMP", "TEMP", "TMPDIR"]})
    import tempfile
    tempfile.tempdir = str(temp)
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_path)
    python = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([python, "-m", "pip", "install", "pip>=24.2,<26", "setuptools>=75,<76", "wheel>=0.44,<0.46"], env)
    torch_args = [python, "-m", "pip", "install", "-r", ROOT / f"requirements/torch-{backend}.txt"]
    if platform.system() == "Darwin":
        torch_args = [python, "-m", "pip", "install", "torch==2.5.1"]
    if args.torch_wheel:
        torch_args = [python, "-m", "pip", "install", args.torch_wheel.resolve(strict=True)]
    run(torch_args, env)
    install = [python, "-m", "pip", "install", "--no-build-isolation"]
    lock = ROOT / f"requirements/locks/{platform.system().lower()}-py{sys.version_info.major}{sys.version_info.minor}-common.txt"
    if lock.is_file():
        install.extend(["-r", lock])
    run([*install, "-e", ".[data]"], env)
    run([python, "-m", "pip", "check"], env)
    check = "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA wheel installed but driver/device unavailable'"
    if backend == "cu124":
        run([python, "-c", check], env)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--exclude-editable"], env=env, text=True)
    # Per-machine resolved versions; a freeze of cu124 must not be reused as CPU requirements.
    (local / f"requirements-{platform.system().lower()}-{backend}.lock.txt").write_text(frozen, encoding="utf-8")
    (local / "environment.json").write_text(json.dumps({"backend": backend, "python": str(python),
                                                       "base_python": sys.executable}, indent=2), encoding="utf-8")
    print(f"Environment ready: {python} ({backend})")


if __name__ == "__main__":
    main()
