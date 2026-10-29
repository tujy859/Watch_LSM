"""Create an isolated local venv and install an explicit CPU/CUDA PyTorch profile.

Use a Python 3.10-3.12 interpreter to run this script. No environment is copied.
The selected torch release is pinned to the project's validated baseline.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
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
    parser.add_argument("--wheelhouse", type=Path, help="Optional complete wheel cache; disable package indexes for offline installation")
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
                "PIP_DEFAULT_TIMEOUT": "30", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    os.environ.update({k: env[k] for k in ["TMP", "TEMP", "TMPDIR"]})
    import tempfile
    tempfile.tempdir = str(temp)
    python = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.is_file():
        probe = subprocess.check_output([str(python), "-c",
            "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix,list(sys.version_info[:2])]))"], env=env, text=True)
        prefix, base_prefix, version = json.loads(probe)
        config = (env_path / "pyvenv.cfg").read_text(encoding="utf-8").lower()
        if (Path(prefix).resolve() != env_path or prefix == base_prefix
                or "include-system-site-packages = false" not in config):
            raise RuntimeError("Existing target is not an isolated project venv; choose a new --venv directory")
        if tuple(version) != sys.version_info[:2]:
            raise RuntimeError("Existing venv has a different Python version. Run bootstrap with its base Python or choose a new --venv directory")
    else:
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_path)
    pip_install = [python, "-m", "pip", "install"]
    if args.wheelhouse:
        pip_install.extend(["--no-index", "--find-links", args.wheelhouse.resolve(strict=True)])
    run([*pip_install, "pip>=24.2,<26", "setuptools>=75,<76", "wheel>=0.44,<0.46"], env)
    torch_args = [*pip_install, "-r", ROOT / f"requirements/torch-{backend}.txt"]
    if platform.system() == "Darwin":
        torch_args = [*pip_install, "torch==2.5.1"]
    if args.torch_wheel:
        # Explicit cache use also repairs interrupted installs. Resolve runtime
        # dependencies in the project install below, after replacing torch.
        torch_args = [*pip_install, "--force-reinstall", "--no-deps", args.torch_wheel.resolve(strict=True)]
    run(torch_args, env)
    install = [*pip_install, "--no-build-isolation"]
    lock = ROOT / f"requirements/locks/{platform.system().lower()}-py{sys.version_info.major}{sys.version_info.minor}-common.txt"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+(?:\s*;.*)?", line):
                raise ValueError("Common dependency locks must contain portable version pins, not local wheel URLs")
            name = line.split("==", 1)[0].lower().replace("_", "-")
            if name == "torch" or name.startswith(("nvidia-", "triton", "pytorch-triton")):
                raise ValueError("Torch and accelerator packages must stay in their hardware profiles, not common locks")
        install.extend(["-r", lock])
    run([*install, "-e", ".[data]"], env)
    run([python, "-m", "pip", "check"], env)
    check = "import torch; print(torch.__version__); assert torch.version.cuda == '12.4', 'Wrong torch CUDA build'; assert torch.cuda.is_available(), 'CUDA wheel installed but driver/device unavailable'"
    if backend == "cu124":
        run([python, "-c", check], env)
    else:
        run([python, "-c", "import torch; assert torch.version.cuda is None, 'CPU profile still has a CUDA build'"], env)
    # cuda_available alone does not catch missing NVRTC libraries or unsupported
    # kernels. Require one real model forward/backward before declaring readiness.
    device = "cuda" if backend == "cu124" else "cpu"
    run([python, ROOT / "scripts/doctor.py", "--device", device, "--train-step",
         "--output", local / f"doctor-{device}.json"], env)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze", "--exclude-editable"], env=env, text=True)
    # Per-machine resolved versions; a freeze of cu124 must not be reused as CPU requirements.
    (local / f"requirements-{platform.system().lower()}-{backend}.lock.txt").write_text(frozen, encoding="utf-8")
    (local / "environment.json").write_text(json.dumps({"backend": backend, "python": str(python),
                                                       "base_python": sys.executable}, indent=2), encoding="utf-8")
    print(f"Environment ready: {python} ({backend})")


if __name__ == "__main__":
    main()
