"""Use native CUDA convolutions with an explicitly recorded benchmark profile.

The precision and original neural search space/budget remain controlled by the
existing worker. A changed convolution backend is kept in a separate output.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if "--output" not in sys.argv:
        raise ValueError("An explicit --output is required")
    import torch
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = {"convolution_backend": "native_cuda", "cudnn_enabled": False, "torch": torch.__version__, "cudnn_version": torch.backends.cudnn.version(), "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "note": "Backend adaptation; not the default cuDNN execution. Precision and HPO settings are recorded separately in execution-profile.json."}
    target = output / "backend-profile.json"
    if target.exists() and json.loads(target.read_text()) != profile:
        raise ValueError("Backend settings changed; use a new output directory")
    target.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    torch.backends.cudnn.enabled = False
    from scripts.ppg_bench_neural_worker import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
