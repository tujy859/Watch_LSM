"""Report this machine's environment and test model execution without training assets."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from watch_lsm.models.backbone import WatchLSMModel
from watch_lsm.models.losses import CompositePretrainLoss
from watch_lsm.paths import PROJECT_ROOT, data_path, model_path
from watch_lsm.runtime import select_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--checkpoint", action="store_true", help="Strictly load this machine's saved Watch-LSM checkpoint")
    parser.add_argument("--train-step", action="store_true", help="One synthetic backward/optimizer step; never saves weights")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = select_device(args.device)
    seed_everything(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = model_path("checkpoints", "pretrain", "best_foundation_model.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    model = WatchLSMModel(**(state.get("model_config", {}) if state else {}))
    if state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.train(args.train_step)
    patches = model.tokenizer.num_patches if hasattr(model.tokenizer, "num_patches") else 32
    # Current architecture baseline uses one PPG and three ACC channels.
    p_size = model.ppg_decoder[-1].out_features
    a_size = model.acc_decoder[-1].out_features // 3
    ppg = torch.randn(2, 1, patches * p_size, device=device)
    acc = torch.randn(2, 3, patches * a_size, device=device)
    with torch.set_grad_enabled(args.train_step):
        outputs = model(ppg, acc)
        loss = CompositePretrainLoss()(outputs)["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite model loss")
        if args.train_step:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError("Nonfinite gradients")
            optimizer.step()
    report = {"python": sys.version, "executable": sys.executable,
              "isolated_venv": sys.prefix != sys.base_prefix,
              "platform": platform.platform(), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "h5py", "wfdb", "mat73", "PyYAML", "gdown"]},
              "datasets": str(data_path()), "models": str(model_path()),
              "checkpoint_loaded": bool(state), "synthetic_backward": args.train_step,
              "loss": loss.item(), "embedding_shape": list(outputs["global_embedding"].shape)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
