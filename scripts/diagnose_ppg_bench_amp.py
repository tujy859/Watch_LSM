"""Compare the failed PaPaGei-S candidate across CUDA arithmetic backends.

Diagnostic only: one fixed candidate on the original training/validation split,
not a replacement for the 48-trial benchmark. Does not use the outer test set.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/ppg-bench-amp-diagnostic")
    parser.add_argument("--modes", nargs="+", choices=["amp_cudnn", "amp_native", "fp32_cudnn"], default=["amp_cudnn", "amp_native", "fp32_cudnn"])
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from ppg_bench.datasets import BenchmarkDataset, select_tasks
    from ppg_bench.models import get_port, load_models
    from ppg_bench.run import make_jobs, seed_everything
    from ppg_bench import neural

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    source = next((ROOT / ".local/ppg-bench-finetune-capacity").rglob("optuna.db"))
    with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as connection:
        params = {name: json.loads(distribution)["attributes"]["choices"][int(value)] for name, value, distribution in connection.execute("select param_name,param_value,distribution_json from trial_params where trial_id=(select trial_id from trials where number=0)")}
    params.pop("aggregation_idx", None)
    config = next(m for m in load_models(ROOT / "watch_lsm/configs/ppg_bench_models.yaml") if m["name"] == "PaPaGei-S")
    job = make_jobs([config], select_tasks(["ppgbp-sysbp"]), ROOT / "datasets/ppg_bench_processed", ["finetune"], [42], [0])[0]
    dataset = BenchmarkDataset(job["root"], job["train"], job["task"])
    train, validation = neural.subject_holdout_split(np.asarray(dataset.subjects), 0.2, 42)
    report = {"diagnostic_only": True, "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "task": job["task"].name, "seed": 42, "fold": 0, "params": params, "train_windows": len(train), "validation_windows": len(validation), "outer_test_used": False, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "bf16_supported": torch.cuda.is_bf16_supported(), "modes": {}}

    def save():
        temporary = args.output / "report.tmp.json"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output / "report.json")

    for mode in args.modes:
        seed_everything(42)
        torch.backends.cudnn.enabled = mode != "amp_native"
        port = get_port(config)
        result = {"amp": mode != "fp32_cudnn", "cudnn_enabled": torch.backends.cudnn.enabled, "epochs": [], "status": "running"}
        report["modes"][mode] = result
        save()
        handles = []
        model = optimizer = loader = None
        try:
            dimension = neural.infer_head_input_dim(port, dataset, config, "cuda")
            model = neural.build_finetune_model(port, config, dimension, 1, params, "cuda")

            def hook(name):
                def check(module, inputs, output):
                    tensors = [output] if isinstance(output, torch.Tensor) else list(output) if isinstance(output, (tuple, list)) else []
                    for tensor in tensors:
                        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                            result["first_nonfinite_module"] = {"name": name, "type": type(module).__name__, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "input_max_abs": float(inputs[0].detach().abs().max()) if inputs and isinstance(inputs[0], torch.Tensor) and torch.isfinite(inputs[0]).all() else None}
                            if isinstance(module, torch.nn.Conv1d):
                                # Compare the exact same input and weights, not
                                # activations from independently trained models.
                                arguments = (module.stride, module.padding, module.dilation, module.groups)
                                with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
                                    native = torch.nn.functional.conv1d(inputs[0], module.weight, module.bias, *arguments)
                                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                                    reference = torch.nn.functional.conv1d(inputs[0].float(), module.weight, module.bias, *arguments)
                                result["same_convolution_comparison"] = {"native_fp16_finite": bool(torch.isfinite(native).all()), "cudnn_fp32_finite": bool(torch.isfinite(reference).all()), "native_max_abs": float(native.abs().max()) if torch.isfinite(native).all() else None, "fp32_max_abs": float(reference.abs().max()) if torch.isfinite(reference).all() else None, "native_fp16_vs_fp32_max_abs_error": float((native.float() - reference).abs().max()) if torch.isfinite(native).all() and torch.isfinite(reference).all() else None}
                            raise FloatingPointError(f"First non-finite activation: {name}")
                return check

            for name, module in model.named_modules():
                if not list(module.children()):
                    handles.append(module.register_forward_hook(hook(name)))
            optimizer = neural.make_optimizer(model, params)
            loader = neural.make_loader(dataset, train, params["batch_size"], True, 0, 42)
            scaler = torch.amp.GradScaler("cuda", enabled=result["amp"])
            for epoch in range(15):
                loss = neural.train_one_epoch(model, port, dataset, loader, config, "regression", None, neural.loss_for_task("regression", None), optimizer, "cuda", result["amp"], scaler, params["batchnorm_mode"])
                validation_result = neural.evaluate_and_score(model, port, dataset, validation, config, {"eval_batch_size": 32, "random_state": 42}, "regression", "subject", "regression", None, {"name": "mean"}, {}, "cuda")
                if not np.isfinite(loss) or not np.isfinite(validation_result["mae"]):
                    raise FloatingPointError("Non-finite loss or validation MAE")
                result["epochs"].append({"epoch": epoch + 1, "train_loss": loss, "validation_mae": validation_result["mae"], "gradient_scale": scaler.get_scale()})
                print(mode, epoch + 1, validation_result["mae"], flush=True)
                save()
            result["status"] = "passed"
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(mode, result["error"], flush=True)
        finally:
            for handle in handles:
                handle.remove()
            model = optimizer = loader = None
            gc.collect()
            torch.cuda.empty_cache()
            save()
    print(args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
