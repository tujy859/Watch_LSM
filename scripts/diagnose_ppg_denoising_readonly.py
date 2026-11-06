"""Recompute read-only denoising diagnostics. Never fit or run a model.

Run from repository root using the existing benchmark environment:
  .local/ppg-bench-venv/Scripts/python.exe scripts/diagnose_ppg_denoising_readonly.py --output .local/ppg-analysis-20261106/denoising_recheck_project_script.json

Only the explicitly named new output may be written. Existing outputs are
refused. Formal sources, predictions, checkpoints and reports remain untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max()),
            "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}


def diagnostic(y, p):
    y = y[:, 64:448].astype(float)
    p = p[:, 64:448].astype(float)
    yc = y - y.mean(axis=1, keepdims=True)
    pc = p - p.mean(axis=1, keepdims=True)
    ye = np.mean(y*y, axis=1)
    pe = np.mean(p*p, axis=1)
    return {"nrmse": np.sqrt(np.mean((p-y)**2, axis=1)/ye),
            "prediction_target_rms_ratio": np.sqrt(pe/ye),
            "prediction_target_centered_std_ratio": np.std(p, axis=1)/np.std(y, axis=1),
            "zero_intercept_gain": np.mean(p*y, axis=1)/ye,
            "correlation": np.sum(pc*yc, axis=1)/np.sqrt(np.sum(pc*pc, axis=1)*np.sum(yc*yc, axis=1))}


def build():
    root = ROOT / ".local/ppg-formal-expansion/denoising"
    runs = json.loads((root / "results.json").read_text())
    methods = sorted({r["method"] for r in runs})
    sources = [root / "EMBC21.npz", root / "EPFL.npz", root / "results.json",
               ROOT / "scripts/ppg_formal_denoising.py", ROOT / "scripts/ppg_temporal_adapters.py",
               ROOT / "scripts/ppg_beat_models.py"]
    out = {
        "purpose": "Read-only diagnostic of already saved data and predictions; no model forward, fitting, new scored model evaluation or edits to formal outputs. Alternative resampling is a preprocessing-only diagnostic, not a replacement formal result.",
        "date": "2026-11-06", "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "metric_note": "NRMSE=sqrt(mean((prediction-target)^2))/sqrt(mean(target^2)) over central six seconds; zero waveform has exactly NRMSE=1. Arithmetic mean NRMSE is not error on an averaged waveform. Prediction ratios and gains computed per window then averaged seeds/heads, then averaged within people; summary includes distribution of those person means.",
        "fit_epochs": {}, "prediction_diagnostics": {}, "reference_diagnostics": {}, "resampling_diagnostic": {}}
    for method in methods:
        rows = [r for r in runs if r["method"] == method]
        out["fit_epochs"][method] = {
            "best_epochs": [r["best_epoch"] for r in rows],
            "executed_epochs": [len(r["validation_curve"]) for r in rows],
            "best_at_cap20": sum(r["best_epoch"] == 20 for r in rows),
            "cap_reached": sum(len(r["validation_curve"]) == 20 for r in rows),
            "best_at_epoch1": sum(r["best_epoch"] == 1 for r in rows),
            "parameters": rows[0]["trainable_parameters"]}
    for ds in ["EMBC21", "EPFL"]:
        d = np.load(root / (ds + ".npz"))
        meta = json.loads((root / (ds + ".json")).read_text())
        y = d["target"]
        people = sorted(set(d["subjects"]))
        base = {name: diagnostic(y, d[key]) for name, key in [("NoisyInput", "x64"), ("FixedBandpass", "baseline")]}
        if ds == "EMBC21":
            strata = {"all": np.ones(len(y), bool), **{
                f"generated_{snr}dB": np.array([r["noise_snr_db"] == snr for r in meta["rows"]]) for snr in [-5, 5, 15]}}
        else:
            ids = np.array([r["record_id"] for r in meta["rows"]])
            nrmse = base["NoisyInput"]["nrmse"]
            copy_snr = {i: float(np.mean(-20*np.log10(nrmse[ids == i]))) for i in set(ids)}
            strata = {"all": np.ones(len(y), bool), **{
                label: np.isin(ids, [i for i, snr in copy_snr.items() if lo <= snr < hi])
                for label, lo, hi in [("lt0", -np.inf, 0), ("0to10", 0, 10), ("10to20", 10, 20), ("ge20", 20, np.inf)]}}
            out["epfl_input_near_identity"] = {
                "nrmse": stats(nrmse), "windows_nrmse_lt_1e-5": int(np.sum(nrmse < 1e-5)),
                "windows_nrmse_lt_1e-3": int(np.sum(nrmse < 1e-3)),
                "copy_mean_snr_db": stats(list(copy_snr.values())),
                "near_zero_copy_ids": sorted({ids[i] for i in np.flatnonzero(nrmse < 1e-5)})}
        allmetrics = dict(base)
        for method in methods:
            arrays = {k: np.zeros(len(y)) for k in base["NoisyInput"]}
            counts = np.zeros(len(y))
            for r in runs:
                if r["method"] != method:
                    continue
                file = root / r["run_folder"] / ("predictions.npz" if ds == "EMBC21" else "epfl_predictions.npz")
                with np.load(file) as z:
                    idx = z["indices"] if ds == "EMBC21" else np.arange(len(y))
                    m = diagnostic(y[idx], z["prediction"])
                    for k, v in m.items():
                        arrays[k][idx] += v
                    counts[idx] += 1
            assert (counts > 0).all()
            allmetrics[method] = {k: v/counts for k, v in arrays.items()}
        out["prediction_diagnostics"][ds] = {}
        for name, m in allmetrics.items():
            out["prediction_diagnostics"][ds][name] = {}
            for label, mask in strata.items():
                item = {"pairs": int(mask.sum()), "people": len(people), "metrics": {}}
                for k, v in m.items():
                    perperson = [float(v[mask & (d["subjects"] == p)].mean()) for p in people]
                    item["metrics"][k] = {"window_mean": float(v[mask].mean()),
                                          "person_mean_distribution": stats(perperson),
                                          "person_means": dict(zip(people, perperson))}
                item["input_snr_db_mean"] = float((-20*np.log10(base["NoisyInput"]["nrmse"][mask])).mean())
                out["prediction_diagnostics"][ds][name][label] = item
        unique = np.arange(0, len(y), 3) if ds == "EMBC21" else np.arange(9)
        yy = y[unique].astype(float)
        raw = yy*d["noisy_std"][unique, None]
        filt = sosfiltfilt(butter(2, [.5, 8], fs=64, btype="bandpass", output="sos"), yy, axis=-1)
        score = diagnostic(yy, filt)["nrmse"]
        s = raw[:, 64:448]
        s -= s.mean(axis=1, keepdims=True)
        ft = np.fft.rfft(s, axis=1)
        freq = np.fft.rfftfreq(s.shape[1], 1/64)
        e = np.abs(ft)**2
        e[:, 1:-1] *= 2
        bands = {name: e[:, mask].sum(axis=1)/e.sum(axis=1) for name, mask in [
            ("below0.5", (freq > 0) & (freq < .5)), ("0.5to8", (freq >= .5) & (freq <= 8)), ("above8", freq > 8)]}
        out["reference_diagnostics"][ds] = {
            "unique_reference_windows": len(unique), "clean_reference_filter_nrmse": stats(score),
            "central_six_seconds_fft_energy_fraction": {k: stats(v) for k, v in bands.items()},
            "spectrum_note": "Rectangular six-second FFT, centered inside scoring interval; descriptive energy fractions with spectral leakage, not physiological band estimates."}
        if ds == "EMBC21":
            out["reference_diagnostics"][ds]["per_person"] = {p: {
                "clean_filter_nrmse": float(score[d["subjects"][unique] == p].mean()),
                "central_fft_fraction_below0.5": float(bands["below0.5"][d["subjects"][unique] == p].mean())} for p in people}
    meta = json.loads((root / "EMBC21.json").read_text())
    raws, legacy, cf, line, means, stds, linerr = [], [], [], [], [], [], []
    for i in range(0, len(meta["rows"]), 3):
        r = meta["rows"][i]
        rec = r["record_id"].split("_", 1)[1]
        arr = loadmat(ROOT / "datasets/embc21/PPG_ACC_dataset" / r["subject"] / (rec + "_ppg.mat"))["PPG"]
        a = int(r["start_s"]*400)
        raw = -arr[a:a+3200, 1].astype(float)
        raws.append(raw)
        means.append(raw.mean())
        stds.append(raw.std())
        l = resample_poly(raw, 5, 16)
        l -= l.mean()
        c = resample_poly(raw-raw.mean(), 5, 16)
        c -= c.mean()
        z = resample_poly(raw, 5, 16, padtype="line")
        z -= z.mean()
        legacy.append(l)
        cf.append(c)
        line.append(z)
        ones = resample_poly(np.ones_like(raw), 5, 16)
        ones -= ones.mean()
        linerr.append(np.max(np.abs((l-c)-raw.mean()*ones)))
    for name, arr, fs in [
        ("native_centered", np.array(raws)-np.array(means)[:, None], 400),
        ("formal_resampling", np.array(legacy), 125),
        ("DC_removed_before_resampling", np.array(cf), 125),
        ("line_padding_before_centering", np.array(line), 125)]:
        arr64 = resample_poly(arr, 64, fs, axis=-1)
        arr64 -= arr64.mean(axis=1, keepdims=True)
        filt = sosfiltfilt(butter(2, [.5, 8], fs=64, btype="bandpass", output="sos"), arr64, axis=-1)
        out["resampling_diagnostic"][name] = {
            "source_rms": stats(np.sqrt(np.mean(arr*arr, axis=1))),
            "central64_rms": stats(np.sqrt(np.mean(arr64[:, 64:448]**2, axis=1))),
            "clean_reference_filter_nrmse": stats(diagnostic(arr64, filt)["nrmse"])}
    out["resampling_diagnostic"]["native_DC_absolute"] = stats(np.abs(means))
    out["resampling_diagnostic"]["native_std"] = stats(stds)
    out["resampling_diagnostic"]["absolute_DC_to_native_std_ratio"] = stats(np.abs(means)/stds)
    for name, arr in [("DC_removed_before_resampling", np.array(cf)), ("line_padding_before_centering", np.array(line))]:
        delta = np.array(legacy)-arr
        ratio = np.mean(np.array(legacy)**2, axis=1)/np.mean(arr**2, axis=1)
        item = out["resampling_diagnostic"][name]
        item["squared_difference_to_formal_fraction_of_formal_reference_energy"] = stats(np.mean(delta**2, axis=1)/np.mean(np.array(legacy)**2, axis=1))
        item["formal_to_alternative_full125_energy_ratio"] = stats(ratio)
        item["formal_synthetic_noise_SNR_relative_alternative_full125_energy_shift_db"] = stats(-10*np.log10(ratio))
    out["feature_shapes"] = {m: [list(np.load(root / "features" / (m+"__seed42") / f"stage{i}.npy", mmap_mode="r").shape) for i in range(6)] for m in ["AnyPPG", "PaPaGei-S"]}
    l = resample_poly(legacy, 64, 125, axis=-1)
    l -= l.mean(axis=1, keepdims=True)
    c = resample_poly(cf, 64, 125, axis=-1)
    c -= c.mean(axis=1, keepdims=True)
    band = butter(2, [.5, 8], fs=64, btype="bandpass", output="sos")
    lf = sosfiltfilt(band, l, axis=-1)
    cfilt = sosfiltfilt(band, c, axis=-1)
    den = np.mean(c[:, 64:448]**2, axis=1)
    def four_stats(x):
        return {k: v for k, v in stats(x).items() if k in ("mean", "median", "min", "max")}
    out["resampling_diagnostic"]["DC_padding_linearity_check"] = {
        "formula": "formal125 - centerFirst125 = native_DC_mean * (resample_poly(ones(3200),5,16) - mean(resample_poly(ones(3200),5,16)))",
        "max_absolute_formula_error": float(np.max(linerr)),
        "central6s_target_difference_NRMSE_against_centerFirst": four_stats(np.sqrt(np.mean((l[:, 64:448]-c[:, 64:448])**2, axis=1)/den)),
        "central6s_fixed_filter_output_difference_NRMSE_against_centerFirst": four_stats(np.sqrt(np.mean((lf[:, 64:448]-cfilt[:, 64:448])**2, axis=1)/den))}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Refuse to overwrite any existing diagnostic")
    report = build()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        print(json.dumps({"output": str(args.output), "sha256": sha(args.output)}))
