"""Validate the EPFL paired resource and run a fixed filtering example.

One underlying resting person cannot establish cross-person generalization.
No supervised model is fitted here; noisy copies are never split as people.
"""
from pathlib import Path
import json
import re
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ppg_hr_poc_features import sha256, write_json


def load_vector(path, key):
    value = np.asarray(loadmat(path)[key], dtype=float).reshape(-1)
    if value.shape != (18001,) or not np.isfinite(value).all():
        raise ValueError("Expected one finite 72-second signal including its endpoint")
    return value


def waveform_metrics(reference, prediction):
    error = np.asarray(prediction)-np.asarray(reference)
    signal_energy = np.dot(reference, reference)
    error_energy = np.dot(error, error)
    if signal_energy <= 0 or error_energy <= 0:
        raise ValueError("Need nonconstant reference and nonzero error for this pilot")
    return {"mae":float(np.abs(error).mean()), "rmse":float(np.sqrt(np.mean(error**2))),
            "snr_db":float(10*np.log10(signal_energy/error_energy)),
            "correlation":float(np.corrcoef(reference,prediction)[0,1])}


def main():
    source=ROOT/"datasets/epfl_synthetic/extracted/DataPPGwNoise/Datasets"
    reference_path=source/"GoodSignals/PPG/ppg.mat"
    reference=load_vector(reference_path,"ppg")
    ecg=load_vector(source/"GoodSignals/ECG/ecg.mat","ecg")
    files=sorted((source/"NoisySignals/PPG").glob("*.mat"),
                 key=lambda p:tuple(map(int,re.findall(r"\d+",p.stem))))
    if len(files)!=340:
        raise ValueError("Unexpected EPFL archive version")
    for path in files:
        load_vector(path,"noise_signal")
    # Fixed low/middle/high positions in the filename order, never chosen by outcome.
    chosen=[files[i] for i in (5,170,335)]
    rows=[]
    arrays={"clean":reference,"ecg":ecg}
    band=butter(2,[.5,8],btype="bandpass",fs=250,output="sos")
    # Exclude the fixed first/last two seconds to reduce filter boundary effects.
    scoring=slice(500,17500)
    clean_ac=reference-reference.mean()
    for path in chosen:
        noisy=load_vector(path,"noise_signal")
        centered=noisy-noisy.mean()
        filtered=sosfiltfilt(band,noisy)
        arrays[path.stem]=noisy
        for method,prediction in [("mean_centered_input",centered),("fixed_bandpass_0.5_to_8_Hz",filtered)]:
            rows.append({"file":path.name,"method":method,**waveform_metrics(clean_ac[scoring],prediction[scoring]),
                         "source_sha256":sha256(path)})
    local=ROOT/".local/ppg-expansion-pilot/denoising"
    local.mkdir(parents=True,exist_ok=True)
    np.savez(local/"examples.npz",**arrays)
    report={"status":"paired_data_and_traditional_filter_pipeline_verified", "foundation_model_evaluated":False,
        "source":"https://zenodo.org/records/4981140","license":"CC-BY-4.0","subjects":1,
        "underlying_clean_signals":1,"noisy_copies":340,"fs":250,"samples":18001,"elapsed_signal_seconds":72,
        "validated_noisy_shapes_and_finite_values":340,"example_noisy_files":[p.name for p in chosen],
        "reference":"Original clean waveform minus its full-record mean; no reference-guided time alignment or gain fitting",
        "scoring":"Fixed 2 to 70 seconds; mean-centered input versus fixed bandpass output; arbitrary amplitude units",
        "training":False,"split":"None; copies of one clean waveform do not represent independent people",
        "limitations":["Resting single person and synthetic noise; cannot establish real-motion or population generalization",
                       "Filename rounded noise intensity is not a measured SNR; metrics computed from actual paired arrays",
                       "Only classical filtering example; foundation waveform decoder remains a separate planned task"],
        "clean_sha256":sha256(reference_path),"code_sha256":sha256(Path(__file__)),"examples":rows}
    target=ROOT/"docs/results/ppg_expansion_pilot/denoising_data_pilot.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    write_json(target,report)
    print(json.dumps({"verified_noisy_pairs":340,"traditional_examples":len(rows),"training":False}))


if __name__=="__main__":
    main()
