"""Auxiliary age prediction using the same audited PPG-BP observations/folds.

Age is removed from every covariate group before it becomes the target.
Frozen feature arrays are reused only after exact waveform identity checks.
This is chronological age prediction, not a validated vascular-age measure.
"""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_health import load_prepared,fit_dataset,MODELS,RANDOM,write_json,array_hash
from scripts.ppg_hr_poc_features import sha256


def age_arrays(source):
    if source["covariate_names"]!=["age_years","male_indicator","bmi_kg_m2"]:
        raise ValueError("Unexpected PPG-BP covariate order")
    result={k:source[k] for k in ("subjects","traditional","windows","waveform_subjects","stats")}
    result["targets"]=source["covariates"][:,:1].copy()
    result["covariates"]=source["covariates"][:,1:].copy()
    return result


def main():
    output=ROOT/".local/ppg-formal-expansion/health"
    source,meta=load_prepared("PPG-BP",output)
    arrays=age_arrays(source)
    directory=output/"PPG-BP-age"; directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/"data.npz",**arrays)
    manifest={**meta,"dataset":"PPG-BP-age","target_names":["chronological_age_years"],"covariate_names":["male_indicator","bmi_kg_m2"],
              "data_npz_sha256":sha256(directory/"data.npz"),"code_sha256":sha256(Path(__file__)),
              "target_source":"Original PPG-BP age column, removed from all predictor inputs",
              "source_parent_manifest_sha256":sha256(output/"PPG-BP/manifest.json"),
              "limitations":[*meta["limitations"],"Chronological age prediction, not vascular or biological age", "Auxiliary task, not a primary health increment hypothesis", "No supervised CNN added to this auxiliary age task"]}
    write_json(directory/"manifest.json",manifest)
    features=directory/"features"; features.mkdir(exist_ok=True)
    for path in sorted((output/"PPG-BP/features").glob("*.npz")):
        evidence=json.loads(path.with_suffix(".json").read_text())
        if evidence["input_sha256"]!=array_hash(arrays["windows"]) or evidence["output_sha256"]!=sha256(path):
            raise ValueError("BP feature waveform provenance mismatch")
        shutil.copyfile(path,features/path.name)
        updated={**evidence,"data_sha256":manifest["data_npz_sha256"],"source_feature_path":path.relative_to(output).as_posix(),
                 "feature_reused_without_refitting":True}
        write_json(features/path.with_suffix(".json").name,updated)
    with threadpool_limits(limits=2):
        fit_dataset("PPG-BP-age",output,[*MODELS,RANDOM])


if __name__=="__main__":
    main()
