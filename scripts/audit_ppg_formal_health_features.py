"""Re-run three fixed raw clips per health feature cache on CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.ppg_formal_health import load_prepared,extract_native_features,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/".local/ppg-formal-expansion/health")
    parser.add_argument("--shared",type=Path,default=ROOT/"docs/results/ppg_formal_expansion/health")
    args=parser.parse_args(); checks=[]
    for dataset in ("ECSMP","PPG-BP"):
        data,_=load_prepared(dataset,args.output)
        for path in sorted((args.output/dataset/"features").glob("*.npz")):
            original=json.loads(path.with_suffix(".json").read_text())
            rebuilt,meta=extract_native_features(original["model"],data["windows"][:3],data["fs"],data["stats"][:3],
                                                 device="cpu",seed=original["seed"],random_init=original["random_init"])
            with np.load(path,allow_pickle=False) as blob:
                expected=blob["window_features"][:3]
            np.testing.assert_allclose(rebuilt,expected,atol=1e-4,rtol=1e-4)
            if original["encoder_state_before"]!=meta["encoder_state_before"]:
                raise ValueError("Fresh encoder state differs")
            check={"dataset":dataset,"cache":path.name,"clips":3,"max_abs_difference":float(np.max(np.abs(rebuilt-expected))),
                   "state_sha256":meta["encoder_state_before"],"passed":True}
            checks.append(check)
            print(json.dumps(check),flush=True)
    write_json(args.shared/"feature_cpu_reforward_audit.json",{"status":"passed","rtol":1e-4,"atol":1e-4,"checks":checks})


if __name__=="__main__":
    main()
