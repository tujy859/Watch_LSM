import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import vitaldb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def extract_case_windows(caseid, target_fs=125, window_sec=60, min_std=1e-4):
    """
    Load continuous PPG (SNUADC/PLETH) and ECG Lead II (SNUADC/ECG_II) from VitalDB
    and segment into continuous synchronized windows.
    """
    track_names = ['SNUADC/PLETH', 'SNUADC/ECG_II']
    try:
        data = vitaldb.load_case(caseid, track_names, interval=1.0 / target_fs)
    except Exception as e:
        print(f"Case {caseid}: failed to load ({e})")
        return None, None

    if data is None or len(data) == 0:
        return None, None

    # Check for valid points
    ppg = data[:, 0]
    ecg = data[:, 1]
    valid_mask = (~np.isnan(ppg)) & (~np.isnan(ecg))
    if not np.any(valid_mask):
        return None, None

    # Find contiguous valid blocks
    # valid_mask is boolean array. Find start and end indices of contiguous True blocks
    diff = np.diff(valid_mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    if valid_mask[0]:
        starts = np.r_[0, starts]
    ends = np.where(diff == -1)[0] + 1
    if valid_mask[-1]:
        ends = np.r_[ends, len(valid_mask)]

    window_len = int(window_sec * target_fs)
    ppg_windows = []
    ecg_windows = []

    for s, e in zip(starts, ends):
        block_len = e - s
        if block_len < window_len:
            continue
        n_windows = block_len // window_len
        for i in range(n_windows):
            ws = s + i * window_len
            we = ws + window_len
            w_ppg = ppg[ws:we]
            w_ecg = ecg[ws:we]

            # Quality checks: not flatline
            if np.std(w_ppg) < min_std or np.std(w_ecg) < min_std:
                continue

            ppg_windows.append(w_ppg)
            ecg_windows.append(w_ecg)

    if not ppg_windows:
        return None, None

    return np.array(ppg_windows, dtype=np.float32), np.array(ecg_windows, dtype=np.float32)

def process_vitaldb(case_list, output_dir, window_sec=60, target_fs=125):
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
        done_cases = set(manifest_df["caseid"].tolist())
    else:
        done_cases = set()

    print(f"Processing {len(case_list)} cases into {window_sec}s windows @ {target_fs}Hz...")
    total_windows = 0
    t0 = time.time()

    for idx, caseid in enumerate(case_list):
        if caseid in done_cases:
            continue
        case_file = os.path.join(output_dir, f"case_{caseid:04d}.npz")
        ppg_w, ecg_w = extract_case_windows(caseid, target_fs=target_fs, window_sec=window_sec)
        if ppg_w is not None and len(ppg_w) > 0:
            np.savez_compressed(
                case_file,
                ppg=ppg_w,
                ecg=ecg_w,
                caseid=caseid,
                fs=target_fs,
                window_sec=window_sec
            )
            n_w = len(ppg_w)
            total_windows += n_w
            dur_min = n_w * (window_sec / 60.0)
            manifest_rows.append({
                "caseid": caseid,
                "windows": n_w,
                "duration_min": dur_min,
                "file": f"case_{caseid:04d}.npz"
            })
            print(f"[{idx+1}/{len(case_list)}] Case {caseid:04d}: Extracted {n_w} windows ({dur_min:.1f} mins)")
        else:
            print(f"[{idx+1}/{len(case_list)}] Case {caseid:04d}: No valid continuous segments")

        # Periodically update manifest
        if manifest_rows and len(manifest_rows) % 5 == 0:
            df = pd.DataFrame(manifest_rows)
            df.to_csv(manifest_path, index=False, mode='a', header=not os.path.exists(manifest_path))
            manifest_rows = []

    if manifest_rows:
        df = pd.DataFrame(manifest_rows)
        df.to_csv(manifest_path, index=False, mode='a', header=not os.path.exists(manifest_path))

    elapsed = time.time() - t0
    print(f"Extraction finished in {elapsed:.1f}s. Extracted total {total_windows} windows (~{total_windows*(window_sec/60.0):.1f} mins).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract continuous 1-min windows from VitalDB")
    parser.add_argument("--cases", type=int, default=30, help="Number of cases to extract")
    parser.add_argument("--window-sec", type=int, default=60, help="Window length in seconds")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "datasets/vitaldb_1min"), help="Output directory (current extractor saves PPG/ECG only)")
    args = parser.parse_args()

    # Query cases with paired tracks
    print("Querying VitalDB cases with paired PPG and ECG...")
    available_cases = vitaldb.find_cases(['SNUADC/PLETH', 'SNUADC/ECG_II'])
    target_cases = available_cases[:args.cases]
    process_vitaldb(target_cases, args.output, window_sec=args.window_sec)
