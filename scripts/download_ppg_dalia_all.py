import os
import time
import subprocess

FILES = [
    ("ppg_dalia_sitting_data.mat", 35095812),
    ("ppg_dalia_walking_data.mat", 40273752),
    ("ppg_dalia_cycling_data.mat", 31577565),
    ("ppg_dalia_stair_climbing_data.mat", 29156227),
    ("ppg_dalia_working_data.mat", 66196769),
    ("ppg_dalia_table_soccer_data.mat", 19800136),
    ("ppg_dalia_car_driving_data.mat", 55829057),
    ("ppg_dalia_lunch_break_data.mat", 107308927),
]

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://zenodo.org/api/records/12793711/files/{filename}/content"
OUT_DIR = str(PROJECT_ROOT / "datasets" / "ppg_dalia")

def download_file(filename, expected_size):
    out_path = os.path.join(OUT_DIR, filename)
    url = BASE_URL.format(filename=filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= expected_size:
        print(f"[OK] {filename} already complete.")
        return

    attempt = 1
    while True:
        cur_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        if cur_size >= expected_size:
            print(f"[OK] {filename} finished ({cur_size} bytes).")
            break
        print(f"[{filename}] Attempt {attempt}: Resuming from {cur_size/(1024*1024):.2f}/{expected_size/(1024*1024):.2f} MB")
        cmd = ["curl.exe", "-L", "-C", "-", "--retry", "5", "--retry-delay", "2", "-o", out_path, url]
        subprocess.run(cmd)
        time.sleep(1)
        attempt += 1

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn, sz in FILES:
        download_file(fn, sz)
    print("All PPG-DaLiA files downloaded successfully!")

if __name__ == "__main__":
    main()
