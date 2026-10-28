import os
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 315300817
URL = "https://zenodo.org/records/17345536/files/pulseppg_model_weights.zip?download=1"
OUT_PATH = str(PROJECT_ROOT / "models" / "weights" / "pulseppg" / "pulseppg_model_weights.zip")

def download_loop():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    attempt = 1
    while True:
        if os.path.exists(OUT_PATH):
            cur_size = os.path.getsize(OUT_PATH)
            if cur_size >= TARGET_SIZE:
                print(f"Download complete: {cur_size} bytes.")
                break
            print(f"Attempt {attempt}: Resuming from {cur_size / (1024*1024):.2f} MB / {TARGET_SIZE / (1024*1024):.2f} MB ({cur_size / TARGET_SIZE * 100:.1f}%)")
        else:
            print(f"Attempt {attempt}: Starting fresh download...")

        cmd = [
            "curl.exe", "-L", "-C", "-",
            "--retry", "5",
            "--retry-delay", "2",
            "-o", OUT_PATH,
            URL
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            cur_size = os.path.getsize(OUT_PATH)
            if cur_size >= TARGET_SIZE:
                print("Download completed successfully!")
                break
        time.sleep(2)
        attempt += 1

if __name__ == "__main__":
    download_loop()
