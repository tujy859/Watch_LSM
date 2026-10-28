import os
import json
import gdown
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL = "https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV"
OUT_MANIFEST = str(PROJECT_ROOT / "datasets/pulsedb/manifest_pulsedb.json")
TARGET_DIR = str(PROJECT_ROOT / "datasets/pulsedb")

def build_manifest():
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    if os.path.exists(OUT_MANIFEST):
        with open(OUT_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = [{"id": item["id"], "path": item["path"].replace("\\", "/")} for item in data]
        with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Manifest already exists with {len(data)} items.")
        return data

    print("Fetching file list from Google Drive folder...")
    files = gdown.download_folder(URL, output=TARGET_DIR, quiet=True, skip_download=True)
    manifest = []
    for f in files:
        manifest.append({
            "id": f.id,
            "path": f.path
        })
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} items to {OUT_MANIFEST}.")
    return manifest

if __name__ == "__main__":
    build_manifest()
