import os
import re

analyses_dir = 'papers/analyses'
for fname in os.listdir(analyses_dir):
    if not fname.endswith('_raw.txt'):
        continue
    pname = fname.replace('_raw.txt', '')
    fpath = os.path.join(analyses_dir, fname)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    gh_matches = re.findall(r'github\.com/[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+', text)
    zenodo_matches = re.findall(r'zenodo\.org/records?/[0-9]+', text)
    huggingface_matches = re.findall(r'huggingface\.co/[a-zA-Z0-9_\-\./]+', text)
    physionet_matches = re.findall(r'physionet\.org/[a-zA-Z0-9_\-\./]+', text)
    
    print(f"=== {pname} ===")
    if gh_matches:
        print("  GitHub:", set(gh_matches))
    if zenodo_matches:
        print("  Zenodo:", set(zenodo_matches))
    if huggingface_matches:
        print("  HuggingFace:", set(huggingface_matches))
    if physionet_matches:
        print("  PhysioNet:", set(physionet_matches))
