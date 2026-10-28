import re
from pathlib import Path

with (Path(__file__).resolve().parents[1] / "datasets/wesad/wesad.zip").open("r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print("Title:", re.findall(r"<title>(.*?)</title>", text))
urls = set(re.findall(r'href="([^"]+)"', text) + re.findall(r'action="([^"]+)"', text))
for u in sorted(urls):
    print(u)
