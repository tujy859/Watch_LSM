import urllib.request
import re
import os

url = 'https://physionet.org/content/wrist/1.0.0/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        # find files
        files = re.findall(r'href=[\"\']([a-zA-Z0-9_\-\.]+\.(?:csv|hea|dat|mat|txt|parquet))[\"\']', html)
        print("Files in wrist dataset:", set(files))
except Exception as e:
    print("Error:", e)
