import urllib.request
import re

url = 'https://sites.google.com/site/researchbyzhang/data'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        # find hrefs
        hrefs = re.findall(r'href=[\"\'](https?://[^\s\"\']+)[\"\']', html)
        for h in hrefs:
            if any(k in h.lower() for k in ['troika', 'spcup', 'ieee', 'ppg', 'zip', 'mat']):
                print(h)
except Exception as e:
    print('Error:', e)
