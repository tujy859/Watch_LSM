import urllib.request
import re
import os
import sys

def get_arxiv_info(arxiv_id):
    url = f"https://arxiv.org/abs/{arxiv_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            title_m = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
            title = title_m.group(1).strip() if title_m else 'Unknown'
            title = re.sub(r'^\[.*?\]\s*', '', title)
            
            abs_m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', html, re.DOTALL)
            abstract = 'Unknown'
            if abs_m:
                abstract = re.sub(r'<span class="descriptor">Abstract:</span>', '', abs_m.group(1)).strip()
                abstract = re.sub(r'<.*?>', '', abstract).strip()
            
            return {
                'id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }
    except Exception as e:
        return {'id': arxiv_id, 'error': str(e)}

def download_pdf(pdf_url, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        print(f"File {output_path} already exists ({os.path.getsize(output_path)} bytes).")
        return True
    print(f"Downloading {pdf_url} to {output_path}...")
    req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        print(f"Successfully downloaded {output_path} ({os.path.getsize(output_path)} bytes)")
        return True
    except Exception as e:
        print(f"Failed to download {output_path}: {e}")
        return False

if __name__ == '__main__':
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else '2405.20759'
    info = get_arxiv_info(arxiv_id)
    print("Title:", info.get('title'))
    print("Abstract:", info.get('abstract')[:300] + "...")
