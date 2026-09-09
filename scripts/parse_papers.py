import os
import pypdf

pdf_dir = 'papers'
out_dir = os.path.join(pdf_dir, 'analyses')

for fname in os.listdir(pdf_dir):
    if not fname.endswith('.pdf'):
        continue
    fpath = os.path.join(pdf_dir, fname)
    txt_name = fname.replace('.pdf', '_raw.txt')
    txt_path = os.path.join(out_dir, txt_name)
    
    print(f"Extracting {fname}...")
    try:
        reader = pypdf.PdfReader(fpath)
        num_pages = len(reader.pages)
        full_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"--- Page {i+1} ---\n" + text)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(full_text))
        print(f"Extracted {num_pages} pages to {txt_path} ({os.path.getsize(txt_path)} bytes)")
    except Exception as e:
        print(f"Error extracting {fname}: {e}")
