import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def search_arxiv(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=ti:%22{urllib.parse.quote(query)}%22&max_results={max_results}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            root = ET.fromstring(resp.read())
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                id_url = entry.find('atom:id', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                results.append({'title': title, 'id': id_url.split('/')[-1], 'summary': summary})
            return results
    except Exception as e:
        print(f"Error searching {query}: {e}")
        return []

if __name__ == '__main__':
    for res in search_arxiv("Towards a General Intelligence"):
        print(res['id'], "->", res['title'])
