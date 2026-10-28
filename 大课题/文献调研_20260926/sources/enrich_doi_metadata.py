import concurrent.futures
import json
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]

def fetch(p):
    result = dict(id=p['id'], doi=p['doi'])
    try:
        r = requests.get('https://api.crossref.org/works/'+p['doi'], timeout=18)
        r.raise_for_status()
        m = r.json()['message']
        result.update(status='ok', metadata={k:m[k] for k in ['title','author','container-title','published','published-print','published-online','volume','issue','page','article-number','DOI','URL'] if k in m})
    except Exception as e:
        result.update(status='unavailable', error=str(e))
    print(p['id'], result['status'], flush=True)
    return result

if __name__ == '__main__':
    rows=[p for p in json.loads((ROOT/'papers.json').read_text()) if p['doi']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(fetch, rows))
    (ROOT/'sources/crossref_metadata.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
