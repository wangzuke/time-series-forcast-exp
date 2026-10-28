"""Run the installed literature-review skill's DOI verifier; preserve failures."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('citation_skill', '/home/wangzk/.codex/skills/literature-review/scripts/verify_citations.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def check(p):
    ok, metadata = MOD.CitationVerifier().verify_doi(p['doi'])
    record = dict(id=p['id'], doi=p['doi'], resolver_ok=ok, crossref_metadata=metadata)
    print(p['id'], ok, metadata.get('title', metadata.get('error', 'metadata unavailable'))[:100], flush=True)
    return record

if __name__ == '__main__':
    rows = [p for p in json.loads((ROOT/'papers.json').read_text()) if p['doi']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(check, rows))
    (ROOT/'sources/citation_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
