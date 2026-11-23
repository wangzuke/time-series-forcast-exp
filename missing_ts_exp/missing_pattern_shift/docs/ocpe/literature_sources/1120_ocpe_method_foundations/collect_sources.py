#!/usr/bin/env python3
"""Fetch primary-source snapshots and metadata; no experiment access or writes.

Network collection is explicit, bounded and repeatable. failures are retained.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import shutil
import subprocess
import requests

ROOT = Path(__file__).resolve().parent
DATE = '2026-11-20'
SOURCES = ROOT / 'sources'

def fetch(job):
    name, url, kind = job
    record = {'name': name, 'url': url, 'kind': kind, 'date': DATE}
    try:
        r = requests.get(url, timeout=(8, 25), headers={'User-Agent': 'OCPE-reference-audit/1.0'}, allow_redirects=True)
        record.update(status_code=r.status_code, final_url=r.url, content_type=r.headers.get('Content-Type', ''))
        if r.ok:
            is_pdf = r.content.startswith(b'%PDF')
            suffix = '.pdf' if is_pdf else '.json' if 'json' in record['content_type'] else '.html' if 'html' in record['content_type'] else '.txt'
            path = SOURCES / (name + suffix)
            path.write_bytes(r.content)
            record.update(path=str(path.relative_to(ROOT)), bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest())
            if kind == 'pdf' and not is_pdf:
                record['warning'] = 'Requested PDF but response is not PDF; do not treat as full text.'
            if is_pdf and shutil.which('pdftotext'):
                text_path = path.with_suffix('.txt')
                subprocess.run(['pdftotext', '-layout', str(path), str(text_path)], check=True, capture_output=True)
                record['text_path'] = str(text_path.relative_to(ROOT))
        else:
            record['error_excerpt'] = r.text[:160]
    except Exception as e:
        record['error'] = str(e)
    return record

def main():
    SOURCES.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    jobs = []
    for ref in manifest:
        jobs.append((ref['id']+'_landing', ref['url'], 'landing'))
        jobs.append((ref['id']+'_paper', ref['pdf_url'], 'pdf'))
        if ref.get('html_url'):
            jobs.append((ref['id']+'_fulltext', ref['html_url'], 'fulltext'))
        doi = ref.get('doi', '')
        if doi and 'arXiv' not in doi:
            jobs.append((ref['id']+'_crossref', 'https://api.crossref.org/works/'+doi, 'crossref_metadata'))
    queries = [
        ('arxiv_search', 'https://export.arxiv.org/api/query?search_query=all%3A%22missingness+shift%22&start=0&max_results=10', 'database_search'),
        ('crossref_search', 'https://api.crossref.org/works?query.bibliographic=missingness%20shift%20prediction&rows=10', 'database_search'),
        ('dblp_search', 'https://dblp.org/search/publ/api?q=missingness%20shift&format=json&h=10', 'database_search'),
        ('skill_credit', 'https://arxiv.org/abs/2609.00065', 'software_credit'),
    ]
    jobs.extend(queries)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, jobs))
    (ROOT / 'source_fetch_log.json').write_text(json.dumps({'date': DATE, 'records': records}, ensure_ascii=False, indent=2))
    print(json.dumps({'requested': len(records), 'downloaded': sum('path' in r for r in records), 'failed': [{k:r[k] for k in ['name','status_code','error'] if k in r} for r in records if 'path' not in r]},ensure_ascii=False))
    # Use the skill verifier for publisher DOIs; arXiv identifiers use primary arXiv pages.
    module_path = Path('/home/wangzk/.codex/skills/literature-review/scripts/verify_citations.py')
    if module_path.is_file():
        spec = importlib.util.spec_from_file_location('ocpe_citation_verifier', module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        def verify(ref):
            ok, metadata = mod.CitationVerifier().verify_doi(ref['doi'])
            return {'id': ref['id'], 'doi': ref['doi'], 'handle_resolved': ok, 'metadata': metadata}
        refs = [r for r in manifest if r.get('doi') and 'arXiv' not in r['doi']]
        with ThreadPoolExecutor(max_workers=3) as pool:
            checks = list(pool.map(verify, refs))
        (ROOT/'doi_verification_raw.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
        print('Publisher DOI checks saved:',len(checks))

if __name__ == '__main__':
    main()
