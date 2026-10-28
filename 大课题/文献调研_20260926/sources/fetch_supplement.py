import concurrent.futures
import json
from fetch_primary import fetch, ROOT

ITEMS = {
    'WindFCS2024': 'https://arxiv.org/pdf/2203.08252',
    'TACTiS2022': 'https://proceedings.mlr.press/v162/drouin22a/drouin22a.pdf',
    'TACTiS2_2024': 'https://proceedings.iclr.cc/paper_files/paper/2024/file/63796148c99205adb0fcac069cc714d4-Paper-Conference.pdf',
    'MOSES2026': 'https://openreview.net/pdf?id=awWi4hJI7O',
    'ProFITi2025': 'https://ojs.aaai.org/index.php/AAAI/article/download/35494/37649',
    'TSIBench2024': 'https://arxiv.org/pdf/2406.12747',
    'Physiome2025': 'https://arxiv.org/pdf/2502.07489',
    'PriSTI2023': 'https://arxiv.org/pdf/2302.09746',
    'SPIN2022': 'https://openreview.net/pdf?id=YXHoPO33rk',
}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, ITEMS.items()))
    (ROOT / 'download_supplement_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
