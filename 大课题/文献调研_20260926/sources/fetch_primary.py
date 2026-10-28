"""Download public primary sources; retain failures and never bypass access controls."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
ITEMS = {
    'VSF2022': 'https://arxiv.org/pdf/2206.12626',
    'TRF2024': 'https://www.ijcai.org/proceedings/2024/0228.pdf',
    'GCNM2023': 'https://arxiv.org/pdf/2212.06419',
    'MTLTraffic2023': 'https://file.jqyu.me/pub/traffic-prediction-with-missing-data-a-multi-task-learning-approach.pdf',
    'UATraffic2023': 'https://arxiv.org/pdf/2309.06800',
    'HiPatch2025': 'https://raw.githubusercontent.com/mlresearch/v267/main/assets/luo25r/luo25r.pdf',
    'CSDI2021': 'https://arxiv.org/pdf/2107.03502',
    'PriSTI2023': 'https://arxiv.org/pdf/2302.09746',
    'SSSD2023': 'https://arxiv.org/pdf/2208.09399',
    'TSDiff2023': 'https://arxiv.org/pdf/2307.11494',
    'CRU2022': 'https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf',
    'NeuralFlows2021': 'https://arxiv.org/pdf/2110.13040',
    'AirBayes2025': 'https://arxiv.org/pdf/2511.02175',
    'AirQualityBench2026': 'https://arxiv.org/pdf/2605.05854',
    'Survey2025': 'https://www.ijcai.org/proceedings/2025/1187.pdf',
    'NuwaTS2024': 'https://arxiv.org/pdf/2405.15317',
    'Seq2Seq2021': 'https://eprints.bournemouth.ac.uk/35723/1/ICAC2021_Accepted_version.pdf',
    'SAITS2023': 'https://arxiv.org/pdf/2202.08516',
    'GRIN2022': 'https://arxiv.org/pdf/2108.00298',
    'SPIN2022': 'https://openreview.net/pdf?id=YXHoPO33rk',
}

def fetch(item):
    key, url = item
    dest = ROOT / 'downloaded_pdf' / (key + '.pdf')
    txt = ROOT / 'downloaded_text' / (key + '.txt')
    record = {'key': key, 'url': url, 'access_date': '2026-09-26'}
    try:
        if not dest.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'AcademicLiteratureReview/1.0'})
            with urllib.request.urlopen(request, timeout=35) as response:
                data = response.read()
            if not data.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            dest.write_bytes(data)
        subprocess.run(['pdftotext', '-layout', str(dest), str(txt)], check=True, capture_output=True)
        record.update(status='downloaded', bytes=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), pdf=str(dest), text=str(txt))
    except Exception as error:
        record.update(status='unavailable', error=str(error))
    print(key, record['status'], flush=True)
    return record

if __name__ == '__main__':
    for name in ['downloaded_pdf', 'downloaded_text']:
        (ROOT / name).mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch, ITEMS.items()))
    (ROOT / 'download_manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
