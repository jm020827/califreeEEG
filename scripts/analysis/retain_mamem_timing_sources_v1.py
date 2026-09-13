"""Retain two already-successful scout sources once; no credentials or raw data."""
import hashlib
import json
import signal
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/whwovy/research-spaces/califree-eeg-experiment-design/sources')
SPECS = [
    ('mamem_cedrus_egi_20260913', 'https://cedrus.com/blog/update-for-egi-users.htm', '.html'),
    ('mamem_author_comments_20260913', 'https://api.github.com/repos/MAMEM/eeg-processing-toolbox/issues/comments?per_page=100', '.json'),
]


def run():
    for name, url, suffix in SPECS:
        target, receipt = ROOT / (name + suffix), ROOT / (name + '.receipt.json')
        if target.exists() or receipt.exists():
            raise RuntimeError('no_retry_or_overwrite')
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-09-13T03:08:00+00:00'):
            raise RuntimeError('deadline')
        row = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': 1}
        def alarm(*args):
            raise TimeoutError('source_wall_timeout')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(30)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'public-research-source-verification'})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(2*1024**2+1)
                if response.status != 200 or len(body) > 2*1024**2:
                    raise ValueError('status_or_body_cap')
                row.update(http_status=response.status, bytes=len(body),
                           sha256=hashlib.sha256(body).hexdigest(), final_url=response.geturl())
            with target.open('xb') as out:
                out.write(body)
            row.update(status='SOURCE_RETAINED', path=str(target))
        except Exception as error:
            row.update(status='FAILED_NO_RETRY', error_type=type(error).__name__)
        finally:
            signal.alarm(0)
        row['completed_at'] = datetime.now(timezone.utc).isoformat()
        with receipt.open('x') as out:
            json.dump(row, out, indent=2)
            out.write('\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    run()
