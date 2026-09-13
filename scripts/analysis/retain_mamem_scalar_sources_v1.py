"""Retain the two successful read-only scout targets, once, within frozen budget."""
import hashlib
import json
import shutil
import signal
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/whwovy/research-spaces/califree-eeg-experiment-design/sources')
SPECS = [
    ('mamem_cedrus_setup_20260913', 'https://cedrus.com/support/stimtracker_1g/tn1468_egi.htm', '.html'),
    ('mamem_moabb_adapter_20260913', 'https://api.github.com/repos/NeuroTechX/moabb/contents/moabb/datasets/ssvep_mamem.py?ref=develop', '.json'),
]


def save(path, content):
    import os
    with path.open('xb') as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())


def run():
    for name, url, suffix in SPECS:
        target, receipt = ROOT / (name + suffix), ROOT / (name + '.receipt.json')
        started = ROOT / (name + '.started.json')
        if any(p.exists() for p in (target, receipt, started)):
            raise RuntimeError('no_retry_or_overwrite')
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-09-13T10:49:00+00:00'):
            raise RuntimeError('source_deadline')
        if shutil.disk_usage(ROOT).free < 7 * 1024**3 + 2 * 1024**2:
            raise RuntimeError('free_space_reserve')
        row = {'url': url, 'started_utc': datetime.now(timezone.utc).isoformat(), 'attempts': 1}
        save(started, json.dumps(dict(row, status='STARTED')).encode())
        def alarm(*args):
            raise TimeoutError('source_wall_timeout')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(30)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'public-research-source-verification'})
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(2 * 1024**2 + 1)
                if response.status != 200 or len(body) > 2 * 1024**2:
                    raise ValueError('status_or_body_cap')
                row.update(http_status=response.status, bytes=len(body),
                           sha256=hashlib.sha256(body).hexdigest(), final_url=response.geturl())
            save(target, body)
            row.update(status='SOURCE_RETAINED', path=str(target))
        except Exception as error:
            row.update(status='FAILED_NO_RETRY', error_type=type(error).__name__)
        finally:
            signal.alarm(0)
        row['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save(receipt, (json.dumps(row, indent=2) + '\n').encode())
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    run()
