"""One attempt at the queued exact release attachment; no dataset arrays."""
import hashlib
import json
import os
import shutil
import signal
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from fetch_mobilebci_public_pair_v1 import BoundedRedirect, Stop, copy_payload, require


def main():
    directory = Path('/home/whwovy/research-spaces/califree-eeg-experiment-design/sources')
    target = directory / 'mamem_i_DataAcquisitionDetails_20260913.pdf'
    partial = target.with_suffix('.pdf.part')
    receipt = target.with_suffix('.receipt.json')
    require(not any(p.exists() for p in (target, partial, receipt)), 'one_attempt_fresh_only')
    require(datetime.now(timezone.utc).timestamp() < datetime.fromisoformat('2026-09-13T03:08:00+00:00').timestamp(), 'expired')
    require(shutil.disk_usage(directory).free > 8*1024**3+16*1024**2, 'disk_reserve')
    result = {'url': 'https://ndownloader.figshare.com/files/3687771',
              'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': 1}
    def stop(*args):
        raise Stop('30second_wall_deadline')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(30)
    try:
        redirect = BoundedRedirect(5)
        with urllib.request.build_opener(redirect).open(result['url'], timeout=30) as source:
            result['http_status'] = source.status
            require(source.status == 200, 'expected200')
            with partial.open('xb') as out:
                size, md5, sha = copy_payload(source, out, 276431, 65536)
                out.flush()
                os.fsync(out.fileno())
        result.update(bytes=size, md5=md5, sha256=sha, redirects=redirect.count)
        require(md5 == 'a0ed5f0d7130f4fa662ee3657ff31036', 'md5_mismatch')
        with partial.open('rb') as file:
            require(file.read(5) == b'%PDF-', 'not_PDF')
        os.link(partial, target)
        partial.unlink()
        result.update(status='PDF_DOWNLOADED_VERIFIED', path=str(target))
    except Exception as error:
        result.update(status='STOPPED_NO_RETRY', error_type=type(error).__name__,
                      reason=str(error) if isinstance(error, Stop) else 'details_not_exported')
    finally:
        signal.alarm(0)
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    with receipt.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
