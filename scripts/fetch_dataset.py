#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.registry import AssetRegistry

BETA_FIGSHARE_ARTICLE_ID = 12264401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["beta", "wang", "wearable", "openbci"])
    parser.add_argument("--assets-config", default="configs/assets.yaml")
    parser.add_argument("--method", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--subjects", default=None, help="Comma-separated subject ids, e.g. 1,2,3. Default: all.")
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned action only.")
    parser.add_argument("--probe-remote", action="store_true", help="Probe public source metadata without downloading data.")
    args = parser.parse_args()
    registry = AssetRegistry.from_yaml(args.assets_config, strict_env=False)
    cfg = registry.dataset(args.dataset)
    if args.dataset == "openbci":
        print("openbci is local private data. Use scripts/openbci_convert.py on a local session folder.")
        return
    if args.dataset == "wearable":
        print(
            "Wearable SSVEP is treated as manual/Figshare source. Download on storage server from:\n"
            "  https://figshare.com/articles/dataset/An_Open_Dataset_for_Wearable_SSVEP-Based_Brain-Computer_Interfaces/13560281\n"
            "Then place files under EEG_DATA_ROOT/raw/wearable and run verify/prepare."
        )
        return
    if args.dataset == "beta":
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/beta")
        article_id = int(cfg.get("figshare_article_id") or BETA_FIGSHARE_ARTICLE_ID)
        article = _figshare_article(article_id)
        mat_files = _select_beta_files(article, _parse_subjects(args.subjects))
        total_size = sum(int(file.get("size", 0)) for file in mat_files)
        print(f"BETA source: Figshare article {article_id}")
        print(f"Selected .mat files: {len(mat_files)} ({_format_bytes(total_size)})")
        if args.probe_remote or args.dry_run:
            _print_figshare_plan(mat_files, raw_dir)
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "figshare_article.json").open("w", encoding="utf-8") as f:
            json.dump(article, f, indent=2)
        downloaded = []
        for file in mat_files:
            dst = raw_dir / str(file["name"])
            _download_figshare_file(file, dst, force_update=args.force_update)
            downloaded.append(dst)
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} BETA .mat file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset beta "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/beta_v1 --config configs/data/beta.yaml"
        )
        return
    if args.dataset == "wang":
        if args.dry_run:
            print("Would use MOABB Wang2016 loader if installed, otherwise manual raw_dir.")
            return
        try:
            from moabb.datasets import Wang2016
        except Exception:
            raise SystemExit(
                "MOABB is not installed. Install with:\n"
                "  pip install -e '.[moabb]'\n"
                "or manually place Wang2016 raw files under $EEG_DATA_ROOT/raw/wang."
            )
        dataset = Wang2016()
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/wang")
        raw_dir.mkdir(parents=True, exist_ok=True)
        subjects = _parse_subjects(args.subjects) or list(getattr(dataset, "subject_list", []))
        if not subjects:
            raise SystemExit("Could not determine Wang2016 subject list from MOABB.")
        print(f"Fetching Wang2016 subjects={subjects} into raw_dir={raw_dir}")
        downloaded = []
        for subject in subjects:
            paths = _moabb_data_path(dataset, subject, raw_dir, force_update=args.force_update)
            downloaded.extend(paths)
            print(f"subject {subject}: {len(paths)} file(s)")
        linked = _link_downloaded_files(downloaded, raw_dir)
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} path(s). Linked {linked} file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset wang "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/wang_v1 --config configs/data/wang.yaml"
        )


def _parse_subjects(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _path_arg(raw: str | None, default: str) -> Path:
    value = str(raw or default)
    value = re.sub(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    return Path(os.path.expandvars(value)).expanduser()


def _figshare_article(article_id: int) -> dict[str, Any]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_beta_files(article: dict[str, Any], subjects: list[int] | None) -> list[dict[str, Any]]:
    files = []
    selected = set(subjects or [])
    for file in article.get("files", []):
        name = str(file.get("name", ""))
        match = re.fullmatch(r"S(\d+)\.mat", name, flags=re.IGNORECASE)
        if not match:
            continue
        subject = int(match.group(1))
        if selected and subject not in selected:
            continue
        files.append({**file, "subject": subject})
    files.sort(key=lambda item: int(item["subject"]))
    if not files:
        raise SystemExit("No matching BETA S*.mat files found in the Figshare article.")
    return files


def _print_figshare_plan(files: list[dict[str, Any]], raw_dir: Path) -> None:
    for file in files[:10]:
        print(f"  {file['name']}: {_format_bytes(int(file.get('size', 0)))}")
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")
    print(f"Target raw_dir: {raw_dir}")
    print("Run without --dry-run/--probe-remote to download.")


def _download_figshare_file(file: dict[str, Any], dst: Path, *, force_update: bool) -> None:
    expected_size = int(file.get("size", 0) or 0)
    expected_md5 = str(file.get("computed_md5") or file.get("supplied_md5") or "")
    if dst.exists() and not force_update:
        if expected_size and dst.stat().st_size == expected_size:
            if not expected_md5 or _md5(dst) == expected_md5:
                print(f"cache ok: {dst}")
                return
        raise SystemExit(
            f"Existing file looks incomplete or mismatched: {dst}\n"
            "Rerun with --force-update to replace it."
        )
    if dst.exists() and force_update:
        dst.unlink()
    url = str(file["download_url"])
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading {file['name']} -> {dst}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if expected_size and tmp.stat().st_size != expected_size:
        raise SystemExit(
            f"Downloaded size mismatch for {dst.name}: got {tmp.stat().st_size}, expected {expected_size}"
        )
    if expected_md5 and _md5(tmp) != expected_md5:
        raise SystemExit(f"Downloaded md5 mismatch for {dst.name}")
    tmp.replace(dst)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _moabb_data_path(dataset, subject: int, raw_dir: Path, *, force_update: bool) -> list[Path]:
    kwargs = {
        "subject": subject,
        "path": str(raw_dir),
        "force_update": force_update,
        "update_path": True,
    }
    try:
        paths = dataset.data_path(**kwargs)
    except TypeError:
        kwargs.pop("update_path", None)
        try:
            paths = dataset.data_path(**kwargs)
        except TypeError:
            paths = dataset.data_path(subject)
    return _flatten_paths(paths)


def _flatten_paths(paths) -> list[Path]:
    out: list[Path] = []
    if paths is None:
        return out
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    if isinstance(paths, dict):
        iterable = paths.values()
    else:
        iterable = paths
    for item in iterable:
        if isinstance(item, (str, os.PathLike)):
            out.append(Path(item))
        elif isinstance(item, dict):
            out.extend(_flatten_paths(item))
        elif isinstance(item, (list, tuple, set)):
            out.extend(_flatten_paths(item))
    return out


def _link_downloaded_files(paths: list[Path], raw_dir: Path) -> int:
    link_dir = raw_dir / "moabb_links"
    link_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in paths:
        if not src.exists() or not src.is_file():
            continue
        dst = link_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            continue
        count += 1
    return count


if __name__ == "__main__":
    main()
