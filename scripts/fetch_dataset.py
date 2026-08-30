#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.registry import AssetRegistry

BETA_FIGSHARE_ARTICLE_ID = 12264401
BETA_FIGSHARE_ARTICLE_VERSION = 3
WEARABLE_FIGSHARE_ARTICLE_ID = 13560281
WEARABLE_FIGSHARE_ARTICLE_VERSION = 4
DONG2023_ZENODO_RECORD_ID = 18847318
WANG2016_ZENODO_RECORD_ID = 14865172


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["beta", "wang", "wearable", "dong2023", "openbci"],
    )
    parser.add_argument("--assets-config", default="configs/assets.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument(
        "--subjects", default=None, help="Comma-separated subject ids, e.g. 1,2,3. Default: all."
    )
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent file downloads (default: 1).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned action only.")
    parser.add_argument(
        "--probe-remote",
        action="store_true",
        help="Probe public source metadata without downloading data.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    registry = AssetRegistry.from_yaml(args.assets_config, strict_env=False)
    cfg = registry.dataset(args.dataset)
    if args.dataset == "openbci":
        print(
            "openbci is local private data. Use scripts/openbci_convert.py on a local session folder."
        )
        return
    if args.dataset == "dong2023":
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/dong2023")
        record_id = int(cfg.get("zenodo_record_id") or DONG2023_ZENODO_RECORD_ID)
        record = _zenodo_record(record_id)
        files = _select_dong2023_files(record, _parse_subjects(args.subjects))
        total_size = sum(int(file.get("size", 0)) for file in files)
        print(f"Dong2023 source: Zenodo record {record_id}")
        print(f"Selected files: {len(files)} ({_format_bytes(total_size)})")
        if args.probe_remote or args.dry_run:
            _print_zenodo_plan(files, raw_dir)
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "zenodo_record.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        downloaded = _download_many(
            files,
            raw_dir,
            name_key="key",
            downloader=_download_zenodo_file,
            force_update=args.force_update,
            workers=args.workers,
        )
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} Dong2023 file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset dong2023 "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/dong2023_v1 "
            "--config configs/data/dong2023.yaml"
        )
        return
    if args.dataset == "wearable":
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/wearable")
        article_id = int(cfg.get("figshare_article_id") or WEARABLE_FIGSHARE_ARTICLE_ID)
        article_version = int(
            cfg.get("figshare_article_version") or WEARABLE_FIGSHARE_ARTICLE_VERSION
        )
        article = _figshare_article(article_id, article_version)
        files = _select_wearable_files(article, _parse_subjects(args.subjects))
        total_size = sum(int(file.get("size", 0)) for file in files)
        print(f"Wearable source: Figshare article {article_id} version {article_version}")
        print(f"Selected files: {len(files)} ({_format_bytes(total_size)})")
        if args.probe_remote or args.dry_run:
            _print_figshare_plan(files, raw_dir)
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "figshare_article.json").open("w", encoding="utf-8") as f:
            json.dump(article, f, indent=2)
        downloaded = _download_many(
            files,
            raw_dir,
            name_key="name",
            downloader=_download_figshare_file,
            force_update=args.force_update,
            workers=args.workers,
        )
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} wearable file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset wearable "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/wearable_v3 "
            "--config configs/data/wearable.yaml"
        )
        return
    if args.dataset == "beta":
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/beta")
        article_id = int(cfg.get("figshare_article_id") or BETA_FIGSHARE_ARTICLE_ID)
        article_version = int(cfg.get("figshare_article_version") or BETA_FIGSHARE_ARTICLE_VERSION)
        article = _figshare_article(article_id, article_version)
        files = _select_beta_files(article, _parse_subjects(args.subjects))
        total_size = sum(int(file.get("size", 0)) for file in files)
        print(f"BETA source: Figshare article {article_id} version {article_version}")
        print(f"Selected files: {len(files)} ({_format_bytes(total_size)})")
        if args.probe_remote or args.dry_run:
            _print_figshare_plan(files, raw_dir)
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "figshare_article.json").open("w", encoding="utf-8") as f:
            json.dump(article, f, indent=2)
        downloaded = _download_many(
            files,
            raw_dir,
            name_key="name",
            downloader=_download_figshare_file,
            force_update=args.force_update,
            workers=args.workers,
        )
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} BETA file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset beta "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/beta_v1 --config configs/data/beta.yaml"
        )
        return
    if args.dataset == "wang":
        raw_dir = _path_arg(args.raw_dir or cfg.get("raw_dir"), "$EEG_DATA_ROOT/raw/wang")
        record_id = int(cfg.get("zenodo_record_id") or WANG2016_ZENODO_RECORD_ID)
        record = _zenodo_record(record_id)
        files = _select_wang_files(record, _parse_subjects(args.subjects))
        total_size = sum(int(file.get("size", 0)) for file in files)
        print(f"Wang2016 source: Zenodo record {record_id}")
        print(f"Selected files: {len(files)} ({_format_bytes(total_size)})")
        if args.probe_remote or args.dry_run:
            _print_zenodo_plan(files, raw_dir)
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "zenodo_record.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        downloaded = _download_many(
            files,
            raw_dir,
            name_key="key",
            downloader=_download_zenodo_file,
            force_update=args.force_update,
            workers=args.workers,
        )
        with (raw_dir / "downloaded_paths.txt").open("w", encoding="utf-8") as f:
            for path in downloaded:
                f.write(f"{path}\n")
        print(f"Downloaded/resolved {len(downloaded)} Wang2016 file(s) under {raw_dir}.")
        print("Next:")
        print(
            "  python scripts/prepare_dataset.py --dataset wang "
            f"--raw_dir {raw_dir} --out_dir $EEG_DATA_ROOT/processed/wang_v1 --config configs/data/wang.yaml"
        )


def _parse_subjects(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _validate_subject_inventory(
    *, dataset: str, found: list[int], selected: set[int], expected_total: int
) -> None:
    expected = selected or set(range(1, expected_total + 1))
    counts = Counter(found)
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(subject for subject, count in counts.items() if count != 1)
    if missing or extra or duplicates:
        raise SystemExit(
            f"{dataset} subject inventory mismatch: missing={missing}, extra={extra}, "
            f"duplicates={duplicates}; "
            f"expected {len(expected)} subject file(s)."
        )


def _validate_support_inventory(*, dataset: str, found: list[str], expected: set[str]) -> None:
    normalized = [name.lower() for name in found]
    counts = Counter(normalized)
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(name for name, count in counts.items() if count != 1)
    if missing or extra or duplicates:
        raise SystemExit(
            f"{dataset} support-file inventory mismatch: missing={missing}, extra={extra}, "
            f"duplicates={duplicates}."
        )


def _path_arg(raw: str | None, default: str) -> Path:
    value = str(raw or default)
    value = re.sub(
        r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )
    return Path(os.path.expandvars(value)).expanduser()


def _figshare_article(article_id: int, version: int | None = None) -> dict[str, Any]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    if version is not None:
        url = f"{url}/versions/{version}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _zenodo_record(record_id: int) -> dict[str, Any]:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_beta_files(article: dict[str, Any], subjects: list[int] | None) -> list[dict[str, Any]]:
    subject_files: list[dict[str, Any]] = []
    support_files: list[dict[str, Any]] = []
    selected = set(subjects or [])
    support_names = {"description.pdf", "note.txt"}
    for file in article.get("files", []):
        name = str(file.get("name", ""))
        match = re.fullmatch(r"S(\d+)\.mat", name, flags=re.IGNORECASE)
        if match:
            subject = int(match.group(1))
            if not selected or subject in selected:
                subject_files.append({**file, "subject": subject})
        elif name.lower() in support_names:
            support_files.append(file)
    subject_files.sort(key=lambda item: int(item["subject"]))
    support_files.sort(key=lambda item: str(item["name"]).lower())
    if not subject_files:
        raise SystemExit("No matching BETA S*.mat files found in the Figshare article.")
    _validate_subject_inventory(
        dataset="BETA",
        found=[int(file["subject"]) for file in subject_files],
        selected=selected,
        expected_total=70,
    )
    _validate_support_inventory(
        dataset="BETA",
        found=[str(file["name"]) for file in support_files],
        expected=support_names,
    )
    return [*subject_files, *support_files]


def _select_wearable_files(
    article: dict[str, Any], subjects: list[int] | None
) -> list[dict[str, Any]]:
    selected_subjects = set(subjects or [])
    subject_files: list[dict[str, Any]] = []
    support_files: list[dict[str, Any]] = []
    support_names = {
        "impedance.mat",
        "readme.pdf",
        "subjects_information.mat",
        "stimulation_information.pdf",
    }
    for file in article.get("files", []):
        name = str(file.get("name", ""))
        match = re.fullmatch(r"S(\d{3})\.mat", name, flags=re.IGNORECASE)
        if match:
            subject = int(match.group(1))
            if not selected_subjects or subject in selected_subjects:
                subject_files.append({**file, "subject": subject})
        elif name.lower() in support_names:
            support_files.append(file)
    subject_files.sort(key=lambda item: int(item["subject"]))
    support_files.sort(key=lambda item: str(item["name"]).lower())
    if not subject_files:
        raise SystemExit("No matching wearable S###.mat files found in the Figshare article.")
    _validate_subject_inventory(
        dataset="Wearable",
        found=[int(file["subject"]) for file in subject_files],
        selected=selected_subjects,
        expected_total=102,
    )
    _validate_support_inventory(
        dataset="Wearable",
        found=[str(file["name"]) for file in support_files],
        expected=support_names,
    )
    return [*subject_files, *support_files]


def _select_dong2023_files(
    record: dict[str, Any], subjects: list[int] | None
) -> list[dict[str, Any]]:
    selected_subjects = set(subjects or [])
    subject_files: list[dict[str, Any]] = []
    support_files: list[dict[str, Any]] = []
    support_names = {"8-channels.mat", "59-subject.mat"}
    for file in record.get("files", []):
        name = str(file.get("key", ""))
        match = re.fullmatch(r"S(\d+)\.mat", name, flags=re.IGNORECASE)
        if match:
            subject = int(match.group(1))
            if not selected_subjects or subject in selected_subjects:
                subject_files.append({**file, "subject": subject})
        elif name.lower() in support_names:
            support_files.append(file)
    subject_files.sort(key=lambda item: int(item["subject"]))
    support_files.sort(key=lambda item: str(item["key"]).lower())
    if not subject_files:
        raise SystemExit("No matching Dong2023 S*.mat files found in the Zenodo record.")
    _validate_subject_inventory(
        dataset="Dong2023",
        found=[int(file["subject"]) for file in subject_files],
        selected=selected_subjects,
        expected_total=59,
    )
    _validate_support_inventory(
        dataset="Dong2023",
        found=[str(file["key"]) for file in support_files],
        expected=support_names,
    )
    return [*subject_files, *support_files]


def _select_wang_files(record: dict[str, Any], subjects: list[int] | None) -> list[dict[str, Any]]:
    selected_subjects = set(subjects or [])
    subject_files: list[dict[str, Any]] = []
    support_files: list[dict[str, Any]] = []
    support_names = {"64-channels.loc", "readme.txt", "sub_info.txt"}
    for file in record.get("files", []):
        name = str(file.get("key", ""))
        match = re.fullmatch(r"S(\d+)\.mat", name, flags=re.IGNORECASE)
        if match:
            subject = int(match.group(1))
            if not selected_subjects or subject in selected_subjects:
                subject_files.append({**file, "subject": subject})
        elif name.lower() in support_names:
            support_files.append(file)
    subject_files.sort(key=lambda item: int(item["subject"]))
    support_files.sort(key=lambda item: str(item["key"]).lower())
    if not subject_files:
        raise SystemExit("No matching Wang2016 S*.mat files found in the Zenodo record.")
    _validate_subject_inventory(
        dataset="Wang2016",
        found=[int(file["subject"]) for file in subject_files],
        selected=selected_subjects,
        expected_total=35,
    )
    _validate_support_inventory(
        dataset="Wang2016",
        found=[str(file["key"]) for file in support_files],
        expected=support_names,
    )
    return [*subject_files, *support_files]


def _print_figshare_plan(files: list[dict[str, Any]], raw_dir: Path) -> None:
    for file in files[:10]:
        print(f"  {file['name']}: {_format_bytes(int(file.get('size', 0)))}")
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")
    print(f"Target raw_dir: {raw_dir}")
    print("Run without --dry-run/--probe-remote to download.")


def _print_zenodo_plan(files: list[dict[str, Any]], raw_dir: Path) -> None:
    for file in files[:10]:
        print(f"  {file['key']}: {_format_bytes(int(file.get('size', 0)))}")
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")
    print(f"Target raw_dir: {raw_dir}")
    print("Run without --dry-run/--probe-remote to download.")


def _download_figshare_file(file: dict[str, Any], dst: Path, *, force_update: bool) -> None:
    expected_size = int(file.get("size", 0) or 0)
    expected_md5 = str(file.get("computed_md5") or file.get("supplied_md5") or "")
    if dst.exists() and not force_update:
        size_matches = not expected_size or dst.stat().st_size == expected_size
        checksum_matches = not expected_md5 or _md5(dst) == expected_md5
        if size_matches and checksum_matches:
            print(f"cache ok: {dst}")
            return
        raise SystemExit(
            f"Existing file looks incomplete or mismatched: {dst}\n"
            "Rerun with --force-update to replace it."
        )
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


def _download_zenodo_file(file: dict[str, Any], dst: Path, *, force_update: bool) -> None:
    checksum = str(file.get("checksum") or "")
    expected_md5 = checksum.removeprefix("md5:") if checksum.startswith("md5:") else ""
    normalized = {
        "name": str(file["key"]),
        "size": int(file.get("size", 0) or 0),
        "computed_md5": expected_md5,
        "download_url": str(file["links"]["self"]),
    }
    _download_figshare_file(normalized, dst, force_update=force_update)


def _download_many(
    files: list[dict[str, Any]],
    raw_dir: Path,
    *,
    name_key: str,
    downloader: Callable[..., None],
    force_update: bool,
    workers: int,
) -> list[Path]:
    paths = [raw_dir / str(file[name_key]) for file in files]
    if workers == 1:
        for file, dst in zip(files, paths, strict=True):
            downloader(file, dst, force_update=force_update)
        return paths

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(downloader, file, dst, force_update=force_update): dst
            for file, dst in zip(files, paths, strict=True)
        }
        for future in as_completed(futures):
            future.result()
    return paths


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


if __name__ == "__main__":
    main()
