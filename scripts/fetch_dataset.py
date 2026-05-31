#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.assets.registry import AssetRegistry


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
    if args.probe_remote and args.dataset == "beta":
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise SystemExit(
                "huggingface_hub is not installed. Install requirements first:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        repo_id = cfg.get("repo_id", "Bingchuan/beta")
        info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(repo_id)
        print(f"remote ok: dataset {repo_id} sha={info.sha}")
        return
    if args.dataset == "beta":
        if args.dry_run:
            print(
                "Would run datasets.load_dataset('Bingchuan/beta') and save_to_disk under "
                "$EEG_DATA_ROOT/raw/beta/hf_dataset."
            )
            return
        from datasets import load_dataset

        raw_dir = Path(args.raw_dir or cfg.get("raw_dir") or "$EEG_DATA_ROOT/raw/beta").expanduser()
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = raw_dir / "hf_cache"
        save_dir = raw_dir / "hf_dataset"
        repo_id = cfg.get("repo_id", "Bingchuan/beta")
        print(f"Fetching BETA dataset={repo_id}")
        print(f"HF datasets cache_dir={cache_dir}")
        print(f"Saving dataset object to={save_dir}")
        ds = load_dataset(repo_id, cache_dir=str(cache_dir))
        ds.save_to_disk(str(save_dir))
        with (raw_dir / "hf_dataset_info.txt").open("w", encoding="utf-8") as f:
            f.write(str(ds))
            f.write("\n\n")
            for split, split_ds in ds.items():
                f.write(f"[{split}]\n")
                f.write(f"features={split_ds.features}\n")
                f.write(f"num_rows={split_ds.num_rows}\n\n")
        print(ds)
        print("Next inspect:")
        print(f"  cat {raw_dir / 'hf_dataset_info.txt'}")
        print("Then prepare if the downloaded files expose numeric EEG arrays:")
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
        raw_dir = Path(args.raw_dir or cfg.get("raw_dir") or "$EEG_DATA_ROOT/raw/wang").expanduser()
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
