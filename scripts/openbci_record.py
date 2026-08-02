#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an OpenBCI session through BrainFlow.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--serial-port", required=True)
    parser.add_argument("--board-id", type=int, default=0, help="BrainFlow board id; 0 is Cyton.")
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--events-file", required=True, help="CSV with trial_id,onset_sec,duration_sec,stimulus_frequency_hz.")
    parser.add_argument("--subject-id", required=True, help="Pseudonym matching subNNN.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--channel-names",
        default="Pz,PO3,PO4,POz,PO7,O1,Oz,O2",
        help="Comma-separated names in board EEG-channel order.",
    )
    parser.add_argument("--reference", default="openbci_default")
    parser.add_argument("--electrode-type", default="dry", choices=["dry", "wet", "gel", "unknown"])
    parser.add_argument("--cap-type", default="wearable")
    parser.add_argument("--environment-note-code", default="unknown")
    parser.add_argument("--reattach-flag", action="store_true")
    parser.add_argument("--time-since-last-session-hours", type=float, default=None)
    args = parser.parse_args()

    if not re.fullmatch(r"sub\d{3,}", args.subject_id):
        raise SystemExit("--subject-id must be a pseudonym such as sub001; do not use a name.")
    events = pd.read_csv(args.events_file)
    required = {"trial_id", "onset_sec", "duration_sec", "stimulus_frequency_hz"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise SystemExit(f"Events CSV is missing columns: {missing}")
    if events.empty:
        raise SystemExit("Events CSV is empty.")
    if float((events["onset_sec"] + events["duration_sec"]).max()) > args.duration_sec:
        raise SystemExit("Event schedule extends beyond --duration-sec.")

    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams
    except Exception as exc:
        raise SystemExit(
            "BrainFlow is not installed. Run CFEG_ENABLE_OPENBCI=1 bash scripts/cfeg.sh setup, "
            "or export CSV from OpenBCI GUI and use scripts/openbci_convert.py."
        ) from exc

    channel_names = [value.strip() for value in args.channel_names.split(",") if value.strip()]
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    board = BoardShim(args.board_id, params)
    board.prepare_session()
    try:
        board.start_stream()
        started = time.monotonic()
        while time.monotonic() - started < args.duration_sec:
            time.sleep(min(0.25, args.duration_sec - (time.monotonic() - started)))
        board.stop_stream()
        data = board.get_board_data()
    finally:
        if board.is_prepared():
            board.release_session()

    eeg_rows = BoardShim.get_eeg_channels(args.board_id)
    if len(channel_names) != len(eeg_rows):
        raise SystemExit(
            f"Provided {len(channel_names)} channel names but board exposes {len(eeg_rows)} EEG channels."
        )
    timestamp_row = BoardShim.get_timestamp_channel(args.board_id)
    timestamps = data[timestamp_row]
    relative_time = timestamps - timestamps[0] if len(timestamps) else timestamps
    frame = pd.DataFrame({"time": relative_time})
    for name, row in zip(channel_names, eeg_rows):
        frame[name] = data[row]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "eeg.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    metadata = {
        "subject_id": args.subject_id,
        "session_id": args.session_id,
        "sfreq": float(BoardShim.get_sampling_rate(args.board_id)),
        "channel_names": channel_names,
        "reference": args.reference,
        "hardware_id": "openbci_cyton" if args.board_id == 0 else f"brainflow_board_{args.board_id}",
        "electrode_type": args.electrode_type,
        "cap_type": args.cap_type,
        "reattach_flag": args.reattach_flag,
        "time_since_last_session_hours": args.time_since_last_session_hours,
        "environment_note_code": args.environment_note_code,
    }
    (out_dir / "session_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Recorded {len(frame)} samples to {out_dir}")
    print(
        "Next: python scripts/openbci_convert.py "
        f"--raw_session_dir {out_dir} --out_dir $EEG_DATA_ROOT/processed/openbci_v1"
    )


if __name__ == "__main__":
    main()
