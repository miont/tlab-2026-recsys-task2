"""Data pipeline (stage 1 of the research plan).

Responsibilities:
- read ``flat/<size>/multi_event.parquet`` with polars;
- Global Temporal Split: train | val day | gap | test day (Yambda protocol);
- item vocabulary built on the train period only (top-N + ``<OOV>``);
- feedback features: played-ratio bins, track-length bins, organic flag, like flag;
- session boundaries (gap > ``session_gap_minutes``);
- sliding training windows cached as numpy arrays under ``Config.processed_dir``.
"""
