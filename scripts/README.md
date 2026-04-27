# 🛠 Utility Scripts

> One-time utilities, dataset migrations, and performance audits.

This folder contains maintenance scripts used to process raw data, validate model performance, and handle repository-level tasks.

## 📋 Highlights

- **`convert_to_parquet.py`**: Optimizes datasets for training by converting them to the Parquet format.
- **`analyze_snapshots.py`**: Audits the distribution of stored goal snapshots.
- **`compare_reward_models.py`**: Benchmarks different iterations of the RA-LeWM reward head.
- **`compress_dataset.py`**: Handles Zarr-to-Video compression for dataset portability.

---
*Part of the [Le-Probe](..) project.*
