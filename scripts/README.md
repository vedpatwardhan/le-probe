# 🛠 Utility Scripts

> One-time utilities, dataset migrations, and performance audits.

This folder contains maintenance scripts used to process raw data, validate model performance, and handle repository-level tasks.

## 📋 Highlights

- **`find_feature_triggers.py`**: Identifies peak activation frames for specific neural features across concatenated datasets.
- **`generate_canonical_triptychs.py`**: Generates bit-perfect "Before/After/Activation" triptychs using original dataset decoders.
- **`reproduce_canonical_states_direct.py`**: Extracts precise 32-dim action/state vectors and high-res images for reproduction trials.
- **`check_target_activations.py`**: Verifies raw magnitudes within the harvested activation dataset for ground-truth validation.
- **`convert_to_parquet.py`**: Optimizes datasets for training by converting them to the Parquet format.
- **`analyze_snapshots.py`**: Audits the distribution of stored goal snapshots.
- **`compare_reward_models.py`**: Benchmarks different iterations of the RA-LeWM reward head.
- **`compress_dataset.py`**: Handles Zarr-to-Video compression for dataset portability.
