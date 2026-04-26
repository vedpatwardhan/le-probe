import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_jsons_to_parquet(json_dir, output_file):
    print(f"📦 Converting JSONs from {json_dir} to Parquet...")
    json_files = list(Path(json_dir).glob("*.json"))
    json_files.sort()

    data = []
    for f in tqdm(json_files, desc="Reading JSONs"):
        with open(f, "r") as f_in:
            try:
                item = json.load(f_in)
                # Ensure it has the necessary keys
                if "observation.images.world_center" in item:
                    data.append(item)
            except Exception as e:
                print(f"⚠️ Skipping {f.name}: {e}")

    print(f"📊 Creating DataFrame for {len(data)} items...")
    df = pd.DataFrame(data)

    print(f"💾 Saving to {output_file}...")
    df.to_parquet(output_file, index=False, compression="snappy")

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Success! Parquet size: {size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred/dataset.parquet",
    )
    args = parser.parse_args()

    convert_jsons_to_parquet(args.dir, args.out)
