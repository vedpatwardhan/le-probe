import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go

# --- Path Stabilization ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LEWM_DIR = ROOT_DIR / "lewm"
if str(LEWM_DIR) not in sys.path:
    sys.path.append(str(LEWM_DIR))

from visualize_manifold import interpolate_color


def visualize_splits(input_file, dataset_path, output_dir):
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load harvested manifold data
    print(f"📂 Loading manifold data from {input_file}...")
    data = torch.load(input_file, weights_only=False)
    latents = data["latents"]
    indices = data["frame_indices"]
    ep_indices = data.get("episode_indices", None)

    if ep_indices is None:
        ep_indices = np.array([i // 32 for i in range(len(indices))])

    # 2. Directly read the parquet metadata to classify episodes
    parquet_path = Path(dataset_path) / "meta/episodes/chunk-000/file-000.parquet"
    print(f"📖 Reading episode metadata from: {parquet_path}")
    df_episodes = pd.read_parquet(parquet_path)

    def get_class(tasks):
        if len(tasks) == 0:
            return "fail"
        task_lower = str(tasks[0]).lower()
        if "success" in task_lower:
            return "success"
        elif "suboptimal" in task_lower:
            return "suboptimal"
        else:
            return "fail"

    df_episodes["class"] = df_episodes["tasks"].apply(get_class)

    success_eps = df_episodes[df_episodes["class"] == "success"]["episode_index"].values
    suboptimal_eps = df_episodes[df_episodes["class"] == "suboptimal"][
        "episode_index"
    ].values
    fail_eps = df_episodes[df_episodes["class"] == "fail"]["episode_index"].values

    print(
        f"📊 Dataset Info: {len(success_eps)} Success, {len(suboptimal_eps)} Suboptimal, {len(fail_eps)} Fail episodes."
    )

    # Loop over all three dimensionality reduction methods
    methods = ["pca", "tsne", "umap"]

    for method in methods:
        print(
            f"\n📉 Computing {method.upper()} dimensionality reduction (shared coordinate space)..."
        )
        if method == "pca":
            reducer = PCA(n_components=3)
        elif method == "tsne":
            reducer = TSNE(n_components=3, perplexity=30, max_iter=1000)
        elif method == "umap":
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
        else:
            raise ValueError(f"Unsupported method: {method}")

        reduced_latents = reducer.fit_transform(latents)

        colors = np.array([interpolate_color(idx) for idx in indices])
        hover_text = np.array(
            [f"Ep: {ep_indices[i]} | Fr: {indices[i]}" for i in range(len(indices))]
        )

        # 4. Generate the 4 plots
        splits = {
            "all": np.ones(len(indices), dtype=bool),
            "success": np.isin(ep_indices, success_eps),
            "suboptimal": np.isin(ep_indices, suboptimal_eps),
            "fail": np.isin(ep_indices, fail_eps),
        }

        for name, mask in splits.items():
            if not mask.any():
                print(f"⚠️ No frames found for split '{name}', skipping.")
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=reduced_latents[mask, 0],
                    y=reduced_latents[mask, 1],
                    z=reduced_latents[mask, 2],
                    mode="markers",
                    name=f"Manifold - {name.capitalize()}",
                    marker=dict(size=3, color=colors[mask], opacity=0.7),
                    text=hover_text[mask],
                    hoverinfo="text",
                )
            )

            fig.update_layout(
                title=f"LeWM Latent Manifold ({method.upper()}) - {name.upper()} ({mask.sum()} frames)",
                scene=dict(
                    xaxis_title=f"{method.upper()} 1",
                    yaxis_title=f"{method.upper()} 2",
                    zaxis_title=f"{method.upper()} 3",
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )

            output_filename = out_path / f"manifold_3d_{method}_{name}.html"
            fig.write_html(str(output_filename))
            print(f"✨ Saved {method.upper()} {name} plot to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="manifold_data_2k_multiview_skeleton_dino.pt"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="le-probe/datasets/gr1_pickup_grasp_2k"
    )
    parser.add_argument(
        "--output_dir", type=str, default="le-probe/manifold_visualization/v2"
    )
    args = parser.parse_args()
    visualize_splits(args.input, args.dataset_path, args.output_dir)
