import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to import gr1_config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gr1_config import COMPACT_WIRE_JOINTS, JOINT_STATS


def visualize_log(log_path="run_logs/15/inference_history.json"):
    if not os.path.exists(log_path):
        print(f"Error: Log file '{log_path}' not found.")
        return

    with open(log_path, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{log_path}'.")
            return

    num_entries = len(history)
    print(f"Loaded {num_entries} entries from {log_path}.")

    while True:
        try:
            user_input = (
                input(f"\nEnter index (0-{num_entries-1}) or 'q' to quit: ")
                .strip()
                .lower()
            )
            if user_input == "q":
                break

            idx = int(user_input)
            if not (0 <= idx < num_entries):
                print(
                    f"Invalid index. Please enter a value between 0 and {num_entries-1}."
                )
                continue

            entry = history[idx]
            batch = entry.get("batch", {})
            actions = np.array(entry.get("output", []))  # (16, 32)
            timestamp = entry.get("timestamp", 0)

            # Reconstruct images
            # observation.images.head, observation.images.world_left, etc.
            # They were stored as CHW (3, 224, 224)
            img_keys = [
                "observation.images.head",
                "observation.images.world_left",
                "observation.images.world_right",
                "observation.images.world_center",
            ]

            fig = plt.figure(figsize=(18, 10))
            # gs = fig.add_gridspec(2, 4)
            gs = fig.add_gridspec(1, 1)

            # 1. Plot Cameras
            # cam_titles = ["Head", "World Left", "World Right", "World Center"]
            # for i, key in enumerate(img_keys):
            #     ax = fig.add_subplot(gs[0, i])
            #     if key in batch:
            #         img_data = np.array(batch[key])
            #         # Convert CHW to HWC
            #         if img_data.ndim == 3:
            #             img_data = img_data.transpose(1, 2, 0)

            #         ax.imshow(img_data)
            #         ax.set_title(cam_titles[i])
            #     else:
            #         ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center")
            #     ax.axis("off")

            final_actions = []
            final_joints = []
            for idx, name in enumerate(COMPACT_WIRE_JOINTS):
                min_v, max_v, _ = JOINT_STATS[idx]
                denom = max_v - min_v
                if "left" not in name and denom > 0:
                    final_actions.append(
                        min_v + ((actions[:, idx] + 1.0) / 2.0) * denom
                    )
                    final_joints.append(name)
            final_actions = np.array(final_actions).T
            print(final_actions[0])

            # 2. Plot Action Heatmap
            # ax_heat = fig.add_subplot(gs[1, :])
            ax_heat = fig.add_subplot(gs[0, :])
            if final_actions.size > 0:
                im = ax_heat.imshow(
                    final_actions.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1
                )
                ax_heat.set_title("Action Heatmap (16x32 Action Chunk)")
                ax_heat.set_ylabel("Joints")
                ax_heat.set_xlabel("Horizon Step")

                # Set Y-ticks to action labels
                ax_heat.set_yticks(range(len(final_joints)))
                ax_heat.set_yticklabels(final_joints, fontsize=8)

                # Add colorbar
                # plt.colorbar(im, ax=ax_heat)
                fig.colorbar(
                    im, ax=ax_heat, orientation="vertical", fraction=0.02, pad=0.04
                )
            else:
                ax_heat.text(0.5, 0.5, "No action data", ha="center", va="center")

            instruction = batch.get("task", "N/A")
            plt.suptitle(
                f"Inference Entry {idx} | Time: {timestamp} | Task: {instruction}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            print(f"Displaying plot for entry {idx}. Close the window to continue...")
            plt.show()

        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    visualize_log()
