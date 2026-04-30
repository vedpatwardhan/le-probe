# --- Path Stabilization ---
# This ensures that project-specific modules like 'lewm' and 'le_wm' are discoverable
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for p in [ROOT_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import zmq
import msgpack
import rerun as rr
import mujoco
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from simulation_base import GR1MuJoCoBase
from gr1_config import COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from lewm.goal_mapper import GoalMapper
from interpretability.clt.clt_model import CrossLayerTranscoder


class InterpretiveTeleopServer(GR1MuJoCoBase):
    """
    Interpretive Teleoperation Server.
    Focused strictly on visual simulation and latent auditing snapshots.

    This server allows you to 'X-ray' the model's decision process while manually
    controlling the robot via the teleop dashboard. It removes all dataset-collection
    logic (recording, syncing, etc.) found in the standard teleop server.
    """

    def __init__(self, model_path, clt_path, port=5556, lock_posture=False):
        super().__init__()
        self.port = port
        self.lock_posture = lock_posture
        self.is_running = True

        # Auto-detect hardware
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"🧠 Loading Interpretability Stack on {self.device}...")

        # 1. Load the World Model (LeWM) via GoalMapper
        self.agent = GoalMapper(model_path, dataset_root=".")
        self.nn_model = self.agent.model.to(self.device)
        self.nn_model.eval()

        # 2. Load the Cross-Layer Transcoder (CLT)
        # The CLT maps visual encoder features to the predictor's latent space
        checkpoint = torch.load(clt_path, map_location=self.device)
        self.norm = checkpoint["norm_stats"]
        self.clt = CrossLayerTranscoder(
            d_model=checkpoint["config"]["d_model"], d_sae=checkpoint["config"]["d_sae"]
        ).to(self.device)
        self.clt.load_state_dict(checkpoint["state_dict"])
        self.clt.eval()

        # Top-3 Mechanistic Features for static PNG snapshots
        self.audit_features = {
            90: "Tactile Engagement",
            358: "Spatial Lockdown",
            743: "Alignment Precision",
        }

        # Directory for PNG snapshots (Strictly offline auditing)
        self.snapshot_dir = os.path.join(ROOT_DIR, "brain_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # State buffers
        self.before_img = None
        self.current_action = np.zeros(32, dtype=np.float32)

    def log_to_rerun(self):
        """Log all 5 camera views to the Rerun stream."""
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            rgb = self.renderer.render()
            rr.log(cam, rr.Image(rgb))

    def capture_before_state(self):
        """Saves the current visual state before an action is applied."""
        self.renderer.update_scene(self.data, camera="world_center")
        rgb = self.renderer.render()
        self.before_img = Image.fromarray(rgb).resize((224, 224))

    @torch.no_grad()
    def save_brain_snapshot(self):
        """Captures the model's latent activations and saves a PNG triptych."""
        self.renderer.update_scene(self.data, camera="world_center")
        rgb = self.renderer.render()
        after_img = Image.fromarray(rgb).resize((224, 224))

        # Predictor Pass (Mind's Eye)
        batch = self.agent.transform({"pixels": np.array(after_img)})
        x = batch["pixels"].unsqueeze(0).to(self.device)

        enc_out = self.nn_model.encoder(x, interpolate_pos_encoding=True)
        pixels_emb = enc_out.last_hidden_state[:, 0]
        z_enc = self.nn_model.projector(pixels_emb).unsqueeze(0)

        action_t = (
            torch.tensor(self.current_action, device=self.device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        act_emb = self.nn_model.action_encoder(action_t)
        z_pred = self.nn_model.predict(z_enc, act_emb)
        z_pred_flat = z_pred.squeeze(0).squeeze(0).unsqueeze(0)

        # Transcode to Sparse Predictor Features
        mean = self.norm["mean_L"].to(self.device)
        std = self.norm["std_L"].to(self.device)
        z_norm = (z_pred_flat - mean) / std

        x_centered = z_norm - self.clt.b_dec
        acts = torch.nn.functional.relu(
            self.clt.encoder(x_centered) + self.clt.b_enc
        ).squeeze()

        self._render_snapshot_png(after_img, acts, self.before_img)

    def _render_snapshot_png(self, after_img, acts, before_img=None):
        """Generates a high-quality Before/After/Activation PNG on disk."""
        plt.style.use("dark_background")
        num_panels = 3 if before_img is not None else 2
        fig, axes = plt.subplots(
            1, num_panels, figsize=(6 * num_panels, 5), facecolor="#111111"
        )

        if num_panels == 3:
            axes[0].imshow(before_img)
            axes[0].set_title(
                "MIND'S EYE (BEFORE ACTION)",
                fontsize=10,
                fontweight="bold",
                color="#AAAAAA",
            )
            axes[0].axis("off")
            axes[1].imshow(after_img)
            axes[1].set_title(
                f"MIND'S EYE (AFTER ACTION - T={self.data.time:.2f}s)",
                fontsize=10,
                fontweight="bold",
                color="#FFFFFF",
            )
            axes[1].axis("off")
            ax_chart = axes[2]
        else:
            axes[0].imshow(after_img)
            axes[0].set_title(
                f"MIND'S EYE (T={self.data.time:.2f}s)",
                fontsize=10,
                fontweight="bold",
                color="#FFFFFF",
            )
            axes[0].axis("off")
            ax_chart = axes[1]

        names = [self.audit_features[fid] for fid in sorted(self.audit_features.keys())]
        vals = [float(acts[fid]) for fid in sorted(self.audit_features.keys())]
        ax_chart.barh(
            names,
            vals,
            color=["#FF4B4B", "#4BFF4B", "#4B4BFF"],
            alpha=0.8,
            edgecolor="white",
        )
        ax_chart.set_xlim(0, 5.0)
        ax_chart.set_title(
            "MECHANISTIC ACTIVATIONS", fontsize=10, fontweight="bold", color="#AAAAAA"
        )
        ax_chart.axis("off")

        path = os.path.join(
            self.snapshot_dir, f"brain_{int(self.data.time*100):05d}.png"
        )
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#111111")
        plt.close()

    def run(self):
        """Server loop handling ZMQ commands from the dashboard."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(f"🚀 Lean Interpretive Teleop Running on port {self.port}")

        while self.is_running:
            data = msgpack.unpackb(socket.recv(), raw=False)
            cmd = data.get("command")

            def send_resp(payload):
                self.renderer.update_scene(self.data, camera="world_center")
                rgb = self.renderer.render()
                img = Image.fromarray(rgb).resize((224, 224))
                img_np = np.array(img).astype(np.uint8)
                payload.update(
                    {
                        "physics": self.get_physics_state(),
                        "image": {
                            "data": img_np.tobytes(),
                            "dtype": str(img_np.dtype),
                            "shape": img_np.shape,
                        },
                    }
                )
                socket.send(msgpack.packb(payload))

            # Handlers for simplified command set
            if cmd == "reset":
                self.before_img = None
                self.reset_env(lock_posture=self.lock_posture)
                send_resp(
                    {
                        "status": "reset_ok",
                        "joints": StandardScaler()
                        .scale_state(self.get_state_32())
                        .tolist(),
                    }
                )

            elif cmd == "wild_randomize":
                self.before_img = None
                self.wild_reset()
                send_resp(
                    {
                        "status": "wild_randomize_ok",
                        "joints": StandardScaler()
                        .scale_state(self.get_state_32())
                        .tolist(),
                    }
                )

            elif cmd == "poll_status":
                send_resp({"status": "status_ok"})

            elif "target" in data:
                # Capture visual state BEFORE the motor dispatch
                self.capture_before_state()
                action_32 = np.array(data["target"], dtype=np.float32)
                self.current_action = action_32

                # Use high-fidelity base class methods for movement
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
                send_resp({"status": "step_ok"})

            elif cmd == "store_snapshot":
                # Manual snapshot trigger (Captures brain triptych)
                self.save_brain_snapshot()
                send_resp({"status": "snapshot_ok"})
            else:
                send_resp({"status": "unknown"})

            # Sync Rerun timeline
            rr.set_time_sequence("step", int(self.data.time * 100))
            self.log_to_rerun()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--clt", type=str, default="clt_weights.pt")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    InterpretiveTeleopServer(
        model_path=args.model, clt_path=args.clt, port=args.port, lock_posture=True
    ).run()
