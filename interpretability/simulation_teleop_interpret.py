# --- Path Stabilization ---
# This ensures that project-specific modules like 'jepa' and 'gr1_modules' are discoverable
# regardless of whether the script is run from the root or the dataset folder.
import os
import sys

# Resolves project root and internal world model directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

# Add all relevant project directories to sys.path in high-priority order
for p in [ROOT_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
# --------------------------

import numpy as np
import zmq
import msgpack
import rerun as rr
import mujoco
import json
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from dataset.simulation_teleop import GR1TeleopServer
from lewm.goal_mapper import GoalMapper
from interpretability.clt.clt_model import CrossLayerTranscoder


class InterpretiveTeleopServer(GR1TeleopServer):
    """
    Interpretive Teleoperation Server.
    Inherits from the base teleop server to maintain 100% logic parity.
    Adds: 3-panel PNG brain snapshots (Before/After/Activation).
    Removes: Feature plots from the Rerun stream (Strictly camera-only).

    This server allows you to 'X-ray' the model's decision process while manually
    controlling the robot via the teleop dashboard.
    """

    def __init__(self, model_path, clt_path, port=5556, lock_posture=False):
        # Initialize the base teleop server (MuJoCo, ZMQ, Recorder)
        super().__init__(port=port, lock_posture=lock_posture)

        # Auto-detect the best available hardware (CUDA, MPS, or CPU)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"🧠 Loading Interpretability Stack on {self.device}...")

        # 1. Load the World Model (LeWM) via GoalMapper
        # NOTE: Using self.nn_model to avoid naming collision with MuJoCo's self.model
        self.agent = GoalMapper(model_path, dataset_root=".")
        self.nn_model = self.agent.model.to(self.device)
        self.nn_model.eval()

        # 2. Load the Cross-Layer Transcoder (CLT)
        # The CLT maps visual encoder features to the predictor's latent space
        if not os.path.exists(clt_path):
            raise FileNotFoundError(f"❌ CLT weights not found at: {clt_path}")

        checkpoint = torch.load(clt_path, map_location=self.device)
        self.norm = checkpoint["norm_stats"]  # Crucial: Must use training normalization
        self.clt = CrossLayerTranscoder(
            d_model=checkpoint["config"]["d_model"], d_sae=checkpoint["config"]["d_sae"]
        ).to(self.device)
        self.clt.load_state_dict(checkpoint["state_dict"])
        self.clt.eval()

        # Identified 'Smoking Gun' features from the Phase III Mechanistic Audit
        # 90: Fires when the model 'wants' to reach for the cube
        # 358: Fires when the model has locked onto the target coordinates
        # 743: Fires when the wrist is perfectly aligned for a grasp
        self.audit_features = {
            90: "Tactile Engagement",
            358: "Spatial Lockdown",
            743: "Alignment Precision",
        }

        # Directory for PNG snapshots (Strictly offline auditing)
        self.snapshot_dir = os.path.join(ROOT_DIR, "brain_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        print(f"📸 Brain snapshots enabled: {self.snapshot_dir}")

        # Buffer for 'Before' image
        self.before_img = None

    def log_to_rerun(self):
        """
        Overrides the base class to render and log ALL 5 cameras to Rerun.
        Uses the canonical camera names from GR1MuJoCoBase.
        """
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
        """
        Captures the model's latent activations and saves a PNG triptych.
        Ensures interpretability data is recorded on disk for every physical change.
        NOTE: Feature data is NOT logged to Rerun; Rerun is strictly for cameras.
        """
        # 1. Capture current visual (World Center Camera) - The 'After' state
        self.renderer.update_scene(self.data, camera="world_center")
        rgb = self.renderer.render()
        after_img = Image.fromarray(rgb).resize((224, 224))

        # 2. Extract Encoder Latents (The pure visual representation)
        batch = self.agent.transform({"pixels": np.array(after_img)})
        x = batch["pixels"].unsqueeze(0).to(self.device)

        # Feed through the neural model (nn_model)
        enc_out = self.nn_model.encoder(x, interpolate_pos_encoding=True)
        pixels_emb = enc_out.last_hidden_state[:, 0]
        enc_latent = self.nn_model.projector(pixels_emb)  # (1, 192)

        # 3. Transcode to Sparse Predictor Features (CLT Path)
        # We normalize the latent before passing it through the transcoder bottleneck
        x_norm = (enc_latent - self.norm["mean_L"].to(self.device)) / self.norm[
            "std_L"
        ].to(self.device)
        x_centered = x_norm - self.clt.b_dec
        acts = torch.nn.functional.relu(self.clt.encoder(x_centered) + self.clt.b_enc)
        acts = acts.squeeze()  # Resulting vector: (1024 sparse features)

        # 4. Generate PNG snapshot for this state
        # If we have a 'before' image, render the 3-panel view.
        self._render_snapshot_png(after_img, acts, self.before_img)

    def _render_snapshot_png(self, after_img, acts, before_img=None):
        """Helper to render a premium static visualization (Before/After/Activation)."""
        plt.style.use("dark_background")

        # Create a 3-panel layout if before_img is available, otherwise 2-panel
        num_panels = 3 if before_img is not None else 2
        fig, axes = plt.subplots(
            1, num_panels, figsize=(6 * num_panels, 5), facecolor="#111111"
        )

        if num_panels == 3:
            # Panel 1: Mind's Eye (Before)
            axes[0].imshow(before_img)
            axes[0].set_title(
                "MIND'S EYE (BEFORE ACTION)",
                fontsize=10,
                fontweight="bold",
                color="#AAAAAA",
            )
            axes[0].axis("off")

            # Panel 2: Mind's Eye (After)
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
            # Panel 1: Mind's Eye (Current/After)
            axes[0].imshow(after_img)
            axes[0].set_title(
                f"MIND'S EYE (T={self.data.time:.2f}s)",
                fontsize=10,
                fontweight="bold",
                color="#FFFFFF",
            )
            axes[0].axis("off")

            ax_chart = axes[1]

        # Activation Bar Chart
        names = [self.audit_features[fid] for fid in sorted(self.audit_features.keys())]
        vals = [float(acts[fid]) for fid in sorted(self.audit_features.keys())]
        colors = ["#FF4B4B", "#4BFF4B", "#4B4BFF"]
        bars = ax_chart.barh(
            names, vals, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5
        )

        ax_chart.set_xlim(0, 5.0)
        ax_chart.set_title(
            "MECHANISTIC ACTIVATIONS", fontsize=10, fontweight="bold", color="#AAAAAA"
        )
        ax_chart.grid(axis="x", linestyle="--", alpha=0.3)
        ax_chart.spines["top"].set_visible(False)
        ax_chart.spines["right"].set_visible(False)

        for bar in bars:
            width = bar.get_width()
            ax_chart.text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        path = os.path.join(
            self.snapshot_dir, f"brain_{int(self.data.time*100):05d}.png"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#111111")
        plt.close()
        print(f"📸 Delta Snapshot saved: {os.path.basename(path)}")

    def run(self):
        """
        Server loop (ZMQ REP).
        Matches the EXACT order and logic of simulation_teleop.py.
        """
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        # Keep original Rerun app name and connection logic
        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(f"🚀 Interpretive Teleop Running on port {self.port}")

        while self.is_running:
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")

            # Restore exact send_resp structure from base script
            def send_resp(payload):
                payload.update(
                    {
                        "upload_queue": self.recorder.pending_uploads,
                        "total_episodes": self.recorder.total_episodes,
                        "batch_status": self.recorder.episodes_since_sync,
                        "physics": self.get_physics_state(),
                    }
                )
                socket.send(msgpack.packb(payload))

            # --- EXACT ORDER RESTORATION FROM simulation_teleop.py ---
            if cmd == "reset":
                # Resets usually don't need a 'Before' image as they are state wipes
                self.before_img = None
                self.reset_env(lock_posture=self.lock_posture)
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp({"status": "reset_ok", "joints": norm_state.tolist()})

            elif cmd == "wild_randomize":
                self.before_img = None
                self.wild_reset()
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp(
                    {"status": "wild_randomize_ok", "joints": norm_state.tolist()}
                )

            elif cmd == "sync":
                self.recorder.force_sync()
                send_resp({"status": "sync_started"})

            elif cmd == "start_recording":
                self.recorder.start_episode(data.get("task", "Pick up red cube"))
                self.is_recording = True
                send_resp({"status": "recording_started"})

            elif cmd == "stop_recording":
                self.recorder.stop_episode()
                self.is_recording = False
                send_resp({"status": "recording_stopped"})

            elif cmd == "discard_recording":
                self.recorder.discard_episode()
                self.is_recording = False
                send_resp({"status": "recording_discarded"})

            elif cmd == "poll_status":
                send_resp({"status": "status_ok"})

            elif cmd == "ik_pickup":
                # Capture state before the multi-phase IK movement starts
                self.capture_before_state()
                self._handle_ik_pickup_logic(
                    phase=data.get("phase", 0), offset_cm=data.get("offset_cm", 5)
                )
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp({"status": "ik_pickup_ok", "joints": norm_state.tolist()})

            elif cmd == "set_cube_pose":
                self.capture_before_state()
                pose = np.array(data["pose"], dtype=np.float32)
                cube_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
                )
                if cube_id != -1:
                    q_idx = self.model.jnt_qposadr[cube_id]
                    self.data.qpos[q_idx : q_idx + 7] = pose
                    mujoco.mj_forward(self.model, self.data)
                send_resp({"status": "cube_pose_ok"})

            elif "target" in data:
                # Capture visual state BEFORE the motor dispatch
                self.capture_before_state()
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
                send_resp({"status": "step_ok"})

            elif cmd == "store_snapshot":
                # Standardized Snapshot logic from base script
                raw_state = self.get_state_32()
                norm_state = StandardScaler().scale_state(raw_state)
                physics = self.get_physics_state()
                self.renderer.update_scene(self.data, camera="world_center")
                rgb = self.renderer.render()
                img = Image.fromarray(rgb).resize((224, 224))
                img_list = np.array(img).transpose(2, 0, 1).tolist()
                snapshot = {
                    "observation.images.world_center": img_list,
                    "observation.state": norm_state.tolist(),
                    "action": norm_state.tolist(),
                    "progress": (1.0 - physics["target_dist"]) * 10.0,
                }
                snap_dir = os.path.join(
                    ROOT_DIR, "datasets", "vedpatwardhan", "gr1_reward_pred"
                )
                os.makedirs(snap_dir, exist_ok=True)
                existing = [f for f in os.listdir(snap_dir) if f.endswith(".json")]
                next_idx = len(existing)
                snap_path = os.path.join(snap_dir, f"{next_idx:04d}.json")
                with open(snap_path, "w") as f:
                    json.dump(snapshot, f)
                print(f"📸 Snapshot {next_idx:04d} stored.")
                send_resp({"status": "snapshot_ok", "index": next_idx})

            else:
                send_resp({"status": "unknown"})

            # Sync Rerun timeline and log all 5 camera views
            rr.set_time_sequence("step", int(self.data.time * 100))
            self.log_to_rerun()

            # Save interpretability snapshot to disk for every move (Except poll_status)
            if cmd != "poll_status":
                self.save_brain_snapshot()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interpretive Teleop Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained World Model checkpoint",
    )
    parser.add_argument(
        "--clt",
        type=str,
        default="clt_weights.pt",
        help="Path to the CLT transcoder weights",
    )
    parser.add_argument("--port", type=int, default=5556, help="ZMQ Port")
    parser.add_argument(
        "--lock-posture",
        action="store_true",
        default=True,
        help="Lock IK joints to specific targets",
    )
    args = parser.parse_args()

    def resolve_path(p):
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return os.path.abspath(p)
        fallback = os.path.join(ROOT_DIR, p)
        if os.path.exists(fallback):
            return fallback
        return os.path.abspath(p)

    m_path = resolve_path(args.model)
    c_path = resolve_path(args.clt)

    server = InterpretiveTeleopServer(
        model_path=m_path,
        clt_path=c_path,
        port=args.port,
        lock_posture=args.lock_posture,
    )
    server.run()
