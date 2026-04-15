import zmq
import msgpack
import torch
import numpy as np
import time
import traceback
import json
import os
import argparse
from transformers import PreTrainedModel
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from gr1_config import JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

# -----------------------------------------------------------------------------
# 1. HARDWARE & COMPATIBILITY
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force FlashAttention fallback for stability on certain GPUs
_orig_check = PreTrainedModel._check_and_adjust_attn_implementation


def patched_check_attn(self, attn_implementation, is_init_check):
    if attn_implementation == "flash_attention_2" or attn_implementation is None:
        return "sdpa"
    return _orig_check(self, attn_implementation, is_init_check)


PreTrainedModel._check_and_adjust_attn_implementation = patched_check_attn


# -----------------------------------------------------------------------------
# 2. UNIFIED MODEL SERVER (Final Protocol Alignment)
# -----------------------------------------------------------------------------
class GR00TInferenceServer:
    """
    Unified Inference Server for GR00T-N1.5.
    Uses native LeRobot pre/post processors to handle Min-Max normalization.
    """

    def __init__(
        self,
        embodiment_tag="gr1",
        port=5555,
        weights_path="nvidia/GR00T-N1.5-3B",
    ):
        print(f"--- GR00T N1.5 Server (Protocol: Min-Max Native) ---")
        self.port = port
        self.weights_path = weights_path
        self.tokenizer = None

        # Load Policy
        print(f"Loading weights from: {self.weights_path}")
        try:
            self.policy = GrootPolicy.from_pretrained(self.weights_path)
        except Exception as e:
            print(f"🔄 Retrying with base fallback due to: {e}")
            self.policy = GrootPolicy.from_pretrained("nvidia/GR00T-N1.5-3B")
            weights_file = os.path.join(self.weights_path, "model.safetensors")
            if os.path.exists(weights_file):
                from safetensors.torch import load_file

                state_dict = load_file(weights_file, device=DEVICE)
                self.policy.load_state_dict(state_dict, strict=False)

        self.policy.to(DEVICE)
        self.policy.eval()

        self.policy.config.embodiment_tag = embodiment_tag
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config
        )
        print("✅ Pre/Post Processors Initialized.")

    def log_diagnostics(self, processed_batch, actions_np, instruction):
        try:
            proc_state_np = processed_batch["observation.state"].cpu().detach().numpy()
            log_entry = {
                "timestamp": time.time(),
                "metrics": {
                    "input_norm_range": [
                        float(np.min(proc_state_np)),
                        float(np.max(proc_state_np)),
                    ],
                    "instruction": instruction,
                },
                "output_stats": [float(np.min(actions_np)), float(np.max(actions_np))],
            }
            log_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "vla_server_diag.json"
            )
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except:
            pass

    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{self.port}")
        print(f"🚀 GR-1 VLA Server listening on port {self.port}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)
                instruction = req.get("instruction", "Pick up the red cube")

                unpack_np = lambda d: (
                    np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
                )

                # Images
                cams = [
                    "world_top",
                    "world_left",
                    "world_right",
                    "world_center",
                    "world_wrist",
                ]
                img_list = []
                for cam in cams:
                    img = unpack_np(req.get(cam))
                    img_list.append(img)

                all_imgs_np = np.stack(img_list)
                all_images_t = torch.as_tensor(
                    all_imgs_np, dtype=torch.uint8, device=DEVICE
                ).permute(0, 3, 1, 2)

                # State Calibration handled simulation-side
                state_np = unpack_np(req.get("state"))
                state_raw_t = torch.tensor(
                    state_np, dtype=torch.float32, device=DEVICE
                ).unsqueeze(
                    0
                )  # [1, 32]

                # Batch
                batch = {
                    f"{OBS_IMAGES}.world_top": all_images_t[0].unsqueeze(0),
                    f"{OBS_IMAGES}.world_left": all_images_t[1].unsqueeze(0),
                    f"{OBS_IMAGES}.world_right": all_images_t[2].unsqueeze(0),
                    f"{OBS_IMAGES}.world_center": all_images_t[3].unsqueeze(0),
                    f"{OBS_IMAGES}.world_wrist": all_images_t[4].unsqueeze(0),
                    OBS_STATE: state_raw_t,
                    "task": instruction,
                    "embodiment_id": torch.tensor(
                        [24], dtype=torch.long, device=DEVICE
                    ),
                }

                processed_batch = self.preprocessor(batch)

                with torch.inference_mode():
                    action_chunk = self.policy.predict_action_chunk(processed_batch)

                # De-normalization and Ensure 2D [T, 32]
                actions_t = self.postprocessor(action_chunk)
                # actions_t shape can be [1, 16, 32] or [16, 32] depending on LeRobot version
                print("Actions Shape:", actions_t.shape)
                actions_np = (
                    actions_t.cpu().numpy()
                    if hasattr(actions_t, "cpu")
                    else np.array(actions_t)
                )

                # Squeeze out batch dim if present: [1, T, D] -> [T, D]
                if actions_np.ndim == 3:
                    actions_np = actions_np[0]  # -> [T, 32]
                elif actions_np.ndim == 1:
                    actions_np = actions_np[np.newaxis, :]  # -> [1, 32] fallback
                # actions_np should now be [T, 32] where T == action_horizon (16)

                print(
                    f"[{time.strftime('%H:%M:%S')}] 🧠 Inference. Norm Range: [{processed_batch[OBS_STATE].min():.2f}, {processed_batch[OBS_STATE].max():.2f}] | Actions: {actions_np.shape}"
                )

                self.log_diagnostics(processed_batch, actions_np, instruction)
                socket.send(
                    msgpack.packb({"action": actions_np.tolist()}, use_bin_type=True)
                )

            except Exception as e:
                print(f"❌ Server Error: {e}")
                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--port", "-p", type=int, default=5555)
    parser.add_argument("--tag", "-t", type=str, default="gr1")
    args = parser.parse_args()

    server = GR00TInferenceServer(
        embodiment_tag=args.tag, port=args.port, weights_path=args.weights
    )
    server.run()
