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
        embodiment_tag="new_embodiment",
        port=5555,
        weights_path="nvidia/GR00T-N1.5-3B",
    ):
        print(f"--- GR00T N1.5 Server (Protocol: Universal Handshake) ---")
        self.port = port
        self.weights_path = weights_path
        self.tokenizer = None
        self.device = DEVICE

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

        # Protocol Detection
        self.normalization_mapping = getattr(
            self.policy.config, "normalization_mapping", {}
        )
        print(f"🔍 Protocol Detected: {self.normalization_mapping}")

        self.policy.config.embodiment_tag = embodiment_tag

        # Embodiment ID Mapping
        self.embodiment_mapping = {
            "new_embodiment": 31,
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }
        self.emb_id = self.embodiment_mapping.get(embodiment_tag, 0)
        print(f"🆔 Embodiment ID set to: {self.emb_id} ({embodiment_tag})")

        # Official Processor Factory
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config
        )
        print("✅ Pre/Post Processors Initialized from Policy Config.")

    def construct_raw_batch(self, req, instruction):
        """Replicates LeRobot Dataset/Trainer batch construction (Nested EnvTransition)."""
        unpack_np = lambda d: (
            np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
        )

        # 1. Observation Dict
        obs_dict = {}
        cams = ["world_top", "world_left", "world_right", "world_center", "world_wrist"]
        for cam in cams:
            key = f"observation.images.{cam}"
            val = req.get(key) if key in req else req.get(cam)
            if val is None:
                raise ValueError(f"Missing camera data for: {cam}")
            img_np = unpack_np(val)
            obs_dict[key] = torch.as_tensor(
                img_np, dtype=torch.uint8, device=DEVICE
            ).unsqueeze(0)

        state_np = unpack_np(req.get("state"))
        obs_dict[OBS_STATE] = torch.tensor(
            state_np, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        # 2. Construct Nested Transition
        transition = {
            "observation": obs_dict,
            "complementary_data": {
                "task": instruction,
                # embodiment_id is overwritten by GrootPackInputsStep based on tag
            },
        }
        return transition

    def log_diagnostics(
        self, batch, processed_batch, action_chunk, actions_t, instruction
    ):
        try:
            # Helper to convert tensors/dicts of tensors to serializable lists
            def to_serializable(data):
                if isinstance(data, torch.Tensor):
                    return data.detach().cpu().numpy().tolist()
                if isinstance(data, dict):
                    return {k: to_serializable(v) for k, v in data.items()}
                if isinstance(data, (np.ndarray, np.generic)):
                    return data.tolist()
                return data

            log_entry = {
                "timestamp": time.time(),
                "instruction": instruction,
                "protocol": str(self.normalization_mapping),
                "lifecycle": {
                    "raw_batch": to_serializable(batch),
                    "processed_batch": to_serializable(processed_batch),
                    "action_chunk_raw": to_serializable(action_chunk),
                    "action_t_final": to_serializable(actions_t),
                },
            }

            # Use a specific filename for the lifecycle log
            log_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "vla_lifecycle_audit.json"
            )
            # Append each inference step as a new line (JSONL style for large files)
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"⚠️ Diagnostic logging failed: {e}")

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

                # 1. Construct Raw Batch
                batch = self.construct_raw_batch(req, instruction)

                # 2. Pre-process (Canonical Callstack)
                processed_batch = self.preprocessor(batch)

                # 3. Model Inference
                with torch.inference_mode():
                    action_chunk = self.policy.predict_action_chunk(processed_batch)

                # 4. Post-process (Canonical Callstack)
                # Note: postprocessor handles tensor -> EnvTransition -> processing -> tensor conversion
                actions_t = self.postprocessor(action_chunk)

                # Convert to numpy for transport
                actions_np = actions_t.cpu().numpy()

                # LOGGING: Support both chunked and single-step returns
                if actions_np.ndim == 3:
                    final_actions = actions_np[0]
                elif actions_np.ndim == 2:
                    final_actions = actions_np
                else:
                    # Single step case (B, D) -> (D,)
                    final_actions = actions_np[0]

                print(
                    f"[{time.strftime('%H:%M:%S')}] 🧠 Inference (Aligned Stack). Actions: {final_actions.shape}"
                )

                self.log_diagnostics(
                    batch, processed_batch, action_chunk, actions_t, instruction
                )
                socket.send(
                    msgpack.packb({"action": final_actions.tolist()}, use_bin_type=True)
                )

                self.log_diagnostics(
                    batch, processed_batch, action_chunk, actions_t, instruction
                )
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
