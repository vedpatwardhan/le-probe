
import zmq
import msgpack
import torch
import numpy as np
import time
import traceback
import json
import os
from transformers import PreTrainedModel
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

# -----------------------------------------------------------------------------
# 1. HARDWARE & COMPATIBILITY
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Force FlashAttention fallback for stability on certain GPUs
_orig_check = PreTrainedModel._check_and_adjust_attn_implementation


def patched_check_attn(self, attn_implementation, is_init_check):
    if attn_implementation == "flash_attention_2":
        return "sdpa"
    return _orig_check(self, attn_implementation, is_init_check)


PreTrainedModel._check_and_adjust_attn_implementation = patched_check_attn

# -----------------------------------------------------------------------------
# 2. UNIFIED MODEL SERVER (Simplified Inference Logic)
# -----------------------------------------------------------------------------
class GR00TInferenceServer:
    """
    Unified Inference Server for GR00T-N1.5.
    Uses the factory pre/post processors to handle SigLIP normalization
    and embodiment-specific joint mapping.
    """

    def __init__(self, embodiment_tag="gr1", port=5555):
        print(f"--- Initializing GR00T N1.5 Server (Embodiment: {embodiment_tag}) ---")
        self.port = port
        self.model_repo = "nvidia/GR00T-N1.5-3B"
        self.tokenizer = None

        # Load Policy
        self.policy = GrootPolicy.from_pretrained(self.model_repo)
        self.policy.to(DEVICE)
        self.policy.eval()

        # Setup Pre/Post Processors
        # Setting the embodiment tag ensures the pipeline uses the correct joint limits (ID 24 for gr1)
        self.policy.config.embodiment_tag = embodiment_tag

        print("Creating processors...")
        self.preprocessor, _ = make_pre_post_processors(policy_cfg=self.policy.config)

        # Extract Tokenizer (for Debugging)
        self.tokenizer = None
        for step in self.preprocessor.steps:
            if hasattr(step, "proc") and hasattr(step.proc, "tokenizer"):
                self.tokenizer = step.proc.tokenizer
                break
            elif hasattr(step, "tokenizer"):
                self.tokenizer = step.tokenizer
                break

        print("✅ Model & Preprocessor Ready!")

        print(
            f"--- Policy Config Embodiment: {getattr(self.policy.config, 'embodiment_tag', 'None')} ---"
        )

    def log_diagnostics(self, batch, processed_batch, actions_np, instruction):
        # General serialization function
        def serialize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        # Prepare dict for logging current step
        proc_state_np = processed_batch["observation.state"].cpu().detach().numpy()
        log_entry = {
            "timestamp": time.time(),
            "batch": serialize(batch),
            "processed_metrics": {
                "observation.state": proc_state_np.tolist(),
                "task": instruction,
                "bounds": {
                    "min": float(np.min(proc_state_np)),
                    "max": float(np.max(proc_state_np)),
                    "mean": float(np.mean(proc_state_np)),
                },
                "has_dataset_stats": getattr(self.preprocessor, "stats", None)
                is not None,
            },
            "output": serialize(actions_np),
            "output_stats": {
                "min": float(np.min(actions_np)),
                "max": float(np.max(actions_np)),
                "mean": float(np.mean(actions_np)),
            },
        }

        # Load existing inference history if any
        log_file = "inference_history.json"
        history = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                history = json.load(f)

        # Write updated history
        history.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)


    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{self.port}")
        print(f"🚀 GR-1 Model Server listening on port {self.port}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)
                print(
                    f"[{time.strftime('%H:%M:%S')}] 📥 Incoming request: '{req.get('instruction')}'"
                )

                # Unpack Inputs from Simulation Client
                unpack_np = lambda d: (
                    np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])
                )
                image_head_np = unpack_np(req.get("head"))
                image_world_left_np = unpack_np(req.get("world_left"))
                image_world_right_np = unpack_np(req.get("world_right"))
                image_world_center_np = unpack_np(req.get("world_center"))
                state_np = unpack_np(req.get("state"))
                instruction = req.get("instruction", "Perform the task.")

                # Transform images ((224, 224, 3) -> (3, 224, 224))
                all_imgs_np = np.stack(
                    [
                        image_head_np,
                        image_world_left_np,
                        image_world_right_np,
                        image_world_center_np,
                    ]
                )
                all_imgs_t = torch.as_tensor(all_imgs_np, dtype=torch.uint8).permute(
                    0, 3, 1, 2
                )
                img_head_t, img_left_t, img_right_t, img_center_t = all_imgs_t

                # Create 64-dim state (Rosetta Protocol 🧪)
                state_full = np.zeros(64, dtype=np.float32)
                state_full[0:7] = state_np[0:7]         # Left Arm
                state_full[7:13] = state_np[7:13]       # Left Hand
                state_full[19:22] = state_np[13:16]     # Neck/Head
                state_full[22:29] = state_np[16:23]     # Right Arm
                state_full[29:35] = state_np[23:29]     # Right Hand
                state_full[41:44] = state_np[29:32]     # Waist
                state_t = torch.tensor(state_full, dtype=torch.float32)

                # Prepared batch for preprocessor
                batch = {
                    f"{OBS_IMAGES}.head": image_head_t,
                    f"{OBS_IMAGES}.world_left": image_world_left_t,
                    f"{OBS_IMAGES}.world_right": image_world_right_t,
                    f"{OBS_IMAGES}.world_center": image_world_center_t,
                    OBS_STATE: state_t,
                    "task": instruction,
                    "embodiment_id": torch.tensor(
                        [24], dtype=torch.long, device=DEVICE
                    ),
                }
                start_time = time.time()

                # Preprocessing for tokenization and move to device (no normalization)
                processed_batch = self.preprocessor(batch)
                for k, v in processed_batch.items():
                    if isinstance(v, torch.Tensor):
                        processed_batch[k] = v.to(DEVICE)

                # Run inference
                print("   [DEBUG] Running Model Inference...")
                with torch.inference_mode():
                    action_chunk = self.policy.predict_action_chunk(processed_batch)
                actions_np = action_chunk[0].cpu().numpy()  # (16, 32)
                inference_time = time.time() - start_time
                print(f"   [DEBUG] Inference Time: {inference_time:.2f} seconds")

                # Diagnostic Data Logging
                self.log_diagnostics(batch, processed_batch, actions_np, instruction)

                # Return canonical 32-dim action (Arms, Hands, Neck, Waist)
                payload = {
                    "action": actions_np.tolist(),
                    "diagnostics": {
                        "instruction": instruction,
                        "inference_time_ms": int((time.time() - start_time) * 1000),
                        "chunk_size": actions_np.shape[0],
                    },
                }
                print(
                    f"   [DEBUG] Sending response back. Chunk Size: {actions_np.shape[0]}"
                )
                socket.send(msgpack.packb(payload, use_bin_type=True))

            except Exception as e:
                import traceback

                print(f"❌ Error in Inference Loop: {e}")
                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


if __name__ == "__main__":
    server = GR00TInferenceServer(embodiment_tag="gr1", port=5555)
    server.run()
