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

# --- FAIL-SAFE STATISTICS (32-dim Compact Protocol) ---
# TODO: GET RID OF THIS AND FETCH THESE VALUES PROGRAMMATICALLY
# These are extracted directly from the gr1_pickup_compact_h264 dataset stats.json
# to ensure perfect un-scaling even if local checkpoint stats fail to load.
ACTION_MEAN_32 = [
    -0.00885169580578804,
    -0.0004610285977832973,
    -0.006922537460923195,
    0.0008136362303048372,
    0.0016450101975351572,
    -0.0023524146527051926,
    0.02196965366601944,
    -0.1878836452960968,
    0.4360756278038025,
    -0.42222365736961365,
    0.15410202741622925,
    -0.34708738327026367,
    -0.5762120485305786,
    -0.029333386570215225,
    0.004611211828887463,
    0.013999348506331444,
    0.015579478815197945,
    -0.014618994668126106,
    0.0096802469342947,
    -0.019505003467202187,
    0.0,
    0.0,
    0.0,
    -0.11612068861722946,
    0.3479006290435791,
    -0.16578777134418488,
    -0.15538309514522552,
    -0.15033003687858582,
    -0.1550322026014328,
    0.0,
    0.0,
    0.0,
]
ACTION_STD_32 = [
    0.04668724164366722,
    0.05634459853172302,
    0.038286760449409485,
    0.05131921544671059,
    0.03601062670350075,
    0.12776821851730347,
    0.1863974630832672,
    0.0816228985786438,
    0.20345927774906158,
    0.1388711780309677,
    0.0957081988453865,
    0.13011333346366882,
    0.27658334374427795,
    1.0996264219284058,
    0.12731678783893585,
    0.18531660735607147,
    0.13456332683563232,
    0.13627128303050995,
    0.13853468000888824,
    0.1295442134141922,
    0.0,
    0.0,
    0.0,
    0.1718074232339859,
    0.42163020372390747,
    0.17843188345432281,
    0.18690456449985504,
    0.19115972518920898,
    0.18743643164634705,
    0.0,
    0.0,
    0.0,
]

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

    def __init__(
        self,
        embodiment_tag="gr1",
        port=5555,
        weights_path="nvidia/GR00T-N1.5-3B",
    ):
        print(f"--- Initializing GR00T N1.5 Server (Embodiment: {embodiment_tag}) ---")
        self.port = port
        self.weights_path = weights_path
        self.tokenizer = None

        # Load Policy (Supports both HF Repo ID and local paths)
        print(f"Loading weights from: {self.weights_path}")

        # 1. Check if it's a local path and validate its existence
        is_local = os.path.isdir(self.weights_path) or (
            "/" in self.weights_path and not self.weights_path.count("/") == 1
        )
        if is_local:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    f"❌ Local weights path '{self.weights_path}' not found. "
                    "If this was meant to be a Hugging Face repo ID, ensure it follows 'namespace/repo' format. "
                    "If it is a relative path, ensure you are running the server from the correct directory."
                )
            self.weights_path = os.path.abspath(self.weights_path)
            print(f"Detected local path, resolved to: {self.weights_path}")
        else:
            print("Detected potential Hugging Face Repo ID...")

        # 2. Attempt to load the Policy
        try:
            print(f"Attempting to load policy from: {self.weights_path}")
            self.policy = GrootPolicy.from_pretrained(self.weights_path)
        except Exception as e:
            # 3. Fallback: Load base model and override weights if local path fails as a policy
            if is_local:
                print(
                    f"⚠️  Failed to load full policy from '{self.weights_path}' (Error: {e})."
                )
                print("🔄 Falling back to base GR00T-N1.5 model with custom weights...")
                self.policy = GrootPolicy.from_pretrained("nvidia/GR00T-N1.5-3B")

                # Find weight file (safetensors preferred)
                weights_file = os.path.join(self.weights_path, "model.safetensors")
                if not os.path.exists(weights_file):
                    weights_file = os.path.join(self.weights_path, "pytorch_model.bin")

                if os.path.exists(weights_file):
                    print(f"📥 Overriding state_dict from: {weights_file}")
                    if weights_file.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        state_dict = load_file(weights_file, device=DEVICE)
                    else:
                        state_dict = torch.load(weights_file, map_location=DEVICE)

                    # GrootPolicy wraps the model in _groot_model
                    try:
                        # Try loading into the policy wrapper first (in case keys are prefixed)
                        self.policy.load_state_dict(state_dict, strict=False)
                    except:
                        # Otherwise load directly into the model backbone
                        self.policy._groot_model.load_state_dict(
                            state_dict, strict=False
                        )
                    print("✅ Weights successfully overridden.")
                else:
                    raise FileNotFoundError(
                        f"❌ No weights file (model.safetensors or pytorch_model.bin) found in {self.weights_path}"
                    )
            else:
                # If it's a Repo ID and it fails, we just re-raise
                print(f"❌ Failed to load policy from HuggingFace: {e}")
                raise e

        self.policy.to(DEVICE)
        self.policy.eval()

        # Setup Pre/Post Processors
        # Setting the embodiment tag ensures the pipeline uses the correct joint limits (ID 24 for gr1)
        self.policy.config.embodiment_tag = embodiment_tag

        print("Creating processors...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config
        )

        # Detect if we have legitimate statistical normalization for the state.
        # Check for the existence of policy_preprocessor.json which and indicates a fine-tuned LeRobot checkpoint.
        self.has_stats = False
        if os.path.isdir(self.weights_path) and os.path.exists(
            os.path.join(self.weights_path, "policy_preprocessor.json")
        ):
            self.has_stats = True

        # Fallback to introspection if the file check fails
        if (
            not self.has_stats
            and hasattr(self.preprocessor, "feature_extractors")
            and OBS_STATE in self.preprocessor.feature_extractors
        ):
            fe = self.preprocessor.feature_extractors[OBS_STATE]
            if hasattr(fe, "do_normalize") and fe.do_normalize:
                # Base model doesn't "normalize" in the feature extractor typically,
                # we want to catch if the checkpoint has provided its own stats.
                self.has_stats = True
            elif hasattr(fe, "mean") and fe.mean is not None:
                self.has_stats = True

        print(
            f"✅ Preprocessor Check: has_stats={self.has_stats} (Fine-tuned Mode: {self.has_stats})"
        )

        # Extract Action Stats for manual un-scaling (bypasses LeRobot Pipeline type-checking errors)
        self.action_mean = None
        self.action_std = None
        if self.has_stats:
            print("🔍 Searching for action statistics...")

            # 1. Try to find stats in the policy configuration (Standard HF format)
            if (
                hasattr(self.policy.config, "stats")
                and "action" in self.policy.config.stats
            ):
                stats = self.policy.config.stats["action"]
                self.action_mean = torch.tensor(stats["mean"], device=DEVICE)
                self.action_std = torch.tensor(stats["std"], device=DEVICE)
                print("   ✅ Found 'action' stats in policy.config.stats")

            # 2. Try to find stats in dataset_config (Local training format)
            elif hasattr(self.policy.config, "dataset_config") and hasattr(
                self.policy.config.dataset_config, "stats"
            ):
                stats = self.policy.config.dataset_config.stats.get("action")
                if stats:
                    self.action_mean = torch.tensor(stats["mean"], device=DEVICE)
                    self.action_std = torch.tensor(stats["std"], device=DEVICE)
                    print(
                        "   ✅ Found 'action' stats in policy.config.dataset_config.stats"
                    )

            # 3. Fallback to Hardcoded Fail-safe for GR1 Compact Protocol
            if self.action_mean is None and embodiment_tag == "gr1":
                print(
                    "   🛠️  Applying Fail-safe Static Constants (GR1 Compact Protocol)"
                )
                self.action_mean = torch.tensor(ACTION_MEAN_32, device=DEVICE)
                self.action_std = torch.tensor(ACTION_STD_32, device=DEVICE)

            if self.action_mean is None:
                print(
                    "   ⚠️ WARNING: No statistics found for 'action'. Output will use fallback scaling."
                )
            else:
                print(
                    f"   📊 Action Stats: Mean range [{self.action_mean.min():.3f}, {self.action_mean.max():.3f}], Std range [{self.action_std.min():.3f}, {self.action_std.max():.3f}]"
                )

        # Pre-cache Joint Limits on Device (for Base Model Mode)
        self.joint_min_t = torch.tensor(
            JOINT_LIMITS_MIN, device=DEVICE, dtype=torch.float32
        )
        self.joint_max_t = torch.tensor(
            JOINT_LIMITS_MAX, device=DEVICE, dtype=torch.float32
        )
        self.joint_rng_t = torch.maximum(
            torch.tensor(1e-4, device=DEVICE), self.joint_max_t - self.joint_min_t
        )

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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(base_dir, "inference_history.json")
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
                image_world_top_np = unpack_np(req.get("world_top"))
                image_world_left_np = unpack_np(req.get("world_left"))
                image_world_right_np = unpack_np(req.get("world_right"))
                image_world_center_np = unpack_np(req.get("world_center"))
                image_world_wrist_np = unpack_np(req.get("world_wrist"))
                state_np = unpack_np(req.get("state"))
                instruction = req.get("instruction", "Pick up the red cube")

                # Transform images ((224, 224, 3) -> (3, 224, 224))
                all_imgs_np = np.stack(
                    [
                        image_world_top_np,
                        image_world_left_np,
                        image_world_right_np,
                        image_world_center_np,
                        image_world_wrist_np,
                    ]
                )
                all_images_t = torch.as_tensor(all_imgs_np, dtype=torch.uint8).permute(
                    0, 3, 1, 2
                )
                (
                    image_world_top_t,
                    image_world_left_t,
                    image_world_right_t,
                    image_world_center_t,
                    image_world_wrist_t,
                ) = all_images_t

                # Create 32-dim state (Compact Protocol 🧪)
                # We skip Rosetta expansion to ensure fingers (indices 23-28) are within the model's 32-dim window.
                state_raw_t = torch.tensor(state_np, dtype=torch.float32, device=DEVICE)

                if self.has_stats:
                    # ✅ FINE-TUNED FIX: Apply manual Z-scoring (Scale to Gaussian)
                    # Training logs show ACTION: IDENTITY, but STATE: MEAN_STD.
                    # Since state/action are symmetric, we use action_stats for the state.
                    if self.action_mean is not None:
                        state_t = (state_raw_t - self.action_mean) / self.action_std
                        print(
                            f"   [📊] Input Normalization Applied. Range: [{state_t.min():.3f}, {state_t.max():.3f}]"
                        )
                    else:
                        state_t = state_raw_t
                else:
                    # Base model lacks stats; we manually scale radians to [-1, 1] bounds
                    print("   [INFO] Applying manual [-1, 1] scaling (Base Model Mode)")
                    state_t = (
                        2.0 * (state_raw_t - self.joint_min_t) / self.joint_rng_t - 1.0
                    )
                    state_t = torch.clamp(state_t, -1.1, 1.1)

                # Prepared batch for preprocessor
                batch = {
                    f"{OBS_IMAGES}.world_top": image_world_top_t,
                    f"{OBS_IMAGES}.world_left": image_world_left_t,
                    f"{OBS_IMAGES}.world_right": image_world_right_t,
                    f"{OBS_IMAGES}.world_center": image_world_center_t,
                    f"{OBS_IMAGES}.world_wrist": image_world_wrist_t,
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

                if self.has_stats:
                    # ✅ FINE-TUNED FIX: Set Output to IDENTITY
                    # Training logs show ACTION: IDENTITY. The model speaks Physical Radians.
                    actions_np = action_chunk[0].cpu().numpy()
                else:
                    # Base model outputs [-1, 1]; we manually scale back to Radians using Joint Limits
                    actions_raw_np = action_chunk[0]  # On DEVICE
                    print(
                        "   [INFO] Applying manual Radians un-scaling (Base Model Mode)"
                    )
                    actions_t = (
                        actions_raw_np + 1.0
                    ) / 2.0 * self.joint_rng_t + self.joint_min_t
                    actions_np = actions_t.cpu().numpy()
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
                print(f"❌ Error in Inference Loop: {e}")
                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T-N1.5 Inference Server")
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="nvidia/GR00T-N1.5-3B",
        help="Hugging Face repo ID or local path to the fine-tuned weights (e.g. /path/to/checkpoint/pretrained_model).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5555,
        help="Port to listen on (default: 5555).",
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default="gr1",
        help="Embodiment tag (default: gr1).",
    )
    args = parser.parse_args()

    server = GR00TInferenceServer(
        embodiment_tag=args.tag, port=args.port, weights_path=args.weights
    )
    server.run()
