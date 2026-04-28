import os
import sys
import argparse
import torch
import zmq
import msgpack
import time
import numpy as np
from pathlib import Path
from gymnasium.spaces import Box

# --- Path Stabilization (Handles the nested directory structure) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for p in [ROOT_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
# -------------------------------------------------------------------

from lewm.lewm_server import LEWMInferenceServer, PORT, MockConfig
from interpretability.steering.clt_steering import CLTSteerer


class SteeredLEWMServer(LEWMInferenceServer):
    def __init__(
        self, model_path, gallery_path, clt_path, boost_features=[90, 358], factor=5.0
    ):
        super().__init__(model_path, gallery_path)

        print(f"🧠 Initializing Latent Steering (CLT)...")
        self.steerer = CLTSteerer(
            self.agent.model,
            clt_path,
            device=str(next(self.agent.model.parameters()).device),
        )

        # Apply the 'Reach Intent' boosts
        for feat in boost_features:
            self.steerer.set_boost(feat, factor)

        # Attach to the handoff point (Encoder Projector)
        self.steerer.attach(layer_path="projector")
        print(f"✅ Steering Active: Boosting Features {boost_features} by {factor}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="le-probe/lewm/goal_gallery.pth")
    parser.add_argument("--clt", type=str, default="clt_weights.pt")
    parser.add_argument("--boost", type=float, default=5.0, help="Steering factor")
    args = parser.parse_args()

    # Smart Path Expansion: Check absolute first, then relative to CWD
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
    g_path = resolve_path(args.gallery)
    c_path = resolve_path(args.clt)

    print(f"📂 Model Path: {m_path}")
    print(f"📂 Gallery Path: {g_path}")
    print(f"📂 CLT Path: {c_path}")

    server = SteeredLEWMServer(
        model_path=m_path, gallery_path=g_path, clt_path=c_path, factor=args.boost
    )
    server.run()
