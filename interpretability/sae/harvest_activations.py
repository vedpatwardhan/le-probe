import os
import sys
import torch
import numpy as np
from pathlib import Path
import tqdm

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from lewm.goal_mapper import GoalMapper
from simulation_base import GR1MuJoCoBase
from interpretability.sae.activation_harvester import ActivationHarvester

def run_harvest(model_path, gallery_path, output_path, num_episodes=10, steps_per_episode=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize simulation (headless)
    print("🎮 Initializing Simulation...")
    sim = GR1MuJoCoBase()
    
    # 2. Initialize Brain
    print("🧠 Loading LeWM Model...")
    agent = GoalMapper(model_path, dataset_root=ROOT_DIR)
    
    # 3. Setup Harvester
    harvester = ActivationHarvester()
    # Target the latent bottleneck and the last predictor layer
    target_layers = {
        "latent_bottleneck": "model.projector", # The 192d projected latent
        "predictor_out": "model.predictor.transformer.norm" # Final transformer state
    }
    harvester.register(agent.model, target_layers)
    
    all_activations = {k: [] for k in target_layers.keys()}
    
    try:
        for ep in range(num_episodes):
            print(f"🎬 Episode {ep+1}/{num_episodes}")
            sim.reset()
            
            # Reset agent state (initial pose)
            raw_sim_state = sim.get_state_32()
            norm_state = agent.transform({"pixels": np.zeros((224, 224, 3))}) # Just to get scaler? No, scaler is in GoalMapper
            # Actually GoalMapper has no scaler. StandardScaler is used in simulation_lewm.py
            
            for step in tqdm.tqdm(range(steps_per_episode)):
                # Simplified interaction for harvesting
                sim.renderer.update_scene(sim.data, camera="world_center")
                img = sim.renderer.render()
                
                # Encode and step through model
                batch = agent.transform({"pixels": img})
                image_t = batch["pixels"].to(device).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 224, 224)
                
                with torch.inference_mode():
                    # This triggers the forward hooks
                    info = agent.model.encode({"pixels": image_t})
                    # We can also run a predictor step if we have actions
                    # For simplicity, we just harvest the encoding for now
                    # (Phase 1 focus: The Latent Bottleneck)
                
                # Execute random or heuristic actions to get diversity
                action = np.random.uniform(-0.1, 0.1, 32)
                sim.process_target_32(action)
                sim.dispatch_action(action, sim.last_target_q, n_steps=10)
                
            # Flush activations to memory
            ep_data = harvester.flush()
            for k, v in ep_data.items():
                all_activations[k].append(v)
                
    finally:
        harvester.remove_hooks()
    
    # 4. Save to Disk
    print(f"💾 Saving Harvest to {output_path}...")
    save_dict = {k: torch.cat(v, dim=0) for k, v in all_activations.items()}
    torch.save(save_dict, output_path)
    print("✅ Harvest Complete.")

if __name__ == "__main__":
    # Example usage (would be triggered by user after review)
    # run_harvest("path/to/model.ckpt", "path/to/gallery.pth", "activations.pt")
    print("🛠 Activation Harvester Ready.")
    print("Usage: python interpretability/sae/harvest_activations.py --model <path>")
