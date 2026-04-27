import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import os
import sys

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from interpretability.clt.clt_model import CrossLayerTranscoder

def train_clt(activation_path, layer_L, layer_L_plus_1, d_sae=12288, lr=3e-4, l1_coeff=0.01, batch_size=4096, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    print(f"📂 Loading activations from {activation_path}...")
    data = torch.load(activation_path, map_location="cpu")
    
    acts_L = data[layer_L].float()
    acts_target = data[layer_L_plus_1].float()
    
    # Flatten (B, T, D) -> (B*T, D)
    if acts_L.ndim == 3:
        acts_L = acts_L.reshape(-1, acts_L.size(-1))
    if acts_target.ndim == 3:
        acts_target = acts_target.reshape(-1, acts_target.size(-1))
    
    print(f"📊 Dataset size: {acts_L.shape}")
    dataset = TensorDataset(acts_L, acts_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize CLT
    d_model = acts_L.size(-1)
    clt = CrossLayerTranscoder(d_model=d_model, d_sae=d_sae, l1_coeff=l1_coeff).to(device)
    optimizer = optim.Adam(clt.parameters(), lr=lr)
    
    # 3. Training Loop
    print(f"🚀 Training CLT (Transcoding {layer_L} -> {layer_L_plus_1})...")
    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch in pbar:
            x_L, x_target = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            out = clt(x_L, x_target)
            loss = out["loss"]
            
            loss.backward()
            optimizer.step()
            
            clt.normalize_decoder()
            
            total_loss += loss.item()
            pbar.set_postfix({"L2": f"{out['l2_loss'].item():.4f}", "L1": f"{out['l1_loss'].item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} complete. Avg Loss: {total_loss/len(loader):.6f}")

    # 4. Save CLT
    output_path = f"clt_{layer_L}_to_{layer_L_plus_1}.pt"
    torch.save({
        "state_dict": clt.state_dict(),
        "config": {"d_model": d_model, "d_sae": d_sae, "l1_coeff": l1_coeff}
    }, output_path)
    print(f"💾 CLT saved to {output_path}")

if __name__ == "__main__":
    print("🚂 CLT Trainer Ready.")
