#!/usr/bin/env python3
"""
Train Hypergraph Neural Network (HGNN) and MLP baseline for CA state prediction.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_data(dataset_path="artifacts/ca_transitions.npz",
              incidence_path="artifacts/hypergraph_incidence.npz",
              train_ratio=0.8):
    """Load dataset and hypergraph incidence matrix."""
    data = np.load(dataset_path)
    X_state = data['X_state']          # (n, 8)
    X_conf = data['X_conf']            # (n, 45)
    X_age = data['X_age']              # (n, 1)
    Y_state = data['Y_state']          # (n, 8)
    
    # Load incidence matrix
    inc = np.load(incidence_path)
    H = inc['H']                       # (8, 45)
    variable_names = inc['variable_names']
    rule_names = inc['rule_names']
    
    # Concatenate input features: state + conf + age
    X = np.concatenate([X_state, X_conf, X_age], axis=1)  # (n, 8+45+1) = (n, 54)
    Y = Y_state
    
    # Normalize state inputs and outputs (mean/std per variable)
    state_mean = X_state.mean(axis=0, keepdims=True)
    state_std = X_state.std(axis=0, keepdims=True) + 1e-8
    X_state_norm = (X_state - state_mean) / state_std
    Y_state_norm = (Y_state - state_mean) / state_std
    
    # Normalize age (mean/std)
    age_mean = X_age.mean(axis=0, keepdims=True)
    age_std = X_age.std(axis=0, keepdims=True) + 1e-8
    X_age_norm = (X_age - age_mean) / age_std
    
    # Reconstruct normalized X
    X_norm = np.concatenate([X_state_norm, X_conf, X_age_norm], axis=1)
    
    # Split train/test
    n = X.shape[0]
    n_train = int(train_ratio * n)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y_state_norm[train_idx], dtype=torch.float32)
    X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
    Y_test = torch.tensor(Y_state_norm[test_idx], dtype=torch.float32)
    
    # Convert H to torch tensor
    H = torch.tensor(H, dtype=torch.float32)
    
    data_info = {
        'state_mean': state_mean,
        'state_std': state_std,
        'age_mean': age_mean,
        'age_std': age_std,
        'variable_names': variable_names,
        'rule_names': rule_names,
        'H': H,
    }
    
    return X_train, Y_train, X_test, Y_test, data_info

class MLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class HypergraphLayer(nn.Module):
    """Hypergraph convolution layer with dynamic hyperedge weights.
    
    Input: node features X (batch x n_vars x feat_dim),
           hyperedge weights W (batch x n_edges)
           incidence matrix H (n_vars x n_edges)
    
    Propagation: X' = σ( D_v^{-1/2} H diag(W) D_e^{-1} H^T D_v^{-1/2} X Θ )
    
    Implementation: compute normalized adjacency A_norm = D_v^{-1/2} H diag(W) D_e^{-1} H^T D_v^{-1/2}
    Then X' = σ( A_norm X Θ )
    """
    def __init__(self, in_features, out_features, n_vars, n_edges, bias=True):
        super().__init__()
        self.n_vars = n_vars
        self.n_edges = n_edges
        self.theta = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.ReLU()
        
        # Precompute D_v and D_e (static)
        # They are computed from H (binary incidence) and passed in forward.
        # We'll store them as buffers after initialization.
        self.register_buffer('D_v', None)
        self.register_buffer('D_e', None)
        self.register_buffer('H', None)
    
    def set_incidence(self, H):
        """Set incidence matrix and compute degree matrices."""
        # H: (n_vars, n_edges)
        self.H = H
        D_v = H.sum(dim=1)  # (n_vars,)
        D_e = H.sum(dim=0)  # (n_edges,)
        self.D_v = D_v
        self.D_e = D_e
    
    def forward(self, X, W):
        """
        Args:
            X: (batch, n_vars, in_features)
            W: (batch, n_edges) hyperedge weights (rule confidences)
        Returns:
            (batch, n_vars, out_features)
        """
        batch_size = X.shape[0]
        # Expand H and degrees to batch dimension
        H = self.H.unsqueeze(0)  # (1, n_vars, n_edges)
        D_v = self.D_v.unsqueeze(0).unsqueeze(-1)  # (1, n_vars, 1)
        D_e = self.D_e.unsqueeze(0).unsqueeze(0)   # (1, 1, n_edges)
        W = W.unsqueeze(1)  # (batch, 1, n_edges)
        
        # D_v^{-1/2}
        D_v_inv_sqrt = torch.where(D_v > 0, 1.0 / torch.sqrt(D_v), torch.zeros_like(D_v))
        # D_e^{-1}
        D_e_inv = torch.where(D_e > 0, 1.0 / D_e, torch.zeros_like(D_e))
        
        # Compute A_norm = D_v^{-1/2} H diag(W) D_e^{-1} H^T D_v^{-1/2}
        # Step 1: H * diag(W) = H * W (broadcast elementwise)
        HW = H * W  # (batch, n_vars, n_edges)
        # Step 2: HW * D_e_inv
        HW_de = HW * D_e_inv
        # Step 3: (HW_de) * H^T
        A = torch.matmul(HW_de, H.transpose(1, 2))  # (batch, n_vars, n_vars)
        # Step 4: left/right multiply D_v^{-1/2}
        A_norm = D_v_inv_sqrt * A * D_v_inv_sqrt.transpose(1, 2)
        
        # Apply convolution: X' = A_norm X Θ
        X_transformed = torch.matmul(A_norm, self.theta(X))
        return self.activation(X_transformed)

class HGNN(nn.Module):
    """Hypergraph Neural Network for CA state prediction.
    
    Input: state (batch, 8), conf (batch, 45), age (batch, 1)
    Output: next state (batch, 8)
    
    Architecture:
    1. Embed state, conf, age into node features.
    2. Stack hypergraph layers.
    3. Readout node features to state prediction.
    """
    def __init__(self, H, hidden_dim=64, n_layers=2):
        super().__init__()
        n_vars, n_edges = H.shape
        self.n_vars = n_vars
        self.n_edges = n_edges
        
        # Initial embedding: each variable gets its own state value + global age + rule confidences?
        # We'll treat each variable as a node with initial feature = state value.
        # We'll also incorporate rule confidences as hyperedge weights.
        # Age can be added as extra node feature (broadcast to all nodes).
        self.state_embed = nn.Linear(1, hidden_dim)  # each state value scalar -> hidden
        self.age_embed = nn.Linear(1, hidden_dim)    # age -> hidden (broadcast)
        
        # Hypergraph layers
        self.hg_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hg_layers.append(HypergraphLayer(hidden_dim, hidden_dim, n_vars, n_edges))
        
        # Readout: each node -> scalar prediction
        self.readout = nn.Linear(hidden_dim, 1)
        
        # Store incidence matrix
        self.register_buffer('H', H)
        # Initialize hypergraph layers with incidence
        for layer in self.hg_layers:
            layer.set_incidence(H)
    
    def forward(self, state, conf, age):
        """
        state: (batch, n_vars)
        conf: (batch, n_edges)
        age: (batch, 1)
        """
        batch_size = state.shape[0]
        
        # Embed state per variable
        state_reshaped = state.unsqueeze(-1)  # (batch, n_vars, 1)
        node_feats = self.state_embed(state_reshaped)  # (batch, n_vars, hidden_dim)
        
        # Add age embedding (broadcast to all nodes)
        age_embed = self.age_embed(age).unsqueeze(1)  # (batch, 1, hidden_dim)
        node_feats = node_feats + age_embed
        
        # Hypergraph layers
        for layer in self.hg_layers:
            node_feats = layer(node_feats, conf)
        
        # Readout
        pred = self.readout(node_feats).squeeze(-1)  # (batch, n_vars)
        return pred

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=50, lr=0.001, batch_size=64):
    """Train a model and evaluate."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            # For MLP, forward pass expects X_batch directly.
            # For HGNN, need to split features.
            if isinstance(model, MLP):
                pred = model(X_batch)
            else:
                # Split X_batch into state, conf, age
                state = X_batch[:, :8]
                conf = X_batch[:, 8:8+45]
                age = X_batch[:, -1:]
                pred = model(state, conf, age)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(device)
            if isinstance(model, MLP):
                pred_test = model(X_test_dev)
            else:
                state = X_test_dev[:, :8]
                conf = X_test_dev[:, 8:8+45]
                age = X_test_dev[:, -1:]
                pred_test = model(state, conf, age)
            test_loss = criterion(pred_test, Y_test.to(device)).item()
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss {avg_train_loss:.6f} | Test Loss {test_loss:.6f}")
    
    return train_losses, test_losses

def main():
    parser = argparse.ArgumentParser(description="Train HGNN or MLP for CA state prediction")
    parser.add_argument("--model", choices=["mlp", "hgnn"], default="mlp",
                        help="Model type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dataset", default="artifacts/ca_transitions.npz")
    parser.add_argument("--incidence", default="artifacts/hypergraph_incidence.npz")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    X_train, Y_train, X_test, Y_test, data_info = load_data(
        args.dataset, args.incidence, args.train_ratio)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {Y_test.shape[0]}")
    
    # Build model
    if args.model == "mlp":
        input_dim = X_train.shape[1]  # 54
        output_dim = Y_train.shape[1] # 8
        model = MLP(input_dim, output_dim, hidden_dims=[args.hidden_dim]*args.n_layers)
    else:
        H = data_info['H']
        model = HGNN(H, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    
    print(f"Model: {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train
    print("Training...")
    train_losses, test_losses = train_model(
        model, X_train, Y_train, X_test, Y_test,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    # Final evaluation (denormalized RMSE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_dev = X_test.to(device)
        if isinstance(model, MLP):
            pred = model(X_test_dev)
        else:
            state = X_test_dev[:, :8]
            conf = X_test_dev[:, 8:8+45]
            age = X_test_dev[:, -1:]
            pred = model(state, conf, age)
        pred_np = pred.cpu().numpy()
    
    # Denormalize predictions and targets
    state_mean = data_info['state_mean']
    state_std = data_info['state_std']
    pred_denorm = pred_np * state_std + state_mean
    Y_test_denorm = Y_test.numpy() * state_std + state_mean
    
    # RMSE per variable
    rmse_per_var = np.sqrt(np.mean((pred_denorm - Y_test_denorm)**2, axis=0))
    print("\nRMSE per variable:")
    for i, name in enumerate(data_info['variable_names']):
        print(f"  {name}: {rmse_per_var[i]:.4f}")
    
    # Overall RMSE
    overall_rmse = np.sqrt(np.mean((pred_denorm - Y_test_denorm)**2))
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    # Save model
    output_dir = Path("artifacts/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model}_ca_predictor.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()