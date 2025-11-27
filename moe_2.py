# moe_rlhf.py
# Minimal Mixture-of-Experts + simple RLHF (PPO-like) prototype using PyTorch
# Save as moe_rlhf.py and run with: python moe_rlhf.py
# Requirements: torch, tqdm

import math
import random
import copy
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm

# ---------------------------
# Toy dataset
# ---------------------------
class ToyRegressionDataset(Dataset):
    # input: vector of dim D, target: scalar in R
    def __init__(self, n_samples=2000, dim=32):
        super().__init__()
        self.X = torch.randn(n_samples, dim)
        # target is a deterministic function of X (for supervised pretrain)
        self.y = (self.X.sum(dim=1) + 0.1 * torch.randn(n_samples)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ---------------------------
# MoE model
# ---------------------------
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, n_experts):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_experts)

    def forward(self, x):
        logits = self.linear(x)  # [B, n_experts]
        gate_prob = torch.softmax(logits, dim=-1)
        return gate_prob

class MoE(nn.Module):
    def __init__(self, input_dim, n_experts=4, expert_hidden=128, expert_out=1):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden, expert_out) for _ in range(n_experts)])
        self.gate = GatingNetwork(input_dim, n_experts)

    def forward(self, x):
        # x: [B, D]
        gate_prob = self.gate(x)  # [B, n]
        expert_outputs = []
        for e in self.experts:
            expert_outputs.append(e(x).unsqueeze(-1))  # [B, out_dim, 1]
        # Stack expert outputs: [B, out_dim, n_experts]
        expert_stack = torch.cat(expert_outputs, dim=-1)
        # Weighted sum across experts using gate_prob
        # gate_prob: [B, n_experts] -> [B, 1, n_experts]
        gate_exp = gate_prob.unsqueeze(1)
        out = (expert_stack * gate_exp).sum(dim=-1)  # [B, out_dim]
        return out, gate_prob

# ---------------------------
# Reward model (learned, simulates human preference)
# ---------------------------
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),  # input plus model's scalar output
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, model_pred):
        # x: [B, D], model_pred: [B, 1]
        inp = torch.cat([x, model_pred], dim=1)
        return self.net(inp)  # scalar reward

# ---------------------------
# Utilities and training routines
# ---------------------------
def pretrain_supervised(model: MoE, dataloader, epochs=5, lr=1e-3, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"Pretrain ep {ep+1}/{epochs} mse: {total/len(dataloader.dataset):.6f}")

def train_reward_model(reward_model: RewardModel, model: MoE, dataloader, epochs=3, lr=1e-3, device='cpu'):
    reward_model = reward_model.to(device)
    opt = optim.Adam(reward_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    reward_model.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in dataloader:
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                pred, _ = model(x)
            # construct pseudo "human" reward: negative abs error to target (higher reward when close)
            # Use this to produce training targets for reward model
            human_reward = -torch.abs(pred.detach() - y)
            r_pred = reward_model(x, pred.detach())
            loss = loss_fn(r_pred, human_reward)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"Reward model ep {ep+1}/{epochs} mse: {total/len(dataloader.dataset):.6f}")

# ---------------------------
# Simple PPO-like update for MoE parameters
# - This example updates all parameters (gate + experts).
# - Uses clipped surrogate objective on predicted reward.
# - Works with continuous scalar outputs.
# ---------------------------
def ppo_train_step(model: MoE, reward_model: RewardModel, batch_x, old_log_probs, old_preds,
                   optimizer, clip_eps=0.2, vf_coef=0.0, entropy_coef=0.0, device='cpu'):
    # batch_x: [B, D]
    model.train()
    reward_model.eval()  # reward model frozen during policy update here
    x = batch_x.to(device)
    pred, gate_prob = model(x)  # pred: [B, 1], gate_prob: [B, n]
    # For continuous outputs, treat "policy" as deterministic mapping + possibly gaussian noise.
    # To keep simple: derive pseudo-log-prob from gate distribution (categorical) as policy component.
    # Compute current log-prob under gating (sum log prob of selected expert mixture expectation).
    # But gating is soft; approximate "action distribution" as categorical with probs=gate_prob and "action" being expected expert index.
    # For prototype, compute surrogate using gate probabilities:
    # old_log_probs: previously recorded log-probs per sample from gate (tensor)
    totd = gate_prob  # [B, n]
    curr_log_probs = torch.log(torch.clamp((totd.sum(dim=1) / totd.size(1)), 1e-9))  # scalar proxy per sample

    # Get reward estimate from reward model
    with torch.no_grad():
        reward_vals = reward_model(x, pred).squeeze(1)  # [B]
    # Advantage: use reward directly minus baseline (use mean as baseline)
    baseline = reward_vals.mean()
    advantages = (reward_vals - baseline).detach()

    # surrogate objective
    ratio = torch.exp(curr_log_probs - old_log_probs.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # optional value loss (not used here)
    value_loss = torch.tensor(0.0, device=device)

    # entropy regularization (encourage spread in gate)
    entropy = - (gate_prob * torch.log(torch.clamp(gate_prob, 1e-9))).sum(dim=1).mean()

    loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "entropy": entropy.item(),
        "avg_reward": reward_vals.mean().item()
    }

def collect_old_log_probs(model: MoE, batch_x, device='cpu'):
    # For prototype: compute a scalar log-prob proxy per sample from gate distribution
    with torch.no_grad():
        _, gate_prob = model(batch_x.to(device))
        # proxy
        logp = torch.log(torch.clamp((gate_prob.sum(dim=1) / gate_prob.size(1)), 1e-9))
    return logp.detach()

# ---------------------------
# Full RLHF loop (toy)
# ---------------------------
def rlhf_loop(model: MoE, reward_model: RewardModel, dataset: ToyRegressionDataset,
              device='cpu', pretrain_epochs=5, ppo_epochs=10, ppo_batch_size=64):
    # Pretrain supervised
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    pretrain_supervised(model, dataloader, epochs=pretrain_epochs, device=device)

    # Train reward model to imitate human judgments (synthetic here)
    train_reward_model(reward_model, model, dataloader, epochs=3, device=device)

    # PPO-like fine-tuning loop
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(ppo_epochs):
        # shuffle and batch
        loader = DataLoader(dataset, batch_size=ppo_batch_size, shuffle=True)
        logs = {"policy_loss": 0.0, "entropy": 0.0, "avg_reward": 0.0}
        count = 0
        for x, _ in loader:
            old_logp = collect_old_log_probs(model, x, device=device)  # old behavior
            stats = ppo_train_step(model, reward_model, x, old_logp, None, optimizer, device=device)
            for k,v in stats.items():
                logs[k] += v
            count += 1
        print(f"PPO epoch {ep+1}/{ppo_epochs} -- policy_loss {logs['policy_loss']/count:.6f}, entropy {logs['entropy']/count:.6f}, avg_reward {logs['avg_reward']/count:.6f}")

# ---------------------------
# Example usage
# ---------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = 32
    n_experts = 6
    dataset = ToyRegressionDataset(n_samples=2000, dim=D)
    model = MoE(input_dim=D, n_experts=n_experts, expert_hidden=128, expert_out=1)
    reward_model = RewardModel(input_dim=D, hidden_dim=128)
    rlhf_loop(model, reward_model, dataset, device=device, pretrain_epochs=4, ppo_epochs=6, ppo_batch_size=128)

    # Quick evaluation
    loader = DataLoader(dataset, batch_size=256)
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for x,y in loader:
            pred, gate = model(x.to(device))
            total_mse += ((pred.cpu()-y)**2).sum().item()
    print("Final MSE:", total_mse / len(dataset))

if __name__ == "__main__":
    main()
