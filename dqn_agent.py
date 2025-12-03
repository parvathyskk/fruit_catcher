import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Optimized DQN agent for FruitCatcher.
    Usage:
      agent = DQNAgent(state_dim=6, action_dim=3)
      action = agent.act(state)
      agent.remember(state, action, reward, next_state, done)
      agent.train_step()   # call every frame; internal logic trains every `train_every` steps
      agent.decay()        # call every episode or every frame (keeps exploration)
      agent.update_target() # optional manual sync
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 64,
        buffer_size: int = 20000,
        batch_size: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        train_every: int = 4,
        target_update_interval: int = 200,
        device: str = None,
    ):
        # dims
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL hyperparams
        self.gamma = gamma

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.eps_min = epsilon_min
        self.eps_decay = epsilon_decay

        # replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # training frequency controls
        self.train_every = train_every
        self.target_update_interval = target_update_interval
        self.train_step_counter = 0     # counts gradient updates (used for target sync)
        self.frame_counter = 0          # counts frames / remember() calls

        # device auto detect
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.q = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        # optimizer & loss
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # -------------------------
    def act(self, state):
        """
        Epsilon-greedy action selection.
        `state` can be a numpy array shape (state_dim,) or a torch tensor.
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Exploitation
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)  # [1, state_dim]
        else:
            s = state.float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            qvals = self.q(s)                      # [1, action_dim]
            action = torch.argmax(qvals, dim=1).item()
        return action

    # -------------------------
    def remember(self, s, a, r, s2, done):
        """Store a single transition (s, a, r, s2, done)"""
        # store as raw (numpy) to minimize tensor conversions when not training
        self.buffer.append((np.array(s, dtype=np.float32),
                            int(a),
                            float(r),
                            np.array(s2, dtype=np.float32),
                            float(done)))
        self.frame_counter += 1

    # -------------------------
    def train_step(self):
        """
        Perform training step according to train_every.
        - If not enough samples: returns immediately.
        - Samples a random batch, computes Bellman target using target network,
          computes loss and applies gradient descent.
        - Periodically syncs target network.
        - Returns loss value if training happened, else None.
        """
        # only train every `train_every` frames
        if self.frame_counter < self.batch_size:
            return None

        if (self.frame_counter % self.train_every) != 0:
            return None

        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        s_batch, a_batch, r_batch, s2_batch, d_batch = zip(*batch)

        # to tensors
        s = torch.from_numpy(np.stack(s_batch)).to(self.device)           # [B, state_dim]
        s2 = torch.from_numpy(np.stack(s2_batch)).to(self.device)         # [B, state_dim]
        a = torch.tensor(a_batch, dtype=torch.long, device=self.device)   # [B]
        r = torch.tensor(r_batch, dtype=torch.float32, device=self.device) # [B]
        done = torch.tensor(d_batch, dtype=torch.float32, device=self.device) # [B]

        # Q(s,a) predicted by online network (select only the taken actions)
        q_all = self.q(s)                                 # [B, action_dim]
        q_pred = q_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

        # target using the target network
        with torch.no_grad():
            next_q_all = self.target(s2)                  # [B, action_dim]
            next_q_max = next_q_all.max(dim=1)[0]        # [B]
            q_target = r + (1.0 - done) * self.gamma * next_q_max  # [B]

        # loss and backward
        loss = self.loss_fn(q_pred, q_target)
        self.optim.zero_grad()
        loss.backward()
        # gradient clipping (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        # update counters and possibly sync target
        self.train_step_counter += 1
        if (self.train_step_counter % self.target_update_interval) == 0:
            self.update_target()
            
        return loss.item(), q_pred.mean().item()

    # -------------------------
    def update_target(self):
      
        self.target.load_state_dict(self.q.state_dict())

    # -------------------------
    def decay(self):
        
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # -------------------------
    # utility helpers
    def save(self, path):
        torch.save({
            "q_state": self.q.state_dict(),
            "target_state": self.target.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q_state"])
        self.target.load_state_dict(ckpt["target_state"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
