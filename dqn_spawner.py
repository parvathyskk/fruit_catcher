# dqn_spawner.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class SpawnerAgent:
    def __init__(self):
        self.state_dim = 4  
        """
        State is:
        [basket_x_norm, basket_y_norm, prev_spawn_x_norm, prev_type_norm]
        """

        self.action_dim = 500   # (5 fruits × 100 columns)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.lr = 1e-4

        self.buffer = deque(maxlen=80000)
        self.batch_size = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.q = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.update_freq = 2000
        self.update_count = 0

    # -----------------------------------------------------
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        qvals = self.q(s)
        return torch.argmax(qvals).item()

    # -----------------------------------------------------
    def remember(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    # -----------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        s2 = torch.FloatTensor(np.array(s2)).to(self.device)
        a = torch.LongTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(np.array(r)).to(self.device)
        d = torch.FloatTensor(np.array(d)).to(self.device)

        qvals = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        next_qvals = self.target(s2).max(1)[0]
        expected = r + (1 - d) * self.gamma * next_qvals

        loss = self.loss_fn(qvals, expected.detach())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Periodic target update
        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            self.update_target()

        return loss.item()

    # -----------------------------------------------------
    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    # -----------------------------------------------------
    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # -----------------------------------------------------
    def save(self, path="spawner_dqn.pth"):
        torch.save(self.q.state_dict(), path)

    def load(self, path="spawner_dqn.pth"):
        self.q.load_state_dict(torch.load(path))
        self.update_target()
