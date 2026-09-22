"""
networks.py

Neural networks for MAPPO (LEGACY -- DEPRECATED / NOT ACTIVE).

mappo.py imports MAPPOModel from gnn_attention.py, NOT from this file.
Kept only for reference/history; do not import from training code.

Architecture
------------
Shared Actor:
    Observation -> Action Probabilities

Central Critic:
    Global Observation -> State Value

Both are simple MLPs and can later be replaced with
Graph Neural Networks or Transformers.
"""

import torch
import torch.nn as nn

from .config import (
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_AGENTS,
)


# ==========================================================
# Utility
# ==========================================================

def build_mlp(input_dim, output_dim):
    """
    Build a simple feed-forward MLP.
    """

    layers = []

    current = input_dim

    for _ in range(NUM_HIDDEN_LAYERS):

        layers.append(nn.Linear(current, HIDDEN_DIM))
        layers.append(nn.ReLU())

        current = HIDDEN_DIM

    layers.append(nn.Linear(current, output_dim))

    return nn.Sequential(*layers)


# ==========================================================
# Shared Actor
# ==========================================================

class SharedActor(nn.Module):
    """
    One policy shared by ALL blue agents.

    Input:
        Local observation (210 dims)

    Output:
        Action logits (242 dims)
    """

    def __init__(self):

        super().__init__()

        self.policy = build_mlp(
            OBS_DIM,
            ACTION_DIM,
        )

    def forward(self, observation):

        logits = self.policy(observation)

        return logits


# ==========================================================
# Central Critic
# ==========================================================

class CentralCritic(nn.Module):
    """
    Centralized critic.

    Receives the GLOBAL STATE.

    Current implementation:

        concat(
            obs0,
            obs1,
            obs2,
            obs3,
            obs4
        )

    Later we can replace this with
    graph embeddings or attention.
    """

    def __init__(self):

        super().__init__()

        # global_dim = OBS_DIM * 5
        global_dim = OBS_DIM * NUM_AGENTS

        self.value = build_mlp(
            global_dim,
            NUM_AGENTS,
        )

    def forward(self, global_state):

        return self.value(global_state)


# ==========================================================
# MAPPO Network
# ==========================================================

class MAPPOModel(nn.Module):
    """
    Holds both actor and critic.

    This makes saving/loading checkpoints easier.
    """

    def __init__(self):

        super().__init__()

        self.actor = SharedActor()

        self.critic = CentralCritic()


    def act(self, observation):

        """
        Returns action logits.

        PPO will convert these into a categorical
        distribution.
        """

        return self.actor(observation)


    def evaluate(self, global_state):

        """
        Returns critic value estimate.
        """

        return self.critic(global_state)