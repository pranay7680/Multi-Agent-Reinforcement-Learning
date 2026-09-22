"""
config.py

Configuration file for MAPPO training on CAGE Challenge 4 (CC4).
Modify hyperparameters here instead of changing them throughout the code.
"""

# ==========================================================
# Environment
# ==========================================================

NUM_AGENTS = 5

EPISODE_LENGTH = 100
# Must match EnterpriseScenarioGenerator(steps=100)

MISSION_PHASES = 3


# ==========================================================
# Observation / Action Dimensions
# ==========================================================

# Blue Agents 0-3
SMALL_OBS_DIM = 92
SMALL_ACTION_DIM = 82

# Blue Agent 4
LARGE_OBS_DIM = 210
LARGE_ACTION_DIM = 242

# Shared-policy dimensions
#
# We pad every observation to 210 features
# and every action distribution to 242 actions.
#
# Invalid actions are masked before sampling.

OBS_DIM = LARGE_OBS_DIM
ACTION_DIM = LARGE_ACTION_DIM


# ==========================================================
# MAPPO Hyperparameters
# ==========================================================
TOTAL_EPISODES = 1000                             

ROLLOUT_STEPS = 512

UPDATE_EPOCHS = 5
# Was 10

MINIBATCH_SIZE = 256

# General learning rate
# Kept because mappo.py imports this value.
LEARNING_RATE = 3e-4

# Separate learning rates
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 5e-5

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
# Separate communication coefficients.
#
# The joint message log-prob sums 7 fields, so its scale/variance is
# larger than the single env action. Without its own coefficient the
# comm policy gradient can dominate the action gradient.
# Likewise comm entropy (mean over senders) must not share the action
# entropy bonus 1:1.
COMM_POLICY_COEF = 0.3
COMM_ENTROPY_COEF = 0.005
MAX_GRAD_NORM = 0.5

# Optional value clipping
VALUE_CLIP = True
# Was False

# Adaptive Action Mask (contextual gating on top of CybORG's structural
# mask -- see action_mask.py). Single source of truth for AAM enable/
# disable. True (default) preserves current behavior: Restore/Remove only
# on flagged hosts, BlockTrafficZone only from flagged zones. False
# completely disables AAM and uses only the structural mask (CybORG
# episode-static validity + Sleep safety net), for the MAPPO +
# structural-mask-only ablation. The heuristic gate changes the action
# space from observation -> action into observation -> heuristic ->
# allowed actions -> policy, so results with it enabled encode domain
# assumptions.
USE_AAM = True

# Standard MAPPO practice
NORMALIZE_ADVANTAGES = True


# ==========================================================
# Neural Network
# ==========================================================

HIDDEN_DIM = 256

NUM_HIDDEN_LAYERS = 5
# Change back to 2 later

ACTIVATION = "relu"

# Resolve to CUDA only when actually available; otherwise fall back to
# CPU instead of crashing inside torch .to("cuda").
import torch as _torch

DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"


# ==========================================================
# Logging
# ==========================================================

PRINT_EVERY = 10

SAVE_EVERY = 250

CHECKPOINT_DIR = "checkpoints/fixedMaybe"
LOG_DIR = "evaluation/fixedMaybe"


# ==========================================================
# Randomness
# ==========================================================

SEED = 42




# ==========================================================
# Curriculum Learning
# ==========================================================

USE_VALUE_NORM = True

CURRICULUM_ENABLED = True

CURRICULUM_STAGES = [
    (0, "RandomSelectRedAgent"),
    (4000, "FiniteStateRedAgent"),
]

CURRICULUM_SWITCH_EPISODE = 200

CURRICULUM_SCHEDULE = [
    (100, 0.10),   # 90% Random, 10% Finite
    (300, 0.40),   # 60% Random, 40% Finite
    (700, 0.80),   # 20% Random, 80% Finite
    (1000, 1.00),   # 100% Finite
    (10000, 1.00),  # 100% Finite
]


# ==========================================================
# Attention
# ==========================================================

EMBED_DIM = 256

NUM_HEADS = 4