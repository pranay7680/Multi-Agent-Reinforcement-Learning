"""
evaluate.py

Evaluation for the current CC4 MAPPO implementation.

Compatible with:

    - GNN + Attention actor
    - V2 structured communication
    - 7-field communication
    - Dynamic Trust
    - padded observations
    - action masking
    - centralized critic
    - ValueNorm
    - checkpointed trust state
    - one-step communication delay

Evaluation is completely frozen:

    NO gradients
    NO PPO updates
    NO rollout buffer
    NO trust updates
    NO MessageEvaluator
    NO ground-truth feedback to the policy

The evaluation path mirrors the training-time rollout:

    observation
        +
    messages from t-1
        +
    trust
        |
        v
      MAPPO
        |
        +---- action
        |
        +---- message for t+1
        |
        v
      CybORG


Usage
-----

Single checkpoint, both Red agents:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100

FiniteStateRedAgent only:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100 \
        --red-agent finite

RandomSelectRedAgent only:

    python -m Marl.mappo.evaluate \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --episodes 100 \
        --red-agent random

Checkpoint sweep:

    python -m Marl.mappo.evaluate \
        --sweep checkpoints/attention_test_curriculum \
        --episodes 50

By default evaluation is greedy.

Use:

    --stochastic

to sample actions from the policy instead.
"""

import argparse
import glob
import os

import numpy as np
import torch
from .env import CC4Env
from .mappo import MAPPO

from .train import (
    pad_observation,
    episode_is_done,
)
from .action_mask import compute_padded_mask

from .gnn_attention import NUM_HQ_SUBNETS, MAX_HOSTS

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
)

from CybORG.Agents import (
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)


# ==========================================================
# Red agents
# ==========================================================

RED_AGENTS = {
    "random": RandomSelectRedAgent,
    "finite": FiniteStateRedAgent,
}


# ==========================================================
# Checkpoint architecture helper
# ==========================================================

def get_checkpoint_num_targets(checkpoint_path):
    """
    Read the communication target vocabulary sizes directly from the
    saved MAPPO checkpoint.

    The checkpoint is authoritative for model construction because the
    decoder's host_target_head/subnet_target_head and the encoder's
    host/subnet target embeddings were created with the target counts
    that existed when the model was trained.

    HOST and SUBNET are separate, independently-sized vocabularies
    (see mappo.py's module docstring and communication/decoder.py) --
    there is no longer a single combined "num_targets". MAPPO.save()
    persists both directly on the checkpoint dict as
    "num_host_targets" / "num_subnet_targets" (see mappo.py.save()),
    so this reads those keys rather than inferring a shape from a
    single target_head weight, which no longer exists as one tensor.

    Current MAPPO checkpoints are saved as:

        {
            "model": model.state_dict(),
            "num_host_targets": int,
            "num_subnet_targets": int or None,
            ...
        }

    Returns
    -------
    (int or None, int or None)
        (num_host_targets, num_subnet_targets) stored in the
        checkpoint. Both are None for older checkpoints saved before
        this field existed -- the caller can fall back to the current
        environment's counts in that case.

    Why this is needed
    ------------------
    CC4Env.get_num_targets()/get_num_subnet_targets() can differ
    depending on the environment instance/scenario used during
    evaluation. Constructing MAPPO from the current environment's
    counts can therefore produce, for example:

        checkpoint: 94 host targets
        evaluation env: 95 host targets

    which causes a state_dict size mismatch in:

        actor.communication.decoder.host_target_head
        actor.communication.decoder.subnet_target_head
        actor.communication.encoder.host_target_embedding
        actor.communication.encoder.subnet_target_embedding
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. Expected a dictionary "
            f"but got {type(checkpoint).__name__}."
        )

    if "model" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain a 'model' state_dict. "
            "Cannot determine the communication target vocabulary."
        )

    num_host_targets = checkpoint.get("num_host_targets")
    num_subnet_targets = checkpoint.get("num_subnet_targets")

    if num_host_targets is not None and num_host_targets <= 0:
        raise RuntimeError(
            f"Invalid checkpoint num_host_targets: {num_host_targets}"
        )

    if num_subnet_targets is not None and num_subnet_targets <= 0:
        raise RuntimeError(
            "Invalid checkpoint num_subnet_targets: "
            f"{num_subnet_targets}"
        )

    return num_host_targets, num_subnet_targets


# ==========================================================
# Communication state helper
# ==========================================================

def get_current_communication_for_agent(
    ppo,
    receiver_id,
    previous_messages,
):
    """
    Return exactly the communication state available to one
    receiver before it chooses its action.

    This mirrors train.py.

    previous_messages are generated at timestep t-1,
    therefore they are the only messages available at timestep t.

    Returns
    -------

    received_messages:
        [NUM_AGENTS, COMMUNICATION_DIM]

    trust_weights:
        [NUM_AGENTS]

    or:

        None, None

    at the first timestep.
    """

    if previous_messages is None:
        return None, None

    received_messages = (
        ppo.get_messages_for_agent(
            receiver_id=receiver_id,
            messages=previous_messages,
        )
    )

    trust_weights = (
        ppo.get_trust_for_agent(
            receiver_id=receiver_id,
        )
    )

    return (
        received_messages,
        trust_weights,
    )


# ==========================================================
# Greedy action
# ==========================================================

@torch.no_grad()
def select_action_greedy(
    ppo,
    observation,
    action_mask,
    received_messages,
    trust_weights,
    host_active_mask=None,
):
    """
    Greedy evaluation.

    Uses the same actor path as training:

        observation
             +
        received communication
             +
        trust
             +
        host_active_mask
             |
             v
          actor
             |
             v
        action mask
             |
             v
        argmax action

    host_active_mask: this agent's TRUE per-episode host-validity
    mask, [NUM_HQ_SUBNETS, MAX_HOSTS] bool -- see
    env.py.get_host_active_mask()/get_all_host_active_masks() and
    gnn_attention.py's module docstring ("Host-level padding"). Unlike
    select_action()/get_outgoing_message(s), actor_forward() does not
    batch-prepare this itself (its docstring says the caller is
    expected to already have batch-aligned data), so it is prepared
    here with the same `_prepare_host_active_mask` helper
    MAPPO.select_action() uses internally, against the same
    batch_size actor_forward()'s own observation batching will use.
    """

    observation = ppo._to_tensor(
        observation,
        dtype=torch.float32,
    )

    action_mask = ppo._to_tensor(
        action_mask,
        dtype=torch.bool,
    )

    batch_size = (
        1 if observation.dim() == 1 else observation.shape[0]
    )

    prepared_host_active_mask = ppo._prepare_host_active_mask(
        host_active_mask,
        batch_size,
    )

    logits = ppo.actor_forward(
        observations=observation,
        action_masks=action_mask,
        received_messages=received_messages,
        trust_weights=trust_weights,
        host_active_mask=prepared_host_active_mask,
    )

    action = torch.argmax(
        logits,
        dim=-1,
    )

    return int(action.item())


# ==========================================================
# Stochastic action
# ==========================================================

@torch.no_grad()
def select_action_stochastic(
    ppo,
    observation,
    action_mask,
    global_state,
    agent_id,
    received_messages,
    trust_weights,
    host_active_mask=None,
):
    """
    Stochastic evaluation.

    This uses MAPPO's normal select_action() path.

    host_active_mask: this agent's TRUE per-episode host-validity
    mask, [NUM_HQ_SUBNETS, MAX_HOSTS] bool -- passed straight through;
    ppo.select_action() already batch-prepares it internally (see
    MAPPO._prepare_host_active_mask), matching train.py's usage
    exactly.
    """

    action, _, _, _ = ppo.select_action(
        observation=observation,
        action_mask=action_mask,
        global_state=global_state,
        agent_id=agent_id,
        received_messages=received_messages,
        trust_weights=trust_weights,
        host_active_mask=host_active_mask,
    )

    return int(action)


# ==========================================================
# Run episodes
# ==========================================================

def run_episodes(
    ppo,
    red_agent_class,
    num_episodes,
    deterministic=True,
    base_seed=100_000,
):
    """
    Evaluate a frozen MAPPO policy against one Red agent.

    Important:

        Trust is NOT updated.

        Ground truth is NOT queried.

        The checkpoint's trust matrix remains frozen.

    Communication timing exactly matches training:

        timestep t:
            receive message generated at t-1
            choose action
            generate message

        timestep t+1:
            receive that message
    """

    # ------------------------------------------------------
    # Create initial environment
    # ------------------------------------------------------

    env = CC4Env(
        red_agent_class=red_agent_class,
    )

    agent_names = sorted(
        env.possible_agents
    )

    if len(agent_names) != NUM_AGENTS:
        raise RuntimeError(
            "Unexpected number of Blue agents: "
            f"expected {NUM_AGENTS}, "
            f"got {len(agent_names)}"
        )

    obs_dims = (
        env.get_observation_dims()
    )

    action_dims = (
        env.get_action_dims()
    )

    # Adaptive Action Mask replaces the old static per-episode mask.
    # Masks are computed fresh every timestep inside the episode loop
    # below via compute_padded_mask() -- see action_mask.py.

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    episode_returns = []
    episode_lengths = []

    # ------------------------------------------------------
    # Episodes
    # ------------------------------------------------------

    for ep in range(num_episodes):

        # Reuse one CC4Env: reset(seed=...) already regenerates the
        # scenario (new host layout) via the reseeded RNG. Recreating
        # the wrapper per episode only wastes construction cost.
        obs_dict, info = env.reset(
            seed=base_seed + ep
        )

        # --------------------------------------------------
        # Communication generated at t-1.
        #
        # None at timestep 0.
        # --------------------------------------------------

        previous_messages = None

        # --------------------------------------------------
        # Episode state
        # --------------------------------------------------

        episode_return = np.zeros(
            NUM_AGENTS,
            dtype=np.float32,
        )

        done = False
        timestep = 0

        # --------------------------------------------------
        # Episode loop
        # --------------------------------------------------

        while (
            not done
            and timestep < EPISODE_LENGTH
        ):

            # ==================================================
            # Pad observations
            # ==================================================

            obs_array = np.zeros(
                (
                    NUM_AGENTS,
                    OBS_DIM,
                ),
                dtype=np.float32,
            )

            for i, name in enumerate(
                agent_names
            ):

                obs_array[i] = (
                    pad_observation(
                        obs_dict[name],
                        obs_dims[name],
                    )
                )

            # ==================================================
            # True per-episode host layout (host_active_mask)
            #
            # Same source, ordering, and reuse convention as
            # train.py: read directly from the environment's true
            # state (never inferred from alert values -- a real,
            # currently-quiet host and a nonexistent host are
            # bit-for-bit identical in the observation vector), once
            # per timestep, for every agent, in the same
            # agent_names order obs_array already uses. Constant for
            # the whole episode; only changes across the fresh
            # CC4Env created at each episode boundary above.
            # ==================================================

            host_active_masks = (
                env.get_all_host_active_masks()
            )

            assert host_active_masks.shape == (
                NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS
            ), (
                "env.get_all_host_active_masks() returned "
                f"{host_active_masks.shape}, expected "
                f"{(NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS)}."
            )

            # Per-sender HOST-target validity in STABLE host order
            # (same mask train.py samples under -- evaluation must use
            # the identical mask or it would emit ids the trained
            # policy never produces). Constant for the whole episode.
            comm_host_valid_masks = (
                env.get_all_host_valid_masks()
            )

            assert comm_host_valid_masks.shape == (
                NUM_AGENTS, ppo.num_host_targets
            ), (
                "env.get_all_host_valid_masks() returned "
                f"{comm_host_valid_masks.shape}, expected "
                f"{(NUM_AGENTS, ppo.num_host_targets)}."
            )

            # Per-sender SUBNET-target validity in STABLE subnet order
            # (same mask train.py samples under). Episode-static. None
            # when the model has no subnet head (legacy
            # num_subnet_targets=None) -- then sampling/evaluation run
            # unmasked, exactly as before.
            if ppo.num_subnet_targets is None:
                comm_subnet_valid_masks = None
            else:
                comm_subnet_valid_masks = (
                    env.get_all_subnet_valid_masks()
                )

                assert comm_subnet_valid_masks.shape == (
                    NUM_AGENTS, ppo.num_subnet_targets
                ), (
                    "env.get_all_subnet_valid_masks() returned "
                    f"{comm_subnet_valid_masks.shape}, expected "
                    f"{(NUM_AGENTS, ppo.num_subnet_targets)}."
                )

            # ==================================================
            # Centralized critic state
            # ==================================================

            global_obs = (
                obs_array.reshape(-1)
            )

            # ==================================================
            # Actions
            # ==================================================

            actions_dict = {}

            for agent_id, name in enumerate(
                agent_names
            ):

                # --------------------------------------------------
                # Communication available BEFORE this action
                # --------------------------------------------------

                (
                    received_messages,
                    trust_weights,
                ) = get_current_communication_for_agent(
                    ppo=ppo,
                    receiver_id=agent_id,
                    previous_messages=previous_messages,
                )

                # --------------------------------------------------
                # Agent-specific action mask
                # --------------------------------------------------

                mask = compute_padded_mask(env, name)

                # --------------------------------------------------
                # Choose action
                # --------------------------------------------------

                if deterministic:

                    action = (
                        select_action_greedy(
                            ppo=ppo,
                            observation=obs_array[agent_id],
                            action_mask=mask,
                            received_messages=received_messages,
                            trust_weights=trust_weights,
                            host_active_mask=host_active_masks[
                                agent_id
                            ],
                        )
                    )

                else:

                    action = (
                        select_action_stochastic(
                            ppo=ppo,
                            observation=obs_array[agent_id],
                            action_mask=mask,
                            global_state=global_obs,
                            agent_id=agent_id,
                            received_messages=received_messages,
                            trust_weights=trust_weights,
                            host_active_mask=host_active_masks[
                                agent_id
                            ],
                        )
                    )

                actions_dict[name] = action

            # ==================================================
            # Generate outgoing communication
            #
            # IMPORTANT:
            #
            # This is generated from the CURRENT observation.
            #
            # It is NOT supplied to the agents' current actions.
            #
            # It becomes available at timestep t+1.
            # ==================================================

            (
                outgoing_vectors,
                _communication_field_ids,
                _communication_log_probs,
                _communication_entropies,
            ) = ppo.get_outgoing_messages(
                obs_array,
                return_decoded=False,
                host_active_mask=host_active_masks,
                host_valid_mask=comm_host_valid_masks,
                subnet_valid_mask=comm_subnet_valid_masks,
            )

            # --------------------------------------------------
            # Safety conversion
            #
            # MAPPO.get_outgoing_messages() returns:
            #
            #     messages
            #     field_ids
            #     log_probs
            #     entropies
            #
            # Evaluation only needs the communication vectors.
            # The other three outputs are used during PPO training
            # and are intentionally ignored here.
            # --------------------------------------------------

            if not isinstance(
                outgoing_vectors,
                torch.Tensor,
            ):

                outgoing_vectors = (
                    torch.as_tensor(
                        outgoing_vectors,
                        dtype=torch.float32,
                        device=ppo.device,
                    )
                )
            else:

                outgoing_vectors = (
                    outgoing_vectors.to(
                        device=ppo.device,
                        dtype=torch.float32,
                    )
                )

            if outgoing_vectors.ndim != 2:
                raise RuntimeError(
                    "MAPPO returned communication vectors with "
                    "unexpected shape: "
                    f"{tuple(outgoing_vectors.shape)}. "
                    "Expected [NUM_AGENTS, COMMUNICATION_DIM]."
                )

            if outgoing_vectors.shape[0] != NUM_AGENTS:
                raise RuntimeError(
                    "MAPPO returned the wrong number of "
                    f"communication vectors: expected {NUM_AGENTS}, "
                    f"got {outgoing_vectors.shape[0]}."
                )

            previous_messages = (
                outgoing_vectors.detach()
            )

            # ==================================================
            # Environment step
            # ==================================================

            (
                next_obs_dict,
                rewards_dict,
                terminated,
                truncated,
                info,
            ) = env.step(
                actions_dict
            )

            # ==================================================
            # Rewards
            # ==================================================

            rewards_arr = np.array(
                [
                    rewards_dict[name]
                    for name in agent_names
                ],
                dtype=np.float32,
            )

            episode_return += (
                rewards_arr
            )

            # ==================================================
            # Done
            # ==================================================

            done = episode_is_done(
                terminated,
                truncated,
            )

            obs_dict = next_obs_dict

            timestep += 1

        # ------------------------------------------------------
        # Episode statistics
        # ------------------------------------------------------

        episode_returns.append(
            float(
                episode_return.sum()
            )
        )

        episode_lengths.append(
            timestep
        )

    return (
        np.asarray(
            episode_returns,
            dtype=np.float32,
        ),
        np.asarray(
            episode_lengths,
            dtype=np.int32,
        ),
    )


# ==========================================================
# Load and evaluate one checkpoint
# ==========================================================

def evaluate_checkpoint(
    checkpoint_path,
    red_agent_names,
    num_episodes,
    deterministic,
):
    """
    Evaluate one checkpoint independently against each Red agent.

    The checkpoint contains:

        model
        actor optimizer
        critic optimizer
        ValueNorm state
        trust state

    MAPPO.load() restores these.

    Trust is then frozen for the entire evaluation.
    """

    print()
    print("=" * 72)
    print(
        f"Checkpoint: "
        f"{os.path.basename(checkpoint_path)}"
    )
    print("=" * 72)

    # ------------------------------------------------------
    # Determine communication target vocabulary.
    #
    # IMPORTANT:
    # The checkpoint is authoritative here.
    #
    # The model's decoder/encoder were built with the target
    # vocabulary that existed during training. We must construct
    # MAPPO with that SAME number before calling ppo.load().
    #
    # Previously this was taken from a fresh evaluation environment,
    # which produced 95 while the checkpoint contained 94.
    # ------------------------------------------------------

    checkpoint_num_host_targets, checkpoint_num_subnet_targets = (
        get_checkpoint_num_targets(
            checkpoint_path
        )
    )

    # Fall back only for old checkpoints that do not contain these
    # fields at all.
    if checkpoint_num_host_targets is None:

        reference_env = CC4Env(
            red_agent_class=RED_AGENTS[
                red_agent_names[0]
            ],
        )

        num_host_targets = (
            reference_env.get_num_targets()
        )

        print(
            "[MAPPO] Checkpoint has no num_host_targets; "
            "using current environment host target count."
        )

    else:

        num_host_targets = checkpoint_num_host_targets

    if checkpoint_num_subnet_targets is None:

        reference_env = CC4Env(
            red_agent_class=RED_AGENTS[
                red_agent_names[0]
            ],
        )

        num_subnet_targets = (
            reference_env.get_num_subnet_targets()
        )

        print(
            "[MAPPO] Checkpoint has no num_subnet_targets; "
            "using current environment subnet target count."
        )

    else:

        num_subnet_targets = checkpoint_num_subnet_targets

    print(
        f"[MAPPO] Checkpoint communication target count: "
        f"host={num_host_targets} subnet={num_subnet_targets}"
    )

    # ------------------------------------------------------
    # Diagnostic: compare against each evaluation environment.
    #
    # This does NOT change the model architecture. It only tells us
    # whether the current CC4 scenario has the same host vocabulary
    # size as the one used to train the checkpoint.
    # ------------------------------------------------------

    for red_name in red_agent_names:

        env_target_count = (
            CC4Env(
                red_agent_class=RED_AGENTS[
                    red_name
                ],
            ).get_num_targets()
        )

        if env_target_count != num_host_targets:

            print(
                f"[WARNING] {red_name} environment reports "
                f"{env_target_count} host targets, but the checkpoint "
                f"was trained with {num_host_targets}."
            )

            print(
                "          The checkpoint will load correctly, "
                "but target-ID semantics must match the training "
                "host ordering for communication evaluation."
            )

        else:

            print(
                f"[MAPPO] {red_name} environment target count: "
                f"{env_target_count} (matches checkpoint)"
            )

    # ------------------------------------------------------
    # Construct the SAME MAPPO architecture used by the checkpoint
    # ------------------------------------------------------

    ppo = MAPPO(
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )

    # ------------------------------------------------------
    # Load trained checkpoint
    #
    # This restores:
    #
    #   model
    #   optimizers
    #   ValueNorm
    #   trust state
    # ------------------------------------------------------

    ppo.load(
        checkpoint_path
    )

    # ------------------------------------------------------
    # Frozen evaluation mode
    # ------------------------------------------------------

    ppo.eval()

    results = {}

    # ------------------------------------------------------
    # Evaluate independently against each Red agent
    # ------------------------------------------------------

    for red_name in red_agent_names:

        red_agent_class = (
            RED_AGENTS[red_name]
        )

        print()
        print(
            f"RedAgent: "
            f"{red_agent_class.__name__}"
        )

        returns, lengths = (
            run_episodes(
                ppo=ppo,
                red_agent_class=red_agent_class,
                num_episodes=num_episodes,
                deterministic=deterministic,
            )
        )

        results[red_name] = {
            "mean": float(
                returns.mean()
            ),
            "std": float(
                returns.std()
            ),
            "median": float(
                np.median(returns)
            ),
            "min": float(
                returns.min()
            ),
            "max": float(
                returns.max()
            ),
            "mean_length": float(
                lengths.mean()
            ),
            "n": int(
                num_episodes
            ),
        }

        print(
            f"  mean      = "
            f"{results[red_name]['mean']:9.2f}"
        )

        print(
            f"  std       = "
            f"{results[red_name]['std']:9.2f}"
        )

        print(
            f"  median    = "
            f"{results[red_name]['median']:9.2f}"
        )

        print(
            f"  min       = "
            f"{results[red_name]['min']:9.2f}"
        )

        print(
            f"  max       = "
            f"{results[red_name]['max']:9.2f}"
        )

        print(
            f"  avg length= "
            f"{results[red_name]['mean_length']:9.2f}"
        )

        print(
            f"  episodes  = "
            f"{results[red_name]['n']}"
        )

    # ------------------------------------------------------
    # Combined mean
    # ------------------------------------------------------

    if (
        "random" in results
        and "finite" in results
    ):

        combined_mean = (
            results["random"]["mean"]
            + results["finite"]["mean"]
        ) / 2.0

        print()
        print(
            f"Combined mean = "
            f"{combined_mean:9.2f}"
        )

    return results


# ==========================================================
# Checkpoint discovery
# ==========================================================

def get_checkpoint_list(
    directory,
):
    """
    Return:

        mappo_ep*.pt

    in numeric episode order.
    """

    checkpoints = glob.glob(
        os.path.join(
            directory,
            "mappo_ep*.pt",
        )
    )

    def checkpoint_number(path):

        filename = os.path.basename(
            path
        )

        digits = "".join(
            ch
            for ch in filename
            if ch.isdigit()
        )

        if not digits:
            return -1

        return int(digits)

    return sorted(
        checkpoints,
        key=checkpoint_number,
    )


# ==========================================================
# CLI
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen CC4 MAPPO checkpoint."
        )
    )

    # ------------------------------------------------------
    # Single checkpoint
    # ------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a single .pt checkpoint."
        ),
    )

    # ------------------------------------------------------
    # Checkpoint sweep
    # ------------------------------------------------------

    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help=(
            "Directory containing mappo_ep*.pt checkpoints."
        ),
    )

    # ------------------------------------------------------
    # Episodes
    # ------------------------------------------------------

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help=(
            "Number of evaluation episodes per Red agent."
        ),
    )

    # ------------------------------------------------------
    # Red agent
    # ------------------------------------------------------

    parser.add_argument(
        "--red-agent",
        type=str,
        choices=[
            "random",
            "finite",
            "both",
        ],
        default="both",
        help=(
            "Red opponent to evaluate against."
        ),
    )

    # ------------------------------------------------------
    # Stochastic evaluation
    # ------------------------------------------------------

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Sample actions instead of using greedy argmax."
        ),
    )

    args = parser.parse_args()

    # ======================================================
    # Validate checkpoint arguments
    # ======================================================

    if (
        args.checkpoint is None
        and args.sweep is None
    ):

        parser.error(
            "Provide either --checkpoint or --sweep."
        )

    if (
        args.checkpoint is not None
        and args.sweep is not None
    ):

        parser.error(
            "Use either --checkpoint or --sweep, "
            "not both."
        )

    # ======================================================
    # Validate episode count
    # ======================================================

    if args.episodes <= 0:

        parser.error(
            "--episodes must be greater than zero."
        )

    # ======================================================
    # Red agent selection
    # ======================================================

    if args.red_agent == "both":

        red_agent_names = [
            "random",
            "finite",
        ]

    else:

        red_agent_names = [
            args.red_agent
        ]

    # ======================================================
    # Evaluation mode
    # ======================================================

    deterministic = (
        not args.stochastic
    )

    # ======================================================
    # Build checkpoint list
    # ======================================================

    if args.checkpoint is not None:

        if not os.path.isfile(
            args.checkpoint
        ):

            parser.error(
                "Checkpoint does not exist: "
                f"{args.checkpoint}"
            )

        checkpoints = [
            args.checkpoint
        ]

    else:

        checkpoints = (
            get_checkpoint_list(
                args.sweep
            )
        )

        if not checkpoints:

            parser.error(
                "No mappo_ep*.pt checkpoints found in: "
                f"{args.sweep}"
            )

    # ======================================================
    # Header
    # ======================================================

    print()
    print("=" * 72)
    print("CC4 MAPPO EVALUATION")
    print("=" * 72)

    print(
        "Architecture : GNN + Attention"
    )

    print(
        "Communication: V2 structured"
    )

    print(
        "Trust        : checkpoint state, frozen"
    )

    print(
        "PPO updates  : disabled"
    )

    print(
        "Ground truth : disabled"
    )

    print(
        "Mode         : "
        + (
            "greedy"
            if deterministic
            else "stochastic"
        )
    )

    print(
        "Episodes     : "
        f"{args.episodes} per Red agent"
    )
    print("=" * 72)

    for checkpoint in checkpoints:

        evaluate_checkpoint(
            checkpoint_path=checkpoint,
            red_agent_names=red_agent_names,
            num_episodes=args.episodes,
            deterministic=deterministic,
        )
if __name__ == "__main__":
    main()