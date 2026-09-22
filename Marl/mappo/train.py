"""
train.py

Training script for MAPPO on CAGE Challenge 4 (CC4).

Communication
-------------

At timestep t:

    previous messages
            +
       trust weights
            |
            v
          MAPPO
            |
       +----+----+
       |         |
     action    message
       |         |
       v         v
     CybORG   communication
       |
       v
    rewards

The message generated at timestep t becomes available to the
other agents at timestep t+1. This one-step delay prevents an agent
from receiving information generated from the same observation it is
currently using to select its action.

Differentiable communication (new)
-----------------------------------

Alongside previous_messages (used for ACTING during rollout, under
no_grad -- unchanged), we now also track previous_obs_array: the raw
multi-agent observation array that PRODUCED those messages. This gets
stored per-row as communication_source_obs, so mappo.py.update() can
re-run it through the CURRENT sender network with gradients enabled and
keep the decoder/encoder in the training graph. communication_valid is
False for the first row of each episode, where no previous observation
exists yet.

Trust is updated only when a valid training-side ground-truth
representation is available (see env.py.get_ground_truth). Ground
truth is NEVER passed to the Blue agents.

Per-receiver trust
-------------------

See `get_receiver_relevance` and `evaluate_and_update_trust` below: a
message's factual correctness (event/target/threat/status) is graded
once per sender against the sender's own zone, since that's an
objective fact independent of who's listening. Only the resulting
message *quality* fed to trust is receiver-specific, via how
operationally relevant the sender's zone is to each particular
receiver -- this is what lets DynamicTrust's already-pairwise
alpha[sender, receiver]/beta[sender, receiver] storage actually diverge
across receivers instead of every receiver getting an identical score.
"""

import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import CC4Env
from .buffer import MAPPOBuffer
from .mappo import MAPPO
from .communication.evaluator import MessageEvaluator
from .action_mask import compute_padded_mask

from .gnn_attention import (
    MISSION_DIM,
    SUBNET_BLOCK_DIM,
    MESSAGE_DIM,
    NUM_HQ_SUBNETS,
    MAX_HOSTS,
)

from CybORG.Agents import (
    SleepAgent,  # not using bhai
    RandomSelectRedAgent,
    FiniteStateRedAgent,
)

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    EPISODE_LENGTH,
    TOTAL_EPISODES,
    ROLLOUT_STEPS,
    SEED,
    PRINT_EVERY,
    CURRICULUM_ENABLED,
    CURRICULUM_SCHEDULE,
    SAVE_EVERY,
    CHECKPOINT_DIR,
    LOG_DIR,
)


RED_AGENT_MAP = {
    "RandomSelectRedAgent": RandomSelectRedAgent,
    "FiniteStateRedAgent": FiniteStateRedAgent,
}


############################################################
# Receiver-relevance topology (for per-receiver trust)
############################################################
#
# A sender's message reports on the SENDER's own zone, so whether it
# is factually CORRECT is an objective fact about the world -- it
# does not depend on who is listening (see get_ground_truth_for_message
# below, which stays sender-only on purpose). How operationally USEFUL
# that same, objectively-true report is, however, genuinely differs by
# receiver: an alert about Operational Zone A matters far more to
# Restricted Zone A's defender (directly network-linked, same deployed
# network, per the CC4 topology) than it does to HQ or to Deployed
# Network B. This table is what MessageEvaluator.evaluate()'s
# `receiver_relevance` argument is built from -- it's the one place
# receiver identity is allowed to change the resulting quality score
# (see evaluator.py's module docstring for the full rationale).
#
# Agent -> zone assignment (fixed, per the CC4 scenario spec):
#   0: Restricted Zone A     1: Operational Zone A
#   2: Restricted Zone B     3: Operational Zone B
#   4: HQ (Admin / Office / Public Access)
_DIRECTLY_LINKED_AGENT_PAIRS = {
    (0, 1), (1, 0),   # Restricted A <-> Operational A
    (2, 3), (3, 2),   # Restricted B <-> Operational B
}

# Baseline relevance for any pair that isn't directly, operationally
# linked (still nonzero: a correct report elsewhere still has some
# general situational-awareness value to every receiver).
_DEFAULT_RECEIVER_RELEVANCE = 0.5


def get_receiver_relevance(sender_id: int, receiver_id: int) -> float:
    """
    How operationally relevant sender_id's zone is to receiver_id, in
    [0, 1]. This is the ONLY signal that varies message quality per
    receiver -- ground truth and objective correctness are computed
    once per sender and reused for every receiver (see
    evaluate_and_update_trust).
    """

    if (sender_id, receiver_id) in _DIRECTLY_LINKED_AGENT_PAIRS:
        return 1.0

    return _DEFAULT_RECEIVER_RELEVANCE


############################################################
# Seeding
############################################################

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################################
# Padding helpers
############################################################

def pad_observation(obs, real_dim):
    """
    Place a possibly-shorter agent observation into the fixed
    OBS_DIM buffer gnn_attention.py._split_entities() expects:

        [ mission | subnet_block_0 .. subnet_block_{NUM_HQ_SUBNETS-1} | messages ]

    Per BlueFlatWrapper._get_init_obs_spaces(), only the HQ agent
    (blue_agent_4) actually has NUM_HQ_SUBNETS subnet blocks in its
    raw observation -- pad_spaces defaults to False and EnterpriseMAE
    never overrides it, so every other agent's raw observation is:

        [ mission | subnet_block_0 | messages ]

    i.e. messages sit immediately after however many real subnet
    blocks that agent has (1, here), NOT after NUM_HQ_SUBNETS of
    them. A naive np.zeros(...) + front-fill puts those real
    messages where subnet_block_1 is expected, and puts zeros where
    messages are expected -- silently deleting communication content
    for every agent except the HQ one. This reassembles instead:
    real subnet blocks go into the first k slots (zero-filling the
    remaining NUM_HQ_SUBNETS - k), and messages always go into the
    fixed tail position regardless of k.
    """

    padded = np.zeros(OBS_DIM, dtype=np.float32)

    if real_dim == OBS_DIM:
        # Already the full [mission | NUM_HQ_SUBNETS blocks | messages]
        # layout (the HQ agent) -- nothing to reassemble.
        padded[:] = obs
        return padded

    num_real_blocks, remainder = divmod(
        real_dim - MISSION_DIM - MESSAGE_DIM,
        SUBNET_BLOCK_DIM,
    )

    if remainder != 0 or not (0 <= num_real_blocks <= NUM_HQ_SUBNETS):
        raise ValueError(
            f"Observation length {real_dim} does not decompose into "
            f"mission + k*SUBNET_BLOCK_DIM + messages for any integer "
            f"k in [0, {NUM_HQ_SUBNETS}] -- pad_observation's layout "
            f"assumption no longer matches the environment."
        )

    # Mission
    padded[:MISSION_DIM] = obs[:MISSION_DIM]

    # Real subnet blocks -> first num_real_blocks slots. The
    # remaining (NUM_HQ_SUBNETS - num_real_blocks) slots stay zero.
    real_blocks_end = MISSION_DIM + num_real_blocks * SUBNET_BLOCK_DIM
    padded[MISSION_DIM:real_blocks_end] = obs[MISSION_DIM:real_blocks_end]

    # Messages -> fixed tail position, wherever they actually sit in
    # the (shorter) source observation.
    tail_start = MISSION_DIM + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
    padded[tail_start:tail_start + MESSAGE_DIM] = obs[
        real_dim - MESSAGE_DIM:real_dim
    ]

    return padded

############################################################
# Episode boundary helper
############################################################

def episode_is_done(terminated, truncated):

    if "__all__" in terminated or "__all__" in truncated:

        return (
            terminated.get("__all__", False)
            or truncated.get("__all__", False)
        )

    return (
        all(terminated.values())
        or all(truncated.values())
    )


############################################################
# Curriculum
############################################################

def get_curriculum_stage(episode):
    """
    Progressive probabilistic curriculum.

    If CURRICULUM_ENABLED is False, the schedule is skipped entirely
    and every episode uses FiniteStateRedAgent (the "fully trained"
    end state of the schedule) -- previously this flag was defined in
    config.py but never checked here, so it had no effect either way.
    """

    if not CURRICULUM_ENABLED:
        return FiniteStateRedAgent

    probability_finite = 1.0

    for max_episode, p in CURRICULUM_SCHEDULE:

        if episode < max_episode:
            probability_finite = p
            break

    if random.random() < probability_finite:
        return FiniteStateRedAgent

    return RandomSelectRedAgent


############################################################
# Communication helpers
############################################################

def get_current_communication_for_agent(
    ppo,
    receiver_id,
    previous_messages,
):
    """
    Return the messages and trust values available to one agent
    before it selects its action. previous_messages are from the
    previous timestep, so there is no information leakage.
    """

    if previous_messages is None:
        return None, None

    received_messages = ppo.get_messages_for_agent(
        receiver_id=receiver_id,
        messages=previous_messages,
    )

    trust_weights = ppo.get_trust_for_agent(
        receiver_id=receiver_id,
    )

    return received_messages, trust_weights


############################################################
# Ground-truth hook
############################################################

def get_ground_truth_for_message(
    env,
    sender_id,
    receiver_id,
    message,
    previous_info,
    current_info,
):
    """
    Obtain pure, message-independent ground truth for the sender's zone.

    CC4Env.get_ground_truth(sender_id) returns what is ACTUALLY true in
    the sender's zone -- it does NOT see the message, the receiver, or the
    pre/post-step info dicts. Grading the message against that truth is
    the MessageEvaluator's job (see communication/evaluator.py), which is
    what keeps a wrong-target claim from being silently rescued.

    receiver_id is intentionally unused: correctness of a claim about the
    SENDER's zone is an objective fact and does not depend on who is
    receiving it. (Receiver-specific behavior belongs in *usefulness*,
    computed separately per receiver in evaluate_and_update_trust via
    get_receiver_relevance -- not here.) The receiver_id / message /
    previous_info / current_info parameters are retained only for
    call-site compatibility and are intentionally unused here.

    Returns None if the environment cannot produce ground truth (e.g. the
    sender id cannot be resolved) -- this deliberately prevents training
    trust on fabricated labels.
    """

    if hasattr(env, "get_ground_truth"):

        return env.get_ground_truth(sender_id)

    return None


############################################################
# Evaluate outgoing messages and update trust
############################################################

def evaluate_and_update_trust(
    ppo,
    evaluator,
    env,
    outgoing_structured_messages,
    previous_info,
    current_info,
):
    """
    Evaluate messages after the environment step and update trust.

    outgoing_structured_messages: list of StructuredMessage, index = sender.

    Trust convention: trust(sender, receiver) means how much receiver
    trusts sender. Ground truth and objective correctness (event /
    target / threat / status) describe the SENDER's own zone and are
    therefore computed ONCE per sender -- they do not depend on who is
    receiving the message. Each receiver still gets its own trust
    update, but what varies per receiver is *usefulness*: how
    operationally relevant the sender's zone is to that specific
    receiver (see get_receiver_relevance). That's what lets
    DynamicTrust's pairwise alpha[sender, receiver]/beta[sender,
    receiver] storage actually diverge across receivers, rather than
    every receiver silently getting an identical quality score. If
    ground truth is unavailable, no trust update is performed.
    """

    if (
        outgoing_structured_messages is None
        or len(outgoing_structured_messages) != NUM_AGENTS
    ):
        return

    for sender_id in range(NUM_AGENTS):

        message = outgoing_structured_messages[sender_id]

        if message is None:
            continue

        if hasattr(message, "is_empty") and message.is_empty():
            continue

        # Objective ground truth for the sender's own zone -- computed
        # once per sender/message, NOT once per receiver, since
        # correctness of a claim about the sender's zone does not
        # depend on who's listening. (Do not swap this to
        # get_ground_truth(receiver_id): the message says nothing
        # about the receiver's zone, so there would be nothing
        # meaningful to check it against.)
        ground_truth = get_ground_truth_for_message(
            env=env,
            sender_id=sender_id,
            receiver_id=None,
            message=message,
            previous_info=previous_info,
            current_info=current_info,
        )

        if ground_truth is None:
            continue

        for receiver_id in range(NUM_AGENTS):

            if sender_id == receiver_id:
                continue

            # The one receiver-specific input: how relevant the
            # sender's (objectively correct-or-not) zone report is to
            # THIS receiver.
            receiver_relevance = get_receiver_relevance(
                sender_id,
                receiver_id,
            )

            evaluation = evaluator.evaluate(
                message=message,
                ground_truth=ground_truth,
                previous_state=previous_info,
                current_state=current_info,
                receiver_relevance=receiver_relevance,
            )

            quality = evaluator.confidence_adjusted_score(evaluation)

            ppo.update_trust(
                sender=sender_id,
                receiver=receiver_id,
                message_quality=quality,
            )
            # print(
            #     f"[TRUST] "
            #     f"{sender_id}->{receiver_id} "
            #     f"quality={quality:.3f} "
            #     f"trust={ppo.get_trust(sender_id, receiver_id):.3f}"
            # )


############################################################
# Main training loop
############################################################

def train():

    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ########################################################
    # Environment
    ########################################################

    current_red_agent = get_curriculum_stage(0)

    env = CC4Env(red_agent_class=current_red_agent, seed=SEED)

    agent_names = sorted(env.possible_agents)

    assert len(agent_names) == NUM_AGENTS, (
        f"Expected {NUM_AGENTS} blue agents, "
        f"found {len(agent_names)}: {agent_names}"
    )

    obs_dims = env.get_observation_dims()

    # Adaptive Action Mask replaces the old static per-episode mask
    # (see action_mask.py). Masks are now computed fresh every
    # timestep inside the rollout loop below via compute_padded_mask().

    ########################################################
    # Initial reset
    ########################################################

    

    obs_dict, info = env.reset(seed=SEED)

########################################################
# Communication target vocabulary
########################################################

    num_host_targets = env.get_num_targets()
    num_subnet_targets = env.get_num_subnet_targets()

    print(
        f"[MAPPO] Communication target count: "
        f"host={num_host_targets} subnet={num_subnet_targets}"
    )

    ########################################################
    # Agent / Buffer
    ########################################################

    ppo = MAPPO(
        num_host_targets=num_host_targets,
        num_subnet_targets=num_subnet_targets,
    )

    buffer = MAPPOBuffer()
    evaluator = MessageEvaluator()

    ########################################################
    # Runtime communication state
    ########################################################

    previous_messages = None
    previous_obs_array = None

    previous_field_ids = None
    previous_comm_log_probs = None
    previous_comm_entropies = None
    previous_comm_host_valid = None
    previous_comm_subnet_valid = None

    episode_return = np.zeros(NUM_AGENTS, dtype=np.float32)
    episode_returns_log = []
    episode_count = 0
    update_count = 0

    ########################################################
    # Training History
    ########################################################

    episode_return_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []

    previous_info = info

    total_timesteps = TOTAL_EPISODES * EPISODE_LENGTH

    start_time = time.time()

    ########################################################
    # PPO update helper (single code path)
    #
    # Used both for full rollouts and for curriculum-flush updates
    # (see the episode-boundary block): identical GAE bootstrap ->
    # PPO update -> stats -> clear sequence in both cases, so update
    # timing/counting/logging cannot drift between the two paths.
    ########################################################

    def run_ppo_update(last_values, reason=""):
        nonlocal update_count

        buffer.compute_advantages(last_values)

        ppo.train()
        stats = ppo.update(buffer)

        actor_loss_history.append(stats["actor_loss"])
        critic_loss_history.append(stats["critic_loss"])
        entropy_history.append(stats["entropy"])

        ppo.eval()

        update_count += 1

        print(
            f"  -> update {update_count:5d}  "
            f"actor_loss={stats['actor_loss']:.4f}  "
            f"critic_loss={stats['critic_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}"
            f"{reason}"
        )

        buffer.clear()

    ########################################################
    # Rollout / Update Loop
    ########################################################

    for t in range(1, total_timesteps + 1):

        ####################################################
        # Build padded local + global observations
        ####################################################

        obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

        for i, name in enumerate(agent_names):
            obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

        global_obs = obs_array.reshape(-1)

        ####################################################
        # True per-episode host layout (host_active_mask)
        #
        # CC4 fixes the real host count per zone at reset, but that
        # count is not recoverable from obs_array itself (see
        # env.py.get_host_active_mask()'s docstring) -- so it is read
        # directly from the environment's true state here, once per
        # timestep, for every agent, in the same agent_names order
        # obs_array/actions_arr/etc. already use. This does NOT
        # change within an episode; it is only recomputed each step
        # because reset(seed=...) regenerates the layout at every
        # episode boundary below.
        ####################################################

        host_active_mask_array = env.get_all_host_active_masks()

        assert host_active_mask_array.shape == (
            NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS
        ), (
            "env.get_all_host_active_masks() returned "
            f"{host_active_mask_array.shape}, expected "
            f"{(NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS)}."
        )

        # Per-sender HOST-target validity in STABLE host order, used
        # to mask per-episode-invalid ids BEFORE the decoder samples
        # (see decoder._masked_host_logits). Same agent_names order
        # as obs_array; constant within an episode.
        comm_host_valid_array = env.get_all_host_valid_masks()

        assert comm_host_valid_array.shape == (
            NUM_AGENTS, num_host_targets
        ), (
            "env.get_all_host_valid_masks() returned "
            f"{comm_host_valid_array.shape}, expected "
            f"{(NUM_AGENTS, num_host_targets)}."
        )

        # Per-sender SUBNET-target validity in STABLE subnet order, used
        # to mask unobservable subnet ids BEFORE the decoder samples
        # (see decoder._masked_subnet_logits and
        # env.get_subnet_valid_mask). Episode-static (the CC4 subnet set
        # is fixed), but carried per-row like the host mask so rollout
        # and PPO replay stay comparable.
        comm_subnet_valid_array = env.get_all_subnet_valid_masks()

        assert comm_subnet_valid_array.shape == (
            NUM_AGENTS, num_subnet_targets
        ), (
            "env.get_all_subnet_valid_masks() returned "
            f"{comm_subnet_valid_array.shape}, expected "
            f"{(NUM_AGENTS, num_subnet_targets)}."
        )

        ####################################################
        # Messages available BEFORE acting
        ####################################################

        current_received_messages = (
            np.zeros(
                (NUM_AGENTS, NUM_AGENTS, ppo.communication.message_dim),
                dtype=np.float32,
            )
            if previous_messages is not None
            else None
        )

        current_trust_weights = (
            np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=np.float32)
            if previous_messages is not None
            else None
        )

        ####################################################
        # Act
        ####################################################

        actions_dict = {}
        actions_arr = np.zeros(NUM_AGENTS, dtype=np.int64)
        log_probs_arr = np.zeros(NUM_AGENTS, dtype=np.float32)
        values_arr = np.zeros(NUM_AGENTS
        , dtype=np.float32)
        masks_arr = np.zeros((NUM_AGENTS, ACTION_DIM), dtype=bool)

        for i, name in enumerate(agent_names):

            mask = compute_padded_mask(env, name)
            masks_arr[i] = mask

            ################################################
            # Communication available to this receiver
            ################################################

            received_messages, trust_weights = get_current_communication_for_agent(
                ppo=ppo,
                receiver_id=i,
                previous_messages=previous_messages,
            )

            ################################################
            # Save communication state for the buffer
            ################################################

            if received_messages is not None:

                current_received_messages[i] = (
                    received_messages.detach().cpu().numpy()
                )
                current_trust_weights[i] = (
                    trust_weights.detach().cpu().numpy()
                )

            ################################################
            # Select action
            ################################################

            action, log_prob, value, entropy = ppo.select_action(
                observation=obs_array[i],
                action_mask=mask,
                global_state=global_obs,
                agent_id=i,
                received_messages=received_messages,
                trust_weights=trust_weights,
                host_active_mask=host_active_mask_array[i],
            )

            actions_dict[name] = action
            actions_arr[i] = action
            log_probs_arr[i] = log_prob.item()
            values_arr[i] = 0.0 if value is None else value.item()

        ####################################################
        # Generate outgoing structured messages
        #
        # Generated from the CURRENT observation, but not delivered
        # until t+1. Still no_grad -- this is behavior, not training.
        ####################################################

        (
            outgoing_vectors,
            outgoing_structured_messages,
            communication_field_ids,
            communication_log_probs,
            communication_entropies,
        ) = ppo.get_outgoing_messages(
            obs_array,
            return_decoded=True,
            host_active_mask=host_active_mask_array,
            host_valid_mask=comm_host_valid_array,
            subnet_valid_mask=comm_subnet_valid_array,
        )

        ####################################################
        # Step environment
        ####################################################

        (
            next_obs_dict,
            rewards_dict,
            terminated,
            truncated,
            info,
        ) = env.step(actions_dict)

        rewards_arr = np.array(
            [rewards_dict[name] for name in agent_names],
            dtype=np.float32,
        )

        done_flag = episode_is_done(terminated, truncated)

        dones_arr = np.full(NUM_AGENTS, float(done_flag), dtype=np.float32)

        ####################################################
        # Evaluate message quality / update trust
        ####################################################

        evaluate_and_update_trust(
            ppo=ppo,
            evaluator=evaluator,
            env=env,
            outgoing_structured_messages=outgoing_structured_messages,
            previous_info=previous_info,
            current_info=info,
        )

        ####################################################
        # Store transition
        #
        # communication_source_obs/communication_valid are built from
        # previous_obs_array -- the obs that produced the messages
        # actually received BEFORE this step's action, mirroring
        # current_received_messages exactly.
        ####################################################

        communication_source = (
            previous_obs_array
            if previous_obs_array is not None
            else np.zeros(
                (NUM_AGENTS, OBS_DIM),
                dtype=np.float32,
            )
        )

        stored_field_ids = (
            previous_field_ids
            if previous_field_ids is not None
            else np.zeros(
                (NUM_AGENTS, 7),
                dtype=np.int64,
            )
        )

        stored_comm_log_probs = (
            previous_comm_log_probs
            if previous_comm_log_probs is not None
            else np.zeros(
                (NUM_AGENTS, 7),
                dtype=np.float32,
            )
        )

        stored_comm_entropies = (
            previous_comm_entropies
            if previous_comm_entropies is not None
            else np.zeros(
                (NUM_AGENTS, 7),
                dtype=np.float32,
            )
        )

        # Validity mask of the PREVIOUS obs (that generated the stored
        # ids), mirroring stored_field_ids exactly. None on the first
        # row of an episode -- that row reads all-True
        # (unmasked-equivalent) and is excluded from the actor loss by
        # communication_valid=False anyway.
        stored_comm_host_valid = (
            previous_comm_host_valid
            if previous_comm_host_valid is not None
            else None
        )

        # Same PREVIOUS-timestep mirroring for the SUBNET mask.
        stored_comm_subnet_valid = (
            previous_comm_subnet_valid
            if previous_comm_subnet_valid is not None
            else None
        )

        buffer.store(
            obs=obs_array,
            global_obs=global_obs,
            actions=actions_arr,
            log_probs=log_probs_arr,
            rewards=rewards_arr,
            values=values_arr,
            dones=dones_arr,
            action_masks=masks_arr,
            received_messages=current_received_messages,
            trust_weights=current_trust_weights,

            communication_source_obs=communication_source,
            communication_valid=(
                previous_obs_array is not None
            ),

            communication_field_ids=stored_field_ids,
            communication_log_probs=stored_comm_log_probs,
            communication_entropies=stored_comm_entropies,
            communication_host_valid=stored_comm_host_valid,
            communication_subnet_valid=stored_comm_subnet_valid,

            host_active_mask=host_active_mask_array,
        )

        ####################################################
        # Newly generated messages/obs become available at t+1
        ####################################################

        # previous_messages = outgoing_vectors.detach()
        # previous_obs_array = obs_array
        previous_messages = outgoing_vectors.detach()
        previous_obs_array = obs_array

        previous_field_ids = communication_field_ids
        previous_comm_log_probs = communication_log_probs
        previous_comm_entropies = communication_entropies
        previous_comm_host_valid = comm_host_valid_array
        previous_comm_subnet_valid = comm_subnet_valid_array
        ####################################################
        # Update evaluator state
        ####################################################

        previous_info = info
        episode_return += rewards_arr
        obs_dict = next_obs_dict

        ####################################################
        # Episode boundary
        ####################################################

        if done_flag:

            episode_count += 1

            # Reuse the same CC4Env across episodes: reset(seed=...)
            # already regenerates the scenario (new host layout) via the
            # reseeded RNG, so recreating the whole wrapper every episode
            # only wastes construction cost and breaks determinism. A new
            # instance is required solely when the curriculum switches
            # the red-agent class, which lives in the scenario generator.
            next_red_agent = get_curriculum_stage(episode_count)
            if next_red_agent is not current_red_agent:
                # Curriculum switch: the buffer holds ONLY
                # old-curriculum transitions (the row just stored ends
                # with dones=1, so GAE already terminates there -- no
                # credit propagates past it). Run the PPO update on
                # that old-only data with a terminal (zero) bootstrap,
                # then clear BEFORE recreating the environment, so the
                # next rollout contains new-curriculum transitions
                # exclusively and no A->B temporal transition exists.
                if len(buffer) > 0:
                    run_ppo_update(
                        np.zeros(NUM_AGENTS, dtype=np.float32),
                        reason="  [curriculum flush]",
                    )
                current_red_agent = next_red_agent
                env = CC4Env(
                    red_agent_class=current_red_agent,
                    seed=SEED + episode_count,
                )
                agent_names = sorted(env.possible_agents)
                obs_dims = env.get_observation_dims()

            team_return = episode_return.sum()
            episode_returns_log.append(team_return)
            episode_return_history.append(team_return)

            if episode_count % PRINT_EVERY == 0:

                recent = episode_returns_log[-PRINT_EVERY:]
                mean_return = float(np.mean(recent))
                elapsed = time.time() - start_time

                print(
                    f"[episode {episode_count:6d}] "
                    f"team_return={mean_return:8.2f}  "
                    f"updates={update_count:5d}  "
                    f"elapsed={elapsed:7.1f}s  "
                    f"RedAgent={current_red_agent.__name__}"
                )

            if episode_count % SAVE_EVERY == 0:

                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"mappo_ep{episode_count}.pt"
                )
                ppo.save(ckpt_path)

            episode_return[:] = 0.0

            # ------------------------------------------------
            # Reset communication memory -- no message and no
            # source observation carries over from the previous
            # episode.
            # ------------------------------------------------

            previous_messages = None
            previous_obs_array = None
            previous_field_ids = None
            previous_comm_log_probs = None
            previous_comm_entropies = None
            previous_comm_host_valid = None
            previous_comm_subnet_valid = None

            obs_dict, info = env.reset(seed=SEED + episode_count)
            previous_info = info

        ####################################################
        # PPO Update
        ####################################################

        if buffer.is_full():

            last_obs_array = np.zeros((NUM_AGENTS, OBS_DIM), dtype=np.float32)

            for i, name in enumerate(agent_names):
                last_obs_array[i] = pad_observation(obs_dict[name], obs_dims[name])

            last_global_obs = last_obs_array.reshape(-1)

            last_values = ppo.get_value(last_global_obs).cpu().numpy()

            run_ppo_update(last_values)

    ########################################################
    # Final checkpoint
    ########################################################

    final_path = os.path.join(CHECKPOINT_DIR, "mappo_final.pt")
    ppo.save(final_path)

    print(f"Training complete. Final checkpoint: {final_path}")

    ########################################################
    # Plots
    ########################################################

    plt.figure(figsize=(10, 5))
    plt.plot(episode_return_history)
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, "episode_return.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(actor_loss_history)
    plt.title("Actor Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, "actor_loss.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(critic_loss_history)
    plt.title("Critic Loss")
    plt.xlabel("PPO Update")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, "critic_loss.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(entropy_history)
    plt.title("Entropy")
    plt.xlabel("PPO Update")
    plt.ylabel("Entropy")
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR, "entropy.png"))

if __name__ == "__main__":
    train()