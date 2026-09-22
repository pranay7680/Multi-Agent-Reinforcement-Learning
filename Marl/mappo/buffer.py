"""
buffer.py

Multi-Agent Rollout Buffer for MAPPO.

Stores trajectories in the form

    [time_step][agent]

instead of flattening everything.

Communication
-------------

received_messages / trust_weights are kept for backward-compatibility /
debugging, per mappo.py's comments -- they are NOT the differentiable
training source anymore.

communication_source_obs[t] stores the PREVIOUS timestep's full
multi-agent observation array -- i.e. the sender observations that were
actually used to generate the messages received at row t. This is what
mappo.py._reconstruct_received_messages() re-runs through the CURRENT
sender network during PPO update to keep the encoder/decoder in the
gradient graph.

communication_valid[t] is False for the first row of each episode
(no previous observation exists yet), True otherwise.
"""

import numpy as np
import torch

from .config import (
    NUM_AGENTS,
    OBS_DIM,
    ACTION_DIM,
    ROLLOUT_STEPS,
    DEVICE,
    GAMMA,
    GAE_LAMBDA,
)

from .gnn_attention import COMMUNICATION_DIM, NUM_HQ_SUBNETS, MAX_HOSTS


class MAPPOBuffer:

    # ==============================================================
    # Initialization
    # ==============================================================

    def __init__(self):

        self.clear()

    # ==============================================================
    # Clear
    # ==============================================================

    def clear(self):

        # ----------------------------------------------------------
        # Local observations
        #
        # shape = [T, N, OBS]
        # ----------------------------------------------------------

        self.obs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, OBS_DIM),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Global observations
        #
        # shape = [T, N * OBS]
        # ----------------------------------------------------------

        self.global_obs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS * OBS_DIM),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Actions
        # ----------------------------------------------------------

        self.actions = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.int64,
        )

        # ----------------------------------------------------------
        # Old log probabilities
        # ----------------------------------------------------------

        self.log_probs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Rewards
        # ----------------------------------------------------------

        self.rewards = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Critic values
        # ----------------------------------------------------------

        self.values = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Done flags
        # ----------------------------------------------------------

        self.dones = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Advantages / returns (filled after rollout)
        # ----------------------------------------------------------

        self.advantages = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        self.returns = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Action masks
        # ----------------------------------------------------------

        self.action_masks = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, ACTION_DIM),
            dtype=bool,
        )

        # ==========================================================
        # STRUCTURED COMMUNICATION -- legacy / debug storage
        #
        # received_messages[t, receiver, sender]
        #
        # Kept for backward compatibility. mappo.py.update() only
        # falls back to this when communication_source_obs is None.
        # ==========================================================

        self.received_messages = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, NUM_AGENTS, COMMUNICATION_DIM),
            dtype=np.float32,
        )

        # ==========================================================
        # TRUST
        #
        # trust_weights[t, receiver, sender]
        # ==========================================================

        self.trust_weights = np.ones(
            (ROLLOUT_STEPS, NUM_AGENTS, NUM_AGENTS),
            dtype=np.float32,
        )

        # ==========================================================
        # DIFFERENTIABLE COMMUNICATION SOURCE (new)
        #
        # communication_source_obs[t]:
        #
        #     [N, OBS]
        #
        # the PREVIOUS timestep's full multi-agent observation array --
        # i.e. the sender observations that produced the messages
        # actually received at row t. Re-run through the CURRENT
        # sender network at PPO-update time so decoder/encoder
        # parameters stay in the gradient graph.
        #
        # communication_valid[t]:
        #
        #     False for the first row of each episode (no previous
        #     observation exists yet), True otherwise.
        # ==========================================================

        self.communication_source_obs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, OBS_DIM),
            dtype=np.float32,
        )

        self.communication_valid = np.zeros(
            (ROLLOUT_STEPS,),
            dtype=bool,
        )

                # ==========================================================
        # V2 STRUCTURED COMMUNICATION
        #
        # Per sender:
        #   event_type
        #   target_type
        #   threat_level
        #   status
        #   priority
        #   confidence
        #   target_id
        #
        # Shape:
        #   [T, N, 7]
        # ==========================================================

        self.communication_field_ids = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, 7),
            dtype=np.int64,
        )

        self.communication_log_probs = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, 7),
            dtype=np.float32,
        )

        self.communication_entropies = np.zeros(
            (ROLLOUT_STEPS, NUM_AGENTS, 7),
            dtype=np.float32,
        )

        # ==========================================================
        # HOST-TARGET VALIDITY (communication_host_valid)
        #
        # communication_host_valid[t, sender] ([H] bool, STABLE host
        # order): which HOST ids that sender was allowed to sample at
        # row t (see env.get_host_valid_mask / decoder.
        # _masked_host_logits). Replayed at PPO-update time so
        # old/new target_id log-probs stay comparable. Lazily
        # allocated all-True (unmasked-equivalent) on first explicit
        # store -- mirroring host_active_mask's safe default -- so
        # rows stored without a mask (e.g. episode-start placeholders,
        # legacy callers) behave exactly like the unmasked legacy
        # path. Stays None until first supplied, in which case
        # get_batches() omits the key and update() trains unmasked.
        # ==========================================================

        self.communication_host_valid = None

        # ==========================================================
        # SUBNET-TARGET VALIDITY (communication_subnet_valid)
        #
        # communication_subnet_valid[t, sender] ([S] bool, STABLE subnet
        # order): which SUBNET ids that sender was allowed to sample at
        # row t (see env.get_subnet_valid_mask / decoder.
        # _masked_subnet_logits). Replayed at PPO-update time so
        # old/new target_id log-probs stay comparable. Lazily
        # allocated all-True (unmasked-equivalent) on first explicit
        # store -- mirroring communication_host_valid's safe default --
        # so rows stored without a mask behave exactly like the unmasked
        # legacy path. Stays None until first supplied, in which case
        # get_batches() omits the key and update() trains unmasked.
        # Unlike the host mask this is episode-static (the CC4 subnet
        # set is fixed), but it is still carried per-row so rollout and
        # replay always use the same mask for a given sender.
        # ==========================================================

        self.communication_subnet_valid = None

        # ==========================================================
        # HOST-LEVEL PADDING (host_active_mask)
        #
        # host_active_mask[t, agent, subnet_slot, host_slot]: True
        # iff a REAL host (this episode) occupies that position for
        # that agent -- see env.py.get_host_active_mask() /
        # gnn_attention.py's module docstring ("Host-level padding")
        # for the full explanation. Sourced from `state.hosts`
        # existence, NEVER from alert values (a real, currently-quiet
        # host and a nonexistent host both read 0 in the observation
        # vector, so alert-derived signals can't distinguish them).
        #
        # One mask per (timestep, agent), reused by mappo.py.update()
        # for both the receiver-side actor call AND the sender-side
        # differentiable communication reconstruction -- see that
        # file's own comment at the point it consumes this key. This
        # reuse is valid because CC4 fixes the real host count per
        # zone for the whole episode at reset time, so a single
        # timestep's mask already describes every other timestep in
        # the same episode.
        #
        # Initialised to all-True (not all-False): rows where the
        # caller supplies no mask must behave exactly like
        # gnn_attention.py's own default (host_active_mask=None --
        # every host slot in an *active* subnet is real; padding
        # subnet slots are still isolated via the observation-derived
        # slot activity). An all-False default would silently declare
        # every host padding. Explicitly supplied masks overwrite the
        # row unchanged (see store()).
        # ==========================================================

        self.host_active_mask = np.ones(
            (ROLLOUT_STEPS, NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS),
            dtype=bool,
        )

        # ----------------------------------------------------------
        # Pointer
        # ----------------------------------------------------------

        self.ptr = 0

    # ==============================================================
    # Store
    # ==============================================================

    def store(
        self,
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        values,
        dones,
        action_masks,
        received_messages=None,
        trust_weights=None,
        communication_source_obs=None,
        communication_valid=None,
        communication_field_ids=None,
        communication_log_probs=None,
        communication_entropies=None,
        host_active_mask=None,
        communication_host_valid=None,
        communication_subnet_valid=None,
    ):
        """
        Store one timestep of the multi-agent rollout.

        New optional arguments
        ----------------------

        communication_source_obs:

            [N, OBS]

            The previous timestep's full multi-agent observation array.
            Pass None (or omit) for the first step of an episode --
            communication_valid will be recorded as False for that row.

        communication_valid:

            bool. Whether communication_source_obs is meaningful for
            this row (False at episode start, before any previous
            observation exists).

        host_active_mask:

            [N, NUM_HQ_SUBNETS, MAX_HOSTS] bool, or None.

            This agent group's TRUE per-episode host-validity mask
            for this timestep -- see env.py.get_host_active_mask()/
            get_all_host_active_masks() and gnn_attention.py's module
            docstring ("Host-level padding"). Row i is agent i's own
            mask (matching `obs[i]`/`agent_names[i]` ordering
            everywhere else in this buffer). Pass None (or omit) to
            fall back to gnn_attention.py's own default -- every host
            slot in an active subnet treated as real, i.e. no
            per-host padding isolation.

        communication_host_valid:

            [N, H] bool, or None (H = STABLE host vocabulary size).

            Row i is agent i's HOST-target validity mask that its
            message at this row was sampled under -- see
            env.get_host_valid_mask(). Pass the PREVIOUS timestep's
            array (mirroring communication_field_ids), or None when
            unavailable: unstamped rows read all-True
            (unmasked-equivalent) and update() trains unmasked if the
            key was never supplied at all.

        communication_subnet_valid:

            [N, S] bool, or None (S = STABLE subnet vocabulary size).

            Row i is agent i's SUBNET-target validity mask that its
            message at this row was sampled under -- see
            env.get_subnet_valid_mask(). Same PREVIOUS-timestep
            mirroring as communication_host_valid.
        """

        if self.ptr >= ROLLOUT_STEPS:

            raise RuntimeError(
                "MAPPOBuffer is full. "
                "Call clear() before storing more transitions."
            )

        t = self.ptr

        # ==========================================================
        # Existing MAPPO data
        # ==========================================================

        self.obs[t] = obs
        self.global_obs[t] = global_obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.action_masks[t] = action_masks
        self.dones[t] = dones

        # ==========================================================
        # Legacy structured communication
        # ==========================================================

        if received_messages is not None:

            received_messages = np.asarray(received_messages, dtype=np.float32)

            expected_shape = (NUM_AGENTS, NUM_AGENTS, COMMUNICATION_DIM)

            if received_messages.shape != expected_shape:

                raise ValueError(
                    "Invalid received_messages shape. "
                    f"Expected {expected_shape}, got {received_messages.shape}"
                )

            self.received_messages[t] = received_messages

        # ==========================================================
        # Dynamic Trust
        # ==========================================================

        if trust_weights is not None:

            trust_weights = np.asarray(trust_weights, dtype=np.float32)

            expected_shape = (NUM_AGENTS, NUM_AGENTS)

            if trust_weights.shape != expected_shape:

                raise ValueError(
                    "Invalid trust_weights shape. "
                    f"Expected {expected_shape}, got {trust_weights.shape}"
                )

            trust_weights = np.clip(trust_weights, 0.0, 1.0)

            self.trust_weights[t] = trust_weights

        # ==========================================================
        # Differentiable communication source (new)
        # ==========================================================

        if communication_source_obs is not None:

            communication_source_obs = np.asarray(
                communication_source_obs, dtype=np.float32
            )

            expected_shape = (NUM_AGENTS, OBS_DIM)

            if communication_source_obs.shape != expected_shape:

                raise ValueError(
                    "Invalid communication_source_obs shape. "
                    f"Expected {expected_shape}, "
                    f"got {communication_source_obs.shape}"
                )

            self.communication_source_obs[t] = communication_source_obs

        self.communication_valid[t] = bool(communication_valid)

                # ==========================================================
        # V2 structured communication
        # ==========================================================

        if communication_field_ids is not None:

            communication_field_ids = np.asarray(
                communication_field_ids,
                dtype=np.int64,
            )

            expected_shape = (NUM_AGENTS, 7)

            if communication_field_ids.shape != expected_shape:
                raise ValueError(
                    "Invalid communication_field_ids shape. "
                    f"Expected {expected_shape}, "
                    f"got {communication_field_ids.shape}"
                )

            self.communication_field_ids[t] = (
                communication_field_ids
            )

        if communication_log_probs is not None:

            communication_log_probs = np.asarray(
                communication_log_probs,
                dtype=np.float32,
            )

            expected_shape = (NUM_AGENTS, 7)

            if communication_log_probs.shape != expected_shape:
                raise ValueError(
                    "Invalid communication_log_probs shape. "
                    f"Expected {expected_shape}, "
                    f"got {communication_log_probs.shape}"
                )

            self.communication_log_probs[t] = (
                communication_log_probs
            )

        if communication_entropies is not None:

            communication_entropies = np.asarray(
                communication_entropies,
                dtype=np.float32,
            )

            expected_shape = (NUM_AGENTS, 7)

            if communication_entropies.shape != expected_shape:
                raise ValueError(
                    "Invalid communication_entropies shape. "
                    f"Expected {expected_shape}, "
                    f"got {communication_entropies.shape}"
                )

            self.communication_entropies[t] = (
                communication_entropies
            )

        # ==========================================================
        # Host-level padding (host_active_mask)
        # ==========================================================

        if host_active_mask is not None:

            host_active_mask = np.asarray(
                host_active_mask, dtype=bool
            )

            expected_shape = (NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS)

            if host_active_mask.shape != expected_shape:
                raise ValueError(
                    "Invalid host_active_mask shape. "
                    f"Expected {expected_shape}, "
                    f"got {host_active_mask.shape}"
                )

            self.host_active_mask[t] = host_active_mask

        # ==========================================================
        # HOST-target validity (communication_host_valid)
        # ==========================================================

        if communication_host_valid is not None:

            communication_host_valid = np.asarray(
                communication_host_valid, dtype=bool
            )

            if communication_host_valid.ndim != 2:
                raise ValueError(
                    "Invalid communication_host_valid shape. "
                    "Expected [NUM_AGENTS, H], "
                    f"got {communication_host_valid.shape}"
                )

            if communication_host_valid.shape[0] != NUM_AGENTS:
                raise ValueError(
                    "Invalid communication_host_valid shape. "
                    f"Expected {NUM_AGENTS} sender rows, "
                    f"got {communication_host_valid.shape}"
                )

            if self.communication_host_valid is None:

                self.communication_host_valid = np.ones(
                    (ROLLOUT_STEPS, NUM_AGENTS)
                    + communication_host_valid.shape[1:],
                    dtype=bool,
                )

            elif (
                communication_host_valid.shape
                != self.communication_host_valid.shape[1:]
            ):

                raise ValueError(
                    "Invalid communication_host_valid shape. "
                    "Expected "
                    f"{self.communication_host_valid.shape[1:]}, "
                    f"got {communication_host_valid.shape}"
                )

            self.communication_host_valid[t] = communication_host_valid

        # ==========================================================
        # SUBNET-target validity (communication_subnet_valid)
        # ==========================================================

        if communication_subnet_valid is not None:

            communication_subnet_valid = np.asarray(
                communication_subnet_valid, dtype=bool
            )

            if communication_subnet_valid.ndim != 2:
                raise ValueError(
                    "Invalid communication_subnet_valid shape. "
                    "Expected [NUM_AGENTS, S], "
                    f"got {communication_subnet_valid.shape}"
                )

            if communication_subnet_valid.shape[0] != NUM_AGENTS:
                raise ValueError(
                    "Invalid communication_subnet_valid shape. "
                    f"Expected {NUM_AGENTS} sender rows, "
                    f"got {communication_subnet_valid.shape}"
                )

            if self.communication_subnet_valid is None:

                self.communication_subnet_valid = np.ones(
                    (ROLLOUT_STEPS, NUM_AGENTS)
                    + communication_subnet_valid.shape[1:],
                    dtype=bool,
                )

            elif (
                communication_subnet_valid.shape
                != self.communication_subnet_valid.shape[1:]
            ):

                raise ValueError(
                    "Invalid communication_subnet_valid shape. "
                    "Expected "
                    f"{self.communication_subnet_valid.shape[1:]}, "
                    f"got {communication_subnet_valid.shape}"
                )

            self.communication_subnet_valid[t] = communication_subnet_valid

        # ----------------------------------------------------------
        # Advance pointer
        # ----------------------------------------------------------

        self.ptr += 1

    # ==============================================================
    # Compute Advantages
    # ==============================================================

    def compute_advantages(self, last_values):

        last_values = np.asarray(last_values, dtype=np.float32)

        if last_values.shape != (NUM_AGENTS,):

            raise ValueError(
                f"last_values must have shape ({NUM_AGENTS},), "
                f"got {last_values.shape}"
            )

        gae = np.zeros(NUM_AGENTS, dtype=np.float32)

        for step in reversed(range(self.ptr)):

            if step == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + GAMMA * next_values * (1.0 - self.dones[step])
                - self.values[step]
            )

            gae = (
                delta
                + GAMMA * GAE_LAMBDA * (1.0 - self.dones[step]) * gae
            )

            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    # ==============================================================
    # Get Batches
    # ==============================================================

    def get_batches(self):

        if self.ptr == 0:

            raise RuntimeError(
                "MAPPOBuffer.get_batches() called on an empty buffer -- "
                "nothing has been stored since the last clear()."
            )

        n = self.ptr

        batch = {

            "obs":
                torch.tensor(self.obs[:n], dtype=torch.float32, device=DEVICE),

            "global_obs":
                torch.tensor(self.global_obs[:n], dtype=torch.float32, device=DEVICE),

            "actions":
                torch.tensor(self.actions[:n], dtype=torch.long, device=DEVICE),

            "log_probs":
                torch.tensor(self.log_probs[:n], dtype=torch.float32, device=DEVICE),

            "returns":
                torch.tensor(self.returns[:n], dtype=torch.float32, device=DEVICE),

            "advantages":
                torch.tensor(self.advantages[:n], dtype=torch.float32, device=DEVICE),

            "values":
                torch.tensor(self.values[:n], dtype=torch.float32, device=DEVICE),

            "action_masks":
                torch.tensor(self.action_masks[:n], dtype=torch.bool, device=DEVICE),

            "received_messages":
                torch.tensor(
                    self.received_messages[:n], dtype=torch.float32, device=DEVICE
                ),

            "trust_weights":
                torch.tensor(
                    self.trust_weights[:n], dtype=torch.float32, device=DEVICE
                ),

            # ======================================================
            # Differentiable communication source (new)
            # ======================================================

            "communication_source_obs":
                torch.tensor(
                    self.communication_source_obs[:n],
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "communication_valid":
                torch.tensor(
                    self.communication_valid[:n],
                    dtype=torch.bool,
                    device=DEVICE,
                ),

                            # ======================================================
            # V2 structured communication
            # ======================================================

            "communication_field_ids":
                torch.tensor(
                    self.communication_field_ids[:n],
                    dtype=torch.long,
                    device=DEVICE,
                ),

            "communication_log_probs":
                torch.tensor(
                    self.communication_log_probs[:n],
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "communication_entropies":
                torch.tensor(
                    self.communication_entropies[:n],
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            # ======================================================
            # Host-level padding (host_active_mask)
            # ======================================================

            "host_active_mask":
                torch.tensor(
                    self.host_active_mask[:n],
                    dtype=torch.bool,
                    device=DEVICE,
                ),
        }

        # Only present when at least one row was stored with an
        # explicit mask; update() treats a missing key as "train
        # unmasked" (legacy behavior).
        if self.communication_host_valid is not None:

            batch["communication_host_valid"] = torch.tensor(
                self.communication_host_valid[:n],
                dtype=torch.bool,
                device=DEVICE,
            )

        if self.communication_subnet_valid is not None:

            batch["communication_subnet_valid"] = torch.tensor(
                self.communication_subnet_valid[:n],
                dtype=torch.bool,
                device=DEVICE,
            )

        return batch

    # ==============================================================
    # Is Full / Length
    # ==============================================================

    def is_full(self):

        return self.ptr >= ROLLOUT_STEPS

    def __len__(self):

        return self.ptr