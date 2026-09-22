"""
mappo.py

Multi-Agent PPO (MAPPO) implementation for CC4.

Contains
--------
- Shared Actor
- Centralized Critic
- Action Selection
- Value Prediction
- Structured Communication
- Dynamic Trust
- Save / Load
- PPO Update

Structured communication training
----------------------------------

Rollout communication is generated under no_grad(), which is correct.

The rollout buffer stores:

    communication_source_obs
    communication_field_ids

During PPO optimization, TWO SEPARATE mechanisms train the
communication pipeline -- they are easy to conflate, so this section
is explicit about which is which.

(a) A genuinely differentiable path, for the RECEIVER side only:

        communication_field_ids (fixed, discrete, from rollout)
                |
                v
          encoder.encode_from_ids(...)
                |
                v
        communication vector
                |
                v
        receiver attention -> policy_input_projection -> PPO loss

    `communication_field_ids` are already-sampled integers pulled
    straight from the buffer -- they are not differentiable tensors,
    so gradients from the receiver's action/value loss flow into the
    `encoder`'s parameters (and everything downstream: receiver
    attention, policy head, critic), but they CANNOT flow past the
    field-id boundary into whatever produced those ids (the `decoder`
    or the sender's `actor`). That upstream computation isn't even
    part of this forward graph -- the ids were sampled by a different,
    earlier (and by update time, stale) copy of the model.

(b) A score-function (REINFORCE/PPO-style) path, for the SENDER side:

        source observation
                |
                v
          sender actor (CURRENT weights)
                |
                v
          decoder.evaluate_message(...) -> log P(stored field_ids)
                |
                v
        communication_ratio = exp(new_log_prob - old_log_prob)
                |
                v
        communication_actor_loss (clipped surrogate, using the
        RECEIVER's advantage as the reward signal for the sender's
        message-generation policy)

    This is exactly how the primary action is trained (same
    ratio/clip/advantage recipe) and is the ONLY mechanism that
    updates `decoder`/sender-`actor` parameters based on message
    quality. It is real, standard, and correct PPO training for a
    discrete action space -- but it is policy-gradient training via
    log-probability, not literal backpropagation "through" the
    sampling step. If you want the latter (e.g. Gumbel-softmax /
    straight-through estimators so gradients flow continuously from
    receiver outcome all the way to the sender's decoder logits),
    that is a genuine architecture change and is out of scope here.

The detached ``received_messages`` stored in the buffer are retained
for compatibility/debugging, but are NOT used as the differentiable
communication source during PPO optimization.

Eval/train lifecycle
---------------------

The model contains dropout (inside the actor's and critic's
nn.MultiheadAttention layers). Dropout must be OFF everywhere a
probability that later enters a PPO ratio is computed, or the ratio
stops being a measurement of "how has the policy changed" and starts
being contaminated by per-call dropout noise that has nothing to do
with the weight update.

Both halves of that are now enforced:

  - Rollout (`select_action`, `get_outgoing_message(s)`) always runs
    under `.eval()` -- unchanged, `__init__` puts the model in
    `.eval()` before returning, so even the very first rollout
    (before any update) is dropout-free.
  - `update()` now ALSO forces `.eval()` at its own start,
    unconditionally, regardless of what the caller left the model in
    beforehand. `.eval()` does not disable autograd or block
    `.backward()`/`optimizer.step()` -- it only changes the forward
    behavior of dropout (and batchnorm, unused here) layers. So
    `evaluate_actions()`'s forward passes now use the exact same
    (deterministic, dropout-off) computation as the rollout that
    produced `old_log_probs`/`old_communication_log_probs`, and the
    PPO ratio is a clean measurement of the actual weight change.

Net effect: dropout is never active anywhere in this pipeline. That
is a deliberate consequence of fixing the ratio, not a partial fix --
if you want dropout's regularization back, it needs to live somewhere
that doesn't feed a PPO ratio (e.g. an auxiliary head), which is out
of scope here. `train.py`'s existing `ppo.train()` call immediately
before `update()` is now a harmless no-op (immediately overridden
inside `update()`); `train()`/`eval()` remain as plain pass-throughs
for anyone who wants to toggle mode for some other purpose.

Host-level padding (host_active_mask)
----------------------------------------
See gnn_attention.py's `SharedActor._build_batch_adjacency` /
`_encode_entities` for the underlying fix: every active subnet slot
always has MAX_HOSTS host nodes, but CC4 randomizes the real host
count per zone per episode, and that count is NOT recoverable from
the observation vector (BlueFlatWrapper ANDs "host exists" together
with "host has an alert" into a single bit before it's ever written).
`SharedActor` therefore accepts an optional per-episode
`host_active_mask` to isolate the fake/padded host nodes -- but only
if a caller actually supplies real values sourced from the
environment.

This file is that caller-facing plumbing: `select_action`,
`get_outgoing_message`, `get_outgoing_messages`, `actor_forward`,
`evaluate_actions`, and `update` all accept/forward a
`host_active_mask` end to end (through both the receiver-side actor
call AND the differentiable communication reconstruction's sender-
side `get_local_hidden` call). Every one of them defaults to `None`,
which reproduces `gnn_attention.py`'s own default ("every host slot
in an active subnet is real") -- i.e. nothing breaks for existing
callers that don't supply it. Actually sourcing correct per-episode,
per-agent host counts from the environment and writing them into the
rollout buffer under the key `"host_active_mask"` (shape
`[T, NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS]`, bool) is `train.py` /
`buffer.py`'s job, and is outside this file.

HOST/SUBNET target vocabularies (num_host_targets/num_subnet_targets)
--------------------------------------------------------------------------
`target_id` is one field in a structured message (schema.py's
MESSAGE_FIELDS is unchanged, buffer/field-order layout is unchanged),
but its MEANING depends on that same message's `target_type`: HOST and
SUBNET are separate, independently-sized vocabularies (see
communication/encoder.py's two embedding tables and
communication/decoder.py's two target heads -- both selected per-row by
target_type, never mixed). `MAPPO.__init__`'s single `num_targets`
parameter is replaced by `num_host_targets` (required) and
`num_subnet_targets` (optional, `None` if that vocabulary isn't
configured), both forwarded unchanged to `MAPPOModel`/`SharedActor`,
which forward them to `StructuredCommunication`, which is expected to
size `MessageEncoder`/`MessageDecoder`'s respective HOST/SUBNET
tables from them. None of the PPO communication training logic,
field ordering, or buffer format described above changes because of
this -- `target_id` is read and written as a single field exactly as
before; only how many valid values it can take (and which embedding
table/head interprets it) now depends on `target_type`.

`save()`/`load()` persist and validate `num_host_targets`/
`num_subnet_targets` against the checkpoint, so a vocabulary-size
mismatch fails with a clear message instead of a cryptic tensor-shape
error inside `load_state_dict()` -- see `load()`.
"""

import numpy as np
import torch
import torch.nn.functional as F

from torch.distributions import Categorical

from .gnn_attention import (
    MAPPOModel,
    COMMUNICATION_DIM,
    NUM_HQ_SUBNETS,
    MAX_HOSTS,
)

from .value_norm import ValueNorm

from .config import (
    DEVICE,
    LEARNING_RATE,
    ACTOR_LEARNING_RATE,
    USE_VALUE_NORM,
    CRITIC_LEARNING_RATE,
    MAX_GRAD_NORM,
    UPDATE_EPOCHS,
    MINIBATCH_SIZE,
    PPO_CLIP,
    VALUE_CLIP,
    VALUE_LOSS_COEF,
    ENTROPY_COEF,
    COMM_POLICY_COEF,
    COMM_ENTROPY_COEF,
    NORMALIZE_ADVANTAGES,
    NUM_AGENTS,
    OBS_DIM,
)


class MAPPO:

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self, num_host_targets, num_subnet_targets=None):

        self.device = DEVICE

        # target_id is one field (schema.py's MESSAGE_FIELDS is
        # unchanged), but its vocabulary depends on target_type -- HOST
        # and SUBNET are separate, independently-sized vocabularies
        # (see communication/encoder.py's dual embedding tables and
        # communication/decoder.py's dual target heads). num_targets is
        # replaced by these two: num_host_targets is required (the host
        # vocabulary is always in play), num_subnet_targets is optional
        # (defaults to None, matching encoder/decoder's own
        # build_target_embeddings()/build_target_heads() semantics of
        # "that vocabulary isn't configured yet" rather than forcing a
        # value). Stored on self so save()/load() can validate a
        # checkpoint was produced with matching vocabulary sizes -- see
        # save()/load() below.
        self.num_host_targets = num_host_targets
        self.num_subnet_targets = num_subnet_targets

        # ------------------------------------------------------
        # Networks
        # ------------------------------------------------------

        self.model = MAPPOModel(
            num_host_targets=num_host_targets,
            num_subnet_targets=num_subnet_targets,
        ).to(self.device)
        self.actor = self.model.actor

        self.critic = self.model.critic

        # ------------------------------------------------------
        # Eval/train lifecycle
        #
        # nn.Module defaults to .train() mode on construction. Start
        # in .eval() so even the very first rollout (before any PPO
        # update) is dropout-free, matching every later rollout AND
        # every PPO-update forward pass (update() now forces .eval()
        # internally too -- see the module docstring's "Eval/train
        # lifecycle" section for why dropout must be off everywhere a
        # PPO ratio is computed, not just during rollout). Net effect:
        # dropout is never active anywhere in this pipeline.
        # ------------------------------------------------------

        self.model.eval()

        # ------------------------------------------------------
        # Structured communication
        # ------------------------------------------------------

        self.communication = (
            self.actor.communication
        )

        # ------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LEARNING_RATE,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LEARNING_RATE,
        )

        # ------------------------------------------------------
        # Gradient clipping
        # ------------------------------------------------------

        self.max_grad_norm = MAX_GRAD_NORM

        # ------------------------------------------------------
        # Value normalization
        # ------------------------------------------------------

        self.value_norm = None

        if USE_VALUE_NORM:

            self.value_norm = ValueNorm(
                device=self.device,
            )

        # ------------------------------------------------------
        # Runtime communication state
        # ------------------------------------------------------

        self.previous_messages = None

        self.current_messages = None

        self.current_decoded_messages = None

        self.trust_weights = None

    # ==========================================================
    # Tensor Utilities
    # ==========================================================

    def _to_tensor(
        self,
        value,
        dtype=torch.float32,
    ):
        """
        Convert numpy/list/tensor to a device tensor.
        """

        if isinstance(value, torch.Tensor):

            return value.to(
                device=self.device,
                dtype=dtype,
            )

        return torch.tensor(
            value,
            dtype=dtype,
            device=self.device,
        )

    # ==========================================================
    # Host-active-mask preparation
    # ==========================================================

    def _prepare_host_active_mask(
        self,
        host_active_mask,
        batch_size,
    ):
        """
        Normalize a caller-supplied per-episode host-validity mask to
        [B, NUM_HQ_SUBNETS, MAX_HOSTS] bool on the right device, or
        pass None through unchanged (gnn_attention.py's own default:
        every host slot in an active subnet is treated as real -- see
        SharedActor._build_batch_adjacency).

        Accepts either an unbatched [NUM_HQ_SUBNETS, MAX_HOSTS] mask
        (a single agent/timestep, matching how `observation` itself
        is accepted unbatched by select_action()/get_outgoing_message)
        or an already-batched [B, NUM_HQ_SUBNETS, MAX_HOSTS] /
        [1, NUM_HQ_SUBNETS, MAX_HOSTS] (broadcast to batch_size).
        """

        if host_active_mask is None:
            return None

        mask = self._to_tensor(host_active_mask, dtype=torch.bool)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if mask.dim() != 3:
            raise ValueError(
                "host_active_mask must have shape [NUM_HQ_SUBNETS, "
                "MAX_HOSTS] or [B, NUM_HQ_SUBNETS, MAX_HOSTS]. Got "
                f"{tuple(mask.shape)}."
            )

        if mask.shape[1:] != (NUM_HQ_SUBNETS, MAX_HOSTS):
            raise ValueError(
                f"host_active_mask's last two dims must be "
                f"(NUM_HQ_SUBNETS={NUM_HQ_SUBNETS}, "
                f"MAX_HOSTS={MAX_HOSTS}). Got {tuple(mask.shape)}."
            )

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)

        return mask

    # ==========================================================
    # Action Mask
    # ==========================================================

    def _apply_action_mask(
        self,
        logits,
        action_masks=None,
    ):
        """
        Apply action masks.

        True  = valid action
        False = invalid action
        """

        if action_masks is None:

            return logits

        return logits.masked_fill(
            ~action_masks.bool(),
            -1e10,
        )

    # ==========================================================
    # Action Selection
    # ==========================================================

    @torch.no_grad()
    def select_action(
        self,
        observation,
        action_mask=None,
        global_state=None,
        agent_id=None,
        received_messages=None,
        trust_weights=None,
        host_active_mask=None,
        communication_valid=None,
    ):
        """
        Select an action using the shared actor.

        Existing return interface is preserved:

            action
            log_prob
            value
            entropy

        Communication can be supplied through:

            received_messages
            trust_weights

        host_active_mask: optional, this agent's own per-episode host
        validity ([NUM_HQ_SUBNETS, MAX_HOSTS] or already-batched --
        see `_prepare_host_active_mask`). Defaults to None (no
        per-host padding isolation, matching prior behavior).
        """

        observation = self._to_tensor(
            observation,
            dtype=torch.float32,
        )

        batch_size = 1 if observation.dim() == 1 else observation.shape[0]

        prepared_host_mask = self._prepare_host_active_mask(
            host_active_mask, batch_size
        )

        # ------------------------------------------------------
        # Actor
        # ------------------------------------------------------

        logits = self.actor(
            observation,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=prepared_host_mask,
            communication_valid=communication_valid,
        )

        # ------------------------------------------------------
        # Action mask
        # ------------------------------------------------------

        if action_mask is not None:

            action_mask = self._to_tensor(
                action_mask,
                dtype=torch.bool,
            )

            logits = self._apply_action_mask(
                logits,
                action_mask,
            )

        # ------------------------------------------------------
        # Distribution
        # ------------------------------------------------------

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(
            action
        )

        entropy = distribution.entropy()

        # ------------------------------------------------------
        # Centralized critic
        # ------------------------------------------------------

        value = None

        if global_state is not None:

            global_state = self._to_tensor(
                global_state,
                dtype=torch.float32,
            )

            values = self.critic(
                global_state
            )

            if agent_id is not None:

                value = values[agent_id]

                if self.value_norm is not None:

                    value = (
                        self.value_norm.denormalize(
                            value
                        )
                    )

            else:

                value = values

                if self.value_norm is not None:

                    value = (
                        self.value_norm.denormalize(
                            value
                        )
                    )

        return (
            action.item(),
            log_prob.detach(),
            None
            if value is None
            else value.detach(),
            entropy.detach(),
        )

    # ==========================================================
    # Generate Outgoing Structured Message
    # ==========================================================

    @torch.no_grad()
    def get_outgoing_message(
        self,
        observation,
        return_decoded=False,
        host_active_mask=None,
        host_valid_mask=None,
        subnet_valid_mask=None,
    ):
        """
        Generate one agent's outgoing communication vector.

        host_active_mask: optional, this agent's own per-episode host
        validity (see `select_action`). This agent's local_hidden is
        computed from its OWN observation in both select_action() and
        here, so the same per-agent mask semantics apply.

        host_valid_mask / subnet_valid_mask: optional per-sender
        HOST/SUBNET validity, forwarded to the decoder so single-agent
        callers get the same BEFORE-sampling masking as
        get_outgoing_messages().
        """

        observation = self._to_tensor(
            observation,
            dtype=torch.float32,
        )

        batch_size = 1 if observation.dim() == 1 else observation.shape[0]

        prepared_host_mask = self._prepare_host_active_mask(
            host_active_mask, batch_size
        )

        local_hidden = (
            self.actor.get_local_hidden(
                observation,
                host_active_mask=prepared_host_mask,
            )
        )

        (
            field_ids,
            message_log_probs,
            message_entropies,
            communication_vector,
        ) = self.communication.generate_message(
            local_hidden,
            host_valid_mask=host_valid_mask,
            subnet_valid_mask=subnet_valid_mask,
        )

        communication_vector = (
            communication_vector.detach()
        )

        self.current_messages = (
            communication_vector
        )

        if return_decoded:

            # Built from the SAME field_ids used above for
            # communication_vector -- NOT a separate decoder.decode()
            # call, which is a different forward pass and isn't
            # guaranteed to agree with what was actually sampled/sent.
            structured_message = (
                self.communication.structured_message_from_ids(
                    field_ids,
                    index=0,
                )
            )

            self.current_decoded_messages = (
                structured_message
            )

            return (
                communication_vector,
                structured_message,
                field_ids,
                message_log_probs,
                message_entropies,
            )

        return communication_vector

    # ==========================================================
    # Generate Messages For All Agents
    # ==========================================================

    @torch.no_grad()
    def get_outgoing_messages(
        self,
        observations,
        return_decoded=False,
        host_active_mask=None,
        host_valid_mask=None,
        subnet_valid_mask=None,
    ):
        """
        Generate communication for every Blue agent.

        observations:

            [NUM_AGENTS, OBS_DIM]

        messages:

            [NUM_AGENTS, COMMUNICATION_DIM]

        host_active_mask: optional [NUM_AGENTS, NUM_HQ_SUBNETS,
        MAX_HOSTS] -- one row per agent, since each agent's own
        subnet slot(s) have their own independent real host counts
        this episode.

        host_valid_mask: optional [NUM_AGENTS, H] bool (H = STABLE
        host vocabulary, matching num_host_targets) -- one row per
        sender; per-episode-invalid HOST ids are masked BEFORE
        sampling (see decoder._masked_host_logits). None preserves
        the legacy unmasked behavior.

        subnet_valid_mask: optional [NUM_AGENTS, S] bool (S = STABLE
        subnet vocabulary, matching num_subnet_targets) -- one row per
        sender; unobservable SUBNET ids are masked BEFORE sampling
        (see decoder._masked_subnet_logits and
        env.get_subnet_valid_mask). None preserves the legacy unmasked
        behavior.
        """

        observations = self._to_tensor(
            observations,
            dtype=torch.float32,
        )

        if observations.ndim != 2:

            raise ValueError(
                "observations must have shape "
                "[NUM_AGENTS, OBS_DIM]"
            )

        if observations.shape[0] != NUM_AGENTS:

            raise ValueError(
                f"Expected {NUM_AGENTS} observations, "
                f"got {observations.shape[0]}"
            )

        prepared_host_mask = self._prepare_host_active_mask(
            host_active_mask, NUM_AGENTS
        )

        prepared_valid_mask = None
        if host_valid_mask is not None:
            prepared_valid_mask = self._to_tensor(
                host_valid_mask,
                dtype=torch.bool,
            )
            if prepared_valid_mask.dim() != 2 or (
                prepared_valid_mask.shape[0] != NUM_AGENTS
                or prepared_valid_mask.shape[1] != self.num_host_targets
            ):
                raise ValueError(
                    "host_valid_mask must have shape "
                    f"[NUM_AGENTS, num_host_targets]=({NUM_AGENTS}, "
                    f"{self.num_host_targets}). Got "
                    f"{tuple(prepared_valid_mask.shape)}."
                )

        prepared_subnet_mask = None
        if subnet_valid_mask is not None:
            if self.num_subnet_targets is None:
                raise ValueError(
                    "subnet_valid_mask supplied but this MAPPO instance "
                    "was constructed with num_subnet_targets=None."
                )
            prepared_subnet_mask = self._to_tensor(
                subnet_valid_mask,
                dtype=torch.bool,
            )
            if prepared_subnet_mask.dim() != 2 or (
                prepared_subnet_mask.shape[0] != NUM_AGENTS
                or prepared_subnet_mask.shape[1] != self.num_subnet_targets
            ):
                raise ValueError(
                    "subnet_valid_mask must have shape "
                    f"[NUM_AGENTS, num_subnet_targets]=({NUM_AGENTS}, "
                    f"{self.num_subnet_targets}). Got "
                    f"{tuple(prepared_subnet_mask.shape)}."
                )

        local_hidden = (
            self.actor.get_local_hidden(
                observations,
                host_active_mask=prepared_host_mask,
            )
        )

        (
            field_ids,
            message_log_probs,
            message_entropies,
            messages,
        ) = self.communication.generate_message(
            local_hidden,
            host_valid_mask=prepared_valid_mask,
            subnet_valid_mask=prepared_subnet_mask,
        )

        # ------------------------------------------------------
        # Convert v2 communication dictionaries to [N, 7]
        # arrays for the rollout buffer.
        # ------------------------------------------------------

        field_order = (
            "event_type",
            "target_type",
            "threat_level",
            "status",
            "priority",
            "confidence",
            "target_id",
        )

        communication_field_ids = torch.stack(
            [
                field_ids[name]
                for name in field_order
            ],
            dim=-1,
        )

        communication_log_probs = torch.stack(
            [
                message_log_probs[name]
                for name in field_order
            ],
            dim=-1,
        )

        communication_entropies = torch.stack(
            [
                message_entropies[name]
                for name in field_order
            ],
            dim=-1,
        )

        messages = messages.detach()

        self.current_messages = messages

        if return_decoded:

            # Built from the SAME field_ids used above for `messages`/
            # `communication_field_ids` -- NOT a separate
            # decoder.decode() forward pass, which is not guaranteed
            # to agree with what sample_message() actually sampled
            # (and therefore what was actually encoded into `messages`
            # and transmitted). This is the StructuredMessage list
            # that feeds evaluate_and_update_trust() in train.py, so a
            # mismatch here meant trust could be updated against a
            # claim the sender never actually sent.
            decoded_messages = [
                self.communication.structured_message_from_ids(
                    field_ids,
                    index=i,
                )
                for i in range(NUM_AGENTS)
            ]

            self.current_decoded_messages = (
                decoded_messages
            )

            return (
                messages,
                decoded_messages,
                communication_field_ids.detach().cpu().numpy(),
                communication_log_probs.detach().cpu().numpy(),
                communication_entropies.detach().cpu().numpy(),
            )

        return (
            messages,
            communication_field_ids.detach().cpu().numpy(),
            communication_log_probs.detach().cpu().numpy(),
            communication_entropies.detach().cpu().numpy(),
        )

    # ==========================================================
    # Build Communication Matrix
    # ==========================================================

    @torch.no_grad()
    def build_received_messages(
        self,
        communication_vectors,
    ):
        """
        Convert:

            [sender, D]

        into:

            [receiver, sender, D]
        """

        communication_vectors = (
            self._to_tensor(
                communication_vectors,
                dtype=torch.float32,
            )
        )

        return (
            self.communication.create_message_matrix(
                communication_vectors
            )
        )

    # ==========================================================
    # Apply Trust To Communication
    # ==========================================================

    @torch.no_grad()
    def get_trusted_messages(
        self,
        communication_vectors,
    ):
        """
        Apply current trust to communication vectors.
        """

        communication_vectors = (
            self._to_tensor(
                communication_vectors,
                dtype=torch.float32,
            )
        )

        return (
            self.communication.apply_trust(
                communication_vectors
            )
        )

    # ==========================================================
    # Trust Matrix
    # ==========================================================

    def get_trust_matrix(self):

        return (
            self.communication.get_trust_matrix()
        )

    # ==========================================================
    # Individual Trust
    # ==========================================================

    def get_trust(
        self,
        sender,
        receiver,
    ):

        return self.communication.get_trust(
            sender,
            receiver,
        )

    # ==========================================================
    # Update Trust
    # ==========================================================

    def update_trust(
        self,
        sender,
        receiver,
        message_quality,
        weight=1.0,
    ):

        return (
            self.communication.update_trust(
                sender=sender,
                receiver=receiver,
                message_quality=message_quality,
                weight=weight,
            )
        )

    # ==========================================================
    # Update Trust Matrix
    # ==========================================================

    def update_trust_matrix(
        self,
        evaluations,
    ):

        return (
            self.communication.update_trust_matrix(
                evaluations
            )
        )

    # ==========================================================
    # Reset Communication
    # ==========================================================

    def reset_communication(self):

        self.previous_messages = None

        self.current_messages = None

        self.current_decoded_messages = None

        self.trust_weights = None

        self.communication.reset_trust()

    # ==========================================================
    # Previous Messages
    # ==========================================================

    def set_previous_messages(
        self,
        messages,
    ):

        if messages is None:

            self.previous_messages = None

            return

        messages = self._to_tensor(
            messages,
            dtype=torch.float32,
        )

        self.previous_messages = (
            messages.detach()
        )

    # ==========================================================
    # Messages For One Receiver
    # ==========================================================

    def get_messages_for_agent(
        self,
        receiver_id,
        messages=None,
    ):
        """
        Return:

            [NUM_AGENTS, COMMUNICATION_DIM]

        for one receiver.
        """

        if messages is None:

            messages = self.previous_messages

        if messages is None:

            return None

        messages = self._to_tensor(
            messages,
            dtype=torch.float32,
        )

        if messages.ndim != 2:

            raise ValueError(
                "messages must have shape "
                "[NUM_AGENTS, COMMUNICATION_DIM]"
            )

        if not (
            0 <= receiver_id < NUM_AGENTS
        ):

            raise ValueError(
                f"receiver_id must be in "
                f"[0, {NUM_AGENTS - 1}]"
            )

        receiver_messages = (
            messages.clone()
        )

        receiver_messages[
            receiver_id
        ] = 0.0

        return receiver_messages

    # ==========================================================
    # Trust For Receiver
    # ==========================================================

    def get_trust_for_agent(
        self,
        receiver_id,
    ):
        """
        Return:

            trust[sender]

        for a fixed receiver.
        """

        trust_matrix = (
            self.get_trust_matrix()
        )

        trust_for_receiver = (
            trust_matrix[:, receiver_id]
        )

        return trust_for_receiver.to(
            device=self.device,
            dtype=torch.float32,
        )

    # ==========================================================
    # Critic
    # ==========================================================

    @torch.no_grad()
    def get_value(
        self,
        global_state,
    ):

        global_state = self._to_tensor(
            global_state,
            dtype=torch.float32,
        )

        values = self.critic(
            global_state
        )

        if self.value_norm is not None:

            values = (
                self.value_norm.denormalize(
                    values
                )
            )

        return values

    # ==========================================================
    # Train / Eval
    # ==========================================================

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

    # ==========================================================
    # Actor Forward
    # ==========================================================

    def actor_forward(
        self,
        observations,
        action_masks=None,
        received_messages=None,
        trust_weights=None,
        host_active_mask=None,
        communication_valid=None,
    ):
        """
        Standard actor forward.

        This remains available for compatibility.

        host_active_mask: optional, already-batched
        [B, NUM_HQ_SUBNETS, MAX_HOSTS] -- passed straight through to
        the actor (unlike select_action()/get_outgoing_message(), the
        caller here is expected to already have batch-aligned data,
        e.g. evaluate_actions()).

        communication_valid: optional [B] bool -- False marks
        episode-start rows with no previous message. Passed through to
        the actor so invalid rows return an exact-zero communication
        context, matching the rollout's received_messages=None path
        (see SharedActor._apply_received_communication).
        """

        logits = self.actor(
            observations,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=host_active_mask,
            communication_valid=communication_valid,
        )

        logits = self._apply_action_mask(
            logits,
            action_masks,
        )

        return logits

    # ==========================================================
    # Critic Forward
    # ==========================================================

    def critic_forward(
        self,
        global_states,
    ):

        return self.critic(
            global_states
        )

    # ==========================================================
    # DIFFERENTIABLE COMMUNICATION RECONSTRUCTION
    # ==========================================================

    def _reconstruct_received_messages(
        self,
        communication_source_obs,
        communication_field_ids,
        agent_ids,
        communication_valid=None,
        trust_weights=None,
    ):

        if communication_source_obs is None:

            return None, trust_weights

        source_obs = communication_source_obs

        # ----------------------------------------------------------
        # Shape checks
        # ----------------------------------------------------------

        if source_obs.ndim != 3:

            raise ValueError(
                "communication_source_obs must have shape "
                "[B, NUM_AGENTS, OBS_DIM]. "
                f"Got {tuple(source_obs.shape)}"
            )

        batch_size = source_obs.shape[0]

        if source_obs.shape[1] != NUM_AGENTS:

            raise ValueError(
                f"Expected {NUM_AGENTS} sender observations, "
                f"got {source_obs.shape[1]}"
            )

        if source_obs.shape[2] != OBS_DIM:

            raise ValueError(
                f"Expected OBS_DIM={OBS_DIM}, "
                f"got {source_obs.shape[2]}"
            )

        # ----------------------------------------------------------
        # Agent IDs
        # ----------------------------------------------------------

        agent_ids = agent_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        if agent_ids.ndim != 1:

            agent_ids = agent_ids.reshape(
                -1
            )

        if agent_ids.shape[0] != batch_size:

            raise ValueError(
                "agent_ids batch dimension does not match "
                "communication_source_obs."
            )

        if torch.any(
            agent_ids < 0
        ) or torch.any(
            agent_ids >= NUM_AGENTS
        ):

            raise ValueError(
                "agent_ids contains an invalid agent index."
            )

        # ----------------------------------------------------------
        # NOTE on gradients: this reconstruction ONLY runs the
        # encoder on the already-sampled, fixed `communication_field_
        # ids` -- see the module docstring's "Structured communication
        # training" section, part (a). It does NOT recompute
        # sender_hidden / re-run the decoder, so it cannot and does
        # not train the sender actor or decoder; that happens
        # separately, via the score-function path in
        # evaluate_actions()/update() (part (b) of the same section).
        # ----------------------------------------------------------

        # sender_hidden = (
        #     self.actor.get_local_hidden(
        #         sender_obs
        #     )
        # )

        # ----------------------------------------------------------
        # Structured decoder + encoder
        #
        # [B*N, hidden]
        #       ↓
        # [B*N, COMMUNICATION_DIM]
        # ----------------------------------------------------------

        # (
        #     sender_field_ids,
        #     sender_log_probs,
        #     sender_entropies,
        #     sender_messages,
        # ) = self.communication.generate_message(
        #     sender_hidden
        # )

        # sender_messages = self.communication.encoder.encode_from_ids(communication_field_ids)
        sender_field_ids = {
            "event_type": communication_field_ids[:, :, 0],
            "target_type": communication_field_ids[:, :, 1],
            "threat_level": communication_field_ids[:, :, 2],
            "status": communication_field_ids[:, :, 3],
            "priority": communication_field_ids[:, :, 4],
            "confidence": communication_field_ids[:, :, 5],
            "target_id": communication_field_ids[:, :, 6],
        }

        sender_messages = self.communication.encoder.encode_from_ids(
            sender_field_ids
        )

        # ----------------------------------------------------------
        # Restore sender dimension
        #
        # [B*N, D]
        #       ↓
        # [B, N, D]
        # ----------------------------------------------------------

        sender_messages = (
            sender_messages.reshape(
                batch_size,
                NUM_AGENTS,
                COMMUNICATION_DIM,
            )
        )

        # ----------------------------------------------------------
        # Every receiver initially sees every sender.
        #
        # [B, N_sender, D]
        #
        # No receiver expansion is required here because each PPO
        # sample already represents ONE receiver.
        # ----------------------------------------------------------

        received_messages = sender_messages

        # ----------------------------------------------------------
        # Remove self-communication.
        #
        # For sample b:
        #
        #     receiver = agent_ids[b]
        #
        # Therefore:
        #
        #     received_messages[b, receiver] = 0
        # ----------------------------------------------------------

        batch_indices = torch.arange(
            batch_size,
            device=self.device,
        )

        received_messages = (
            received_messages.clone()
        )

        received_messages[
            batch_indices,
            agent_ids,
            :
        ] = 0.0

        # ----------------------------------------------------------
        # Trust
        #
        # trust_weights is ALREADY:
        #
        #     [B, sender]
        #
        # because update() flattened:
        #
        #     [T, receiver, sender]
        #
        # into:
        #
        #     [T*N, sender]
        #
        # Therefore DO NOT expand it again.
        # ----------------------------------------------------------

        receiver_trust = None

        if trust_weights is not None:

            if trust_weights.ndim != 2:

                raise ValueError(
                    "trust_weights must have shape "
                    "[B, NUM_AGENTS]. "
                    f"Got {tuple(trust_weights.shape)}"
                )

            if trust_weights.shape[0] != batch_size:

                raise ValueError(
                    "trust_weights batch dimension does not "
                    "match communication_source_obs."
                )

            if trust_weights.shape[1] != NUM_AGENTS:

                raise ValueError(
                    f"Expected trust for {NUM_AGENTS} senders, "
                    f"got {trust_weights.shape[1]}"
                )

            receiver_trust = torch.clamp(
                trust_weights,
                0.0,
                1.0,
            )

        # ----------------------------------------------------------
        # Episode boundary handling
        # ----------------------------------------------------------

        if communication_valid is not None:

            valid = communication_valid

            if valid.ndim > 1:

                valid = valid.reshape(
                    batch_size
                )

            valid = valid.bool()

            if valid.shape[0] != batch_size:

                raise ValueError(
                    "communication_valid must have shape [B]."
                )

            # Invalid communication samples receive zero messages.
            received_messages = (
                received_messages
                * valid.to(
                    dtype=received_messages.dtype
                ).view(
                    batch_size,
                    1,
                    1,
                )
            )

            if receiver_trust is not None:

                receiver_trust = (
                    receiver_trust
                    * valid.to(
                        dtype=receiver_trust.dtype
                    ).view(
                        batch_size,
                        1,
                    )
                )

        return (
            received_messages,
            receiver_trust,
        )

    # ==========================================================
    # Evaluate Actions
    # ==========================================================

    def evaluate_actions(
        self,
        observations,
        global_states,
        actions,
        agent_ids,
        action_masks=None,
        received_messages=None,
        trust_weights=None,
        communication_source_obs=None,
        communication_field_ids=None,
        communication_valid=None,
        host_active_mask=None,
        communication_host_active_mask=None,
        communication_host_valid=None,
        communication_subnet_valid=None,
    ):
        """
        Evaluate actions during PPO optimization.

        Communication modes
        -------------------

        1. Differentiable mode:

            communication_source_obs is supplied.

            The receiver's incoming communication vector is rebuilt
            from the fixed, already-sampled communication_field_ids
            via the CURRENT encoder (see module docstring, part (a)
            -- this trains encoder/receiver-side parameters, NOT the
            sender's decoder/actor). Separately, communication_log_
            probs/communication_entropy are recomputed by re-running
            the CURRENT sender actor + decoder over the stored
            communication_field_ids (part (b) -- this is what trains
            decoder/sender-actor parameters, via the score-function
            communication_actor_loss built in update()).

        2. Legacy mode:

            received_messages is supplied.

            These are treated as already-generated tensors.

        3. No communication:

            both are None.

        host_active_mask: optional [B, NUM_HQ_SUBNETS, MAX_HOSTS] --
        the RECEIVER's own per-host validity, used for the
        `actor_forward` call below (the receiver's own observation).

        communication_host_active_mask: optional
        [B, NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS] -- each SENDER's
        own per-host validity, used when recomputing sender_hidden
        for communication_log_probs/entropy above. Mirrors how
        communication_source_obs itself carries one observation per
        sender.

        communication_host_valid: optional [B, NUM_AGENTS, H] bool
        (H = STABLE host vocabulary) -- each SENDER's HOST-target
        validity for the episode that row was sampled in. Replayed
        through the CURRENT decoder together with the stored ids so
        old/new target_id log-probs stay comparable (see decoder.
        _masked_host_logits). None replays unmasked (legacy).

        communication_subnet_valid: optional [B, NUM_AGENTS, S] bool
        (S = STABLE subnet vocabulary) -- each SENDER's SUBNET-target
        validity for the episode that row was sampled in. Replayed
        through the CURRENT decoder together with the stored ids so
        old/new target_id log-probs stay comparable (see decoder.
        _masked_subnet_logits). None replays unmasked (legacy).

        Returns
        -------

        log_probs
        entropy
        values
        communication_log_prob_fields ([B, NUM_AGENTS, 7] per-field
        log-probs in mappo field_order, NOT the joint sum -- update()
        builds per-field PPO ratios from these)
        communication_entropy
        """

        # ======================================================
        # Differentiable communication reconstruction
        # ======================================================

        if communication_source_obs is not None:
            received_messages, reconstructed_trust = (
                self._reconstruct_received_messages(
                    communication_source_obs=communication_source_obs,
                    communication_field_ids=communication_field_ids,
                    agent_ids=agent_ids,
                    communication_valid=communication_valid,
                    trust_weights=trust_weights,
                )
            )

            if reconstructed_trust is not None:

                trust_weights = (
                    reconstructed_trust
                )

        # ======================================================
        # Actor
        # ======================================================

        communication_log_probs = None
        communication_entropy = None
        if (
            communication_source_obs is not None
            and communication_field_ids is not None
        ):

            batch_size = communication_source_obs.shape[0]

            sender_obs = communication_source_obs.reshape(
                batch_size * NUM_AGENTS,
                OBS_DIM,
            )

            flat_comm_host_mask = None

            if communication_host_active_mask is not None:

                if communication_host_active_mask.shape != (
                    batch_size,
                    NUM_AGENTS,
                    NUM_HQ_SUBNETS,
                    MAX_HOSTS,
                ):

                    raise ValueError(
                        "communication_host_active_mask must have "
                        f"shape ({batch_size}, {NUM_AGENTS}, "
                        f"{NUM_HQ_SUBNETS}, {MAX_HOSTS}). Got "
                        f"{tuple(communication_host_active_mask.shape)}."
                    )

                flat_comm_host_mask = (
                    communication_host_active_mask
                    .to(device=self.device, dtype=torch.bool)
                    .reshape(
                        batch_size * NUM_AGENTS,
                        NUM_HQ_SUBNETS,
                        MAX_HOSTS,
                    )
                )

            sender_hidden = self.actor.get_local_hidden(
                sender_obs,
                host_active_mask=flat_comm_host_mask,
            )

            # Select the receiver's relevant sender observations.
            #
            # We need all NUM_AGENTS because the decoder produces
            # a message for every sender.
            sender_hidden = sender_hidden.reshape(
                batch_size,
                NUM_AGENTS,
                -1,
            )

            # Evaluate stored message IDs under CURRENT decoder.
            #
            # evaluate_message() expects [B, hidden] so flatten first.
            flat_sender_hidden = sender_hidden.reshape(
                batch_size * NUM_AGENTS,
                -1,
            )

            flat_field_ids = {
                key: value.reshape(
                    batch_size * NUM_AGENTS
                )
                for key, value in {
                    "event_type": communication_field_ids[:, :, 0],
                    "target_type": communication_field_ids[:, :, 1],
                    "threat_level": communication_field_ids[:, :, 2],
                    "status": communication_field_ids[:, :, 3],
                    "priority": communication_field_ids[:, :, 4],
                    "confidence": communication_field_ids[:, :, 5],
                    "target_id": communication_field_ids[:, :, 6],
                }.items()
            }

            flat_host_valid_mask = None

            if communication_host_valid is not None:

                if communication_host_valid.shape != (
                    batch_size,
                    NUM_AGENTS,
                    self.num_host_targets,
                ):

                    raise ValueError(
                        "communication_host_valid must have shape "
                        f"({batch_size}, {NUM_AGENTS}, "
                        f"{self.num_host_targets}). Got "
                        f"{tuple(communication_host_valid.shape)}."
                    )

                flat_host_valid_mask = (
                    communication_host_valid
                    .to(device=self.device, dtype=torch.bool)
                    .reshape(
                        batch_size * NUM_AGENTS,
                        self.num_host_targets,
                    )
                )

            flat_subnet_valid_mask = None

            if communication_subnet_valid is not None:

                if self.num_subnet_targets is None:

                    raise ValueError(
                        "communication_subnet_valid supplied but this "
                        "MAPPO instance was constructed with "
                        "num_subnet_targets=None."
                    )

                if communication_subnet_valid.shape != (
                    batch_size,
                    NUM_AGENTS,
                    self.num_subnet_targets,
                ):

                    raise ValueError(
                        "communication_subnet_valid must have shape "
                        f"({batch_size}, {NUM_AGENTS}, "
                        f"{self.num_subnet_targets}). Got "
                        f"{tuple(communication_subnet_valid.shape)}."
                    )

                flat_subnet_valid_mask = (
                    communication_subnet_valid
                    .to(device=self.device, dtype=torch.bool)
                    .reshape(
                        batch_size * NUM_AGENTS,
                        self.num_subnet_targets,
                    )
                )

            message_log_probs, message_entropies = (
                self.communication.decoder.evaluate_message(
                    flat_sender_hidden,
                    flat_field_ids,
                    flat_host_valid_mask,
                    flat_subnet_valid_mask,
                )
            )

            # Explicit field order (mappo/buffer layout -- NOT the
            # decoder dict order, so old/new stay aligned by
            # construction). Kept PER-FIELD [B, N, 7]: update() builds
            # independent PPO ratios per field instead of clipping one
            # joint exp(sum) ratio, which saturated ~7x too readily.
            comm_field_order = (
                "event_type",
                "target_type",
                "threat_level",
                "status",
                "priority",
                "confidence",
                "target_id",
            )

            communication_log_probs = torch.stack(
                [
                    message_log_probs[name]
                    for name in comm_field_order
                ],
                dim=-1,
            ).reshape(
                batch_size,
                NUM_AGENTS,
                7,
            )

            communication_entropy = torch.stack(
                list(message_entropies.values()),
                dim=-1,
            ).mean(dim=-1)

            communication_entropy = (
                communication_entropy.reshape(
                    batch_size,
                    NUM_AGENTS,
                )
            )

        logits = self.actor_forward(
            observations,
            action_masks,
            received_messages=received_messages,
            trust_weights=trust_weights,
            host_active_mask=host_active_mask,
            communication_valid=communication_valid,
        )

        # ======================================================
        # Action distribution
        # ======================================================

        distribution = Categorical(
            logits=logits
        )

        log_probs = distribution.log_prob(
            actions
        )

        entropy = distribution.entropy()

        # ======================================================
        # Centralized critic
        # ======================================================

        values = self.critic_forward(
            global_states
        )

        values = values[
            torch.arange(
                values.size(0),
                device=self.device,
            ),
            agent_ids,
        ]

        return (
            log_probs,
            entropy,
            values,
            communication_log_probs,
            communication_entropy,
        )

    # ==========================================================
    # Save
    # ==========================================================

    def save(
        self,
        path,
    ):
        """
        Save MAPPO checkpoint.

        Also saves num_host_targets/num_subnet_targets (the sizes this
        instance's host/subnet target embeddings and heads were
        actually constructed with) so load() can catch a vocabulary-
        size mismatch with a clear error instead of a cryptic tensor-
        shape mismatch from load_state_dict(). See load().
        """

        trust_state = None

        if hasattr(
            self.communication,
            "get_trust_state",
        ):

            trust_state = (
                self.communication.get_trust_state()
            )

        checkpoint = {

            "model":
                self.model.state_dict(),

            "actor_optimizer":
                self.actor_optimizer.state_dict(),

            "critic_optimizer":
                self.critic_optimizer.state_dict(),

            "value_norm":
                None
                if self.value_norm is None
                else self.value_norm.state_dict(),

            "trust_state":
                trust_state,

            "num_host_targets":
                self.num_host_targets,

            "num_subnet_targets":
                self.num_subnet_targets,
        }

        torch.save(
            checkpoint,
            path,
        )

    # ==========================================================
    # Load
    # ==========================================================

    def load(
        self,
        path,
    ):
        """
        Load MAPPO checkpoint.

        Old checkpoints without trust state remain compatible. Old
        checkpoints without num_host_targets/num_subnet_targets (saved
        before the HOST/SUBNET target split) also remain loadable --
        those keys are simply absent, so the vocabulary-size check
        below is skipped rather than failing. Old checkpoints WITH the
        old single-vocabulary architecture will still fail inside
        load_state_dict() below on a genuine shape mismatch (there's no
        way to reinterpret a single target_embedding/target_head as
        two separate ones), just as before this change -- this method
        only makes the NEW two-vocabulary case fail with a clear
        message instead of a cryptic one.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        # ------------------------------------------------------
        # Vocabulary-size validation
        #
        # This MAPPO instance's host_target_embedding/host_target_head
        # (and the subnet equivalents) were already constructed at a
        # fixed size in __init__/MAPPOModel -- load_state_dict() below
        # can only succeed if the checkpoint's saved tensors are that
        # exact size. Checking here first turns a shape-mismatch stack
        # trace deep inside load_state_dict() into an actionable
        # message naming exactly which vocabulary and which two sizes
        # disagree.
        # ------------------------------------------------------

        checkpoint_num_host_targets = checkpoint.get(
            "num_host_targets"
        )

        if (
            checkpoint_num_host_targets is not None
            and checkpoint_num_host_targets != self.num_host_targets
        ):

            raise ValueError(
                "Checkpoint was saved with "
                f"num_host_targets={checkpoint_num_host_targets}, but "
                "this MAPPO instance was constructed with "
                f"num_host_targets={self.num_host_targets}. Reconstruct "
                "MAPPO with the checkpoint's value before calling "
                "load()."
            )

        checkpoint_num_subnet_targets = checkpoint.get(
            "num_subnet_targets"
        )

        if (
            checkpoint_num_subnet_targets is not None
            and checkpoint_num_subnet_targets != self.num_subnet_targets
        ):

            raise ValueError(
                "Checkpoint was saved with "
                f"num_subnet_targets={checkpoint_num_subnet_targets}, "
                "but this MAPPO instance was constructed with "
                f"num_subnet_targets={self.num_subnet_targets}. "
                "Reconstruct MAPPO with the checkpoint's value before "
                "calling load()."
            )

        model_state = checkpoint["model"]

        # Backward compatibility with checkpoints saved before the
        # per-sample topology refactor: those contain the old fixed
        # `actor.gnn{1,2}.adjacency` buffers, while the current model
        # builds `actor.structural_adjacency` / `actor.real_topology`
        # instead. Drop the obsolete keys and load non-strictly so the
        # (fixed, non-learned) topology buffers simply keep their
        # freshly constructed values. Any other missing/unexpected key
        # still fails loudly -- this only whitelists the known rename.
        obsolete_keys = {"actor.gnn1.adjacency", "actor.gnn2.adjacency"}
        model_state = {
            key: value
            for key, value in model_state.items()
            if key not in obsolete_keys
        }
        missing, unexpected = self.model.load_state_dict(
            model_state, strict=False
        )
        expected_missing = {
            "actor.structural_adjacency",
            "actor.real_topology",
        }
        unexpected_missing = set(missing) - expected_missing
        if unexpected_missing:
            raise RuntimeError(
                "Checkpoint is missing model keys: "
                + ", ".join(sorted(unexpected_missing))
            )
        if set(unexpected):
            raise RuntimeError(
                "Checkpoint has unexpected model keys: "
                + ", ".join(sorted(set(unexpected)))
            )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        if (
            self.value_norm is not None
            and checkpoint.get(
                "value_norm"
            ) is not None
        ):

            self.value_norm.load_state_dict(
                checkpoint["value_norm"]
            )

        trust_state = checkpoint.get(
            "trust_state"
        )

        if (
            trust_state is not None
            and hasattr(
                self.communication,
                "load_trust_state",
            )
        ):

            self.communication.load_trust_state(
                trust_state
            )

        # Loading a checkpoint doesn't go through __init__, so the
        # same eval-by-default guarantee is restored here explicitly
        # -- otherwise a freshly loaded model would sit in whatever
        # mode it happened to be saved in.
        self.model.eval()

    # ==========================================================
    # PPO UPDATE
    # ==========================================================

    def update(
        self,
        buffer,
    ):
        """
        Perform one MAPPO update.

        Communication training
        ----------------------

        The buffer provides:

            communication_source_obs
            communication_valid
            trust_weights
            host_active_mask (optional; see module docstring)

        The source observations are passed through the CURRENT
        sender communication network during every PPO minibatch.
        See the module docstring's "Structured communication
        training" section for exactly what is and isn't
        differentiable here -- in short, the encoder/receiver side is
        truly differentiable; the decoder/sender-actor side is
        trained via the separate score-function
        communication_actor_loss below, not literal backprop.

        ``received_messages`` is deliberately NOT used as the
        differentiable source when ``communication_source_obs``
        exists.

        Communication credit assignment: see module docstring. The
        communication advantage for each (sender, receiver) pair is
        re-weighted by that receiver's trust in that sender
        (``mb_trust_weights``) before being used in
        ``comm_surrogate1/2`` below, so a receiver's advantage doesn't
        reinforce every incoming sender equally.

        Eval/train lifecycle: this method forces the model into
        `.eval()` at the very start (see module docstring), regardless
        of what mode the caller left it in, so every forward pass
        computed here uses the same dropout-off behavior as the
        rollout that produced `old_log_probs`/
        `old_communication_log_probs` -- the PPO ratio needs both
        sides computed under the same conditions to mean what it's
        supposed to mean. `.eval()` does not block gradients or
        `optimizer.step()`.
        """

        # `.eval()` is forced here, not left to the caller (see
        # module + method docstrings): dropout must be off for every
        # forward pass in this method, and that has to hold
        # regardless of whether train.py remembered to call
        # ppo.eval() first.
        self.model.eval()

        batch = buffer.get_batches()

        # ======================================================
        # Core PPO tensors
        # ======================================================

        obs = batch["obs"]

        global_obs = batch["global_obs"]

        actions = batch["actions"]

        old_log_probs = batch["log_probs"]

        returns = batch["returns"]

        advantages = batch["advantages"]

        action_masks = batch["action_masks"]

        old_values = batch["values"]

        # ======================================================
        # Communication tensors
        # ======================================================

        communication_source_obs = batch.get(
            "communication_source_obs",
            None,
        )

        communication_valid = batch.get(
            "communication_valid",
            None,
        )

        trust_weights = batch.get(
            "trust_weights",
            None,
        )
        communication_field_ids = batch.get(
            "communication_field_ids",
            None,
        )

        old_communication_log_probs = batch.get(
            "communication_log_probs",
            None,
        )

        # Per-sender HOST-target validity for the episode each row was
        # sampled in ([T, N, H], STABLE host order). Absent for legacy
        # buffers -- replay then trains unmasked, exactly as before.
        # Shape-validated in the minibatch block below (where T is
        # known), mirroring communication_source_obs.
        communication_host_valid = batch.get(
            "communication_host_valid",
            None,
        )

        # Per-sender SUBNET-target validity for the episode each row was
        # sampled in ([T, N, S], STABLE subnet order). Absent for legacy
        # buffers -- replay then trains unmasked, exactly as before.
        communication_subnet_valid = batch.get(
            "communication_subnet_valid",
            None,
        )

        # ------------------------------------------------------
        # Fallback for older buffers
        # ------------------------------------------------------

        received_messages = batch.get(
            "received_messages",
            None,
        )

        # ------------------------------------------------------
        # Host validity (see module docstring). Raw buffer shape:
        # [T, NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS]. Kept in this
        # raw form for now -- it's reshaped two different ways below
        # (receiver-flattened, and sender-expanded-per-receiver),
        # mirroring how `communication_source_obs` itself is handled.
        # ------------------------------------------------------

        host_active_mask = batch.get(
            "host_active_mask",
            None,
        )

        # ======================================================
        # Dimensions
        # ======================================================

        T = obs.shape[0]

        # ======================================================
        # Flatten local actor data
        # ======================================================

        obs = obs.reshape(
            T * NUM_AGENTS,
            OBS_DIM,
        )

        actions = actions.reshape(
            T * NUM_AGENTS,
        )

        old_log_probs = old_log_probs.reshape(
            T * NUM_AGENTS,
        )

        returns = returns.reshape(
            T * NUM_AGENTS,
        )

        advantages = advantages.reshape(
            T * NUM_AGENTS,
        )

        old_values = old_values.reshape(
            T * NUM_AGENTS,
        )

        action_masks = action_masks.reshape(
            T * NUM_AGENTS,
            -1,
        )

        # ======================================================
        # Flatten communication source observations
        # ======================================================

        if communication_source_obs is not None:

            communication_source_obs = (
                communication_source_obs.to(
                    device=self.device,
                    dtype=torch.float32,
                )
            )

            expected_source_shape = (T, NUM_AGENTS, OBS_DIM)

            if tuple(communication_source_obs.shape) != expected_source_shape:

                raise ValueError(
                    "communication_source_obs must have shape "
                    f"{expected_source_shape} (T, NUM_AGENTS, "
                    f"OBS_DIM); got "
                    f"{tuple(communication_source_obs.shape)}."
                )

        # ======================================================
        # Flatten communication validity
        # ======================================================

        if communication_valid is not None:

            communication_valid = (
                communication_valid.to(
                    device=self.device,
                    dtype=torch.bool,
                )
            )

        # ======================================================
        # HOST-target validity (see communication_host_valid fetch
        # above). Raw buffer shape [T, NUM_AGENTS, H]; H must equal
        # this instance's host vocabulary or mask positions would
        # silently misalign with the decoder head.
        # ======================================================

        if communication_host_valid is not None:

            communication_host_valid = (
                communication_host_valid.to(
                    device=self.device,
                    dtype=torch.bool,
                )
            )

            expected_valid_shape = (T, NUM_AGENTS, self.num_host_targets)

            if tuple(communication_host_valid.shape) != expected_valid_shape:

                raise ValueError(
                    "communication_host_valid must have shape "
                    f"{expected_valid_shape} (T, NUM_AGENTS, "
                    f"num_host_targets); got "
                    f"{tuple(communication_host_valid.shape)}."
                )

        # ======================================================
        # SUBNET-target validity (see communication_subnet_valid fetch
        # above). Raw buffer shape [T, NUM_AGENTS, S]; S must equal
        # this instance's subnet vocabulary or mask positions would
        # silently misalign with the decoder head.
        # ======================================================

        if communication_subnet_valid is not None:

            if self.num_subnet_targets is None:

                raise ValueError(
                    "communication_subnet_valid present in buffer but "
                    "this MAPPO instance was constructed with "
                    "num_subnet_targets=None."
                )

            communication_subnet_valid = (
                communication_subnet_valid.to(
                    device=self.device,
                    dtype=torch.bool,
                )
            )

            expected_subnet_shape = (T, NUM_AGENTS, self.num_subnet_targets)

            if tuple(communication_subnet_valid.shape) != expected_subnet_shape:

                raise ValueError(
                    "communication_subnet_valid must have shape "
                    f"{expected_subnet_shape} (T, NUM_AGENTS, "
                    f"num_subnet_targets); got "
                    f"{tuple(communication_subnet_valid.shape)}."
                )

        # ======================================================
        # Flatten fallback messages
        # ======================================================

        if received_messages is not None:

            received_messages = (
                received_messages.reshape(
                    T * NUM_AGENTS,
                    NUM_AGENTS,
                    COMMUNICATION_DIM,
                )
            )

        # ======================================================
        # Trust
        #
        # Buffer convention:
        #
        # trust_weights[t, receiver, sender]
        #
        # We need:
        #
        # [T*N, sender]
        #
        # so each receiver sample gets its own trust vector.
        # ======================================================

        if trust_weights is not None:

            trust_weights = (
                trust_weights.reshape(
                    T * NUM_AGENTS,
                    NUM_AGENTS,
                )
            )

        # ======================================================
        # Host validity
        #
        # Buffer convention: host_active_mask[t, agent, subnet_slot,
        # host_slot] -- one mask per (timestep, agent), reused for
        # two purposes below:
        #
        #   receiver-flattened: exactly like `obs` -- row (t,agent)
        #   is that agent's OWN mask when it was the one acting.
        #
        #   sender-expanded: exactly like `communication_source_obs`
        #   -- broadcast across the receiver dimension so every
        #   receiver's minibatch row carries all NUM_AGENTS senders'
        #   masks, for the differentiable reconstruction's per-sender
        #   get_local_hidden() call.
        # ======================================================

        host_active_mask_flat = None
        host_active_mask_raw = None

        if host_active_mask is not None:

            host_active_mask_raw = host_active_mask.to(
                device=self.device, dtype=torch.bool
            )

            expected_host_shape = (
                T, NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS
            )

            if tuple(host_active_mask_raw.shape) != expected_host_shape:

                raise ValueError(
                    "host_active_mask must have shape "
                    f"{expected_host_shape} (T, NUM_AGENTS, "
                    f"NUM_HQ_SUBNETS, MAX_HOSTS); got "
                    f"{tuple(host_active_mask_raw.shape)}."
                )

            host_active_mask_flat = host_active_mask_raw.reshape(
                T * NUM_AGENTS, NUM_HQ_SUBNETS, MAX_HOSTS
            )

        # ======================================================
        # Centralized critic input
        # ======================================================

        global_obs = (
            global_obs.repeat_interleave(
                NUM_AGENTS,
                dim=0,
            )
        )

        # ======================================================
        # Agent IDs
        #
        # Pattern:
        #
        # timestep 0:
        #   0 1 2 3 4
        #
        # timestep 1:
        #   0 1 2 3 4
        #
        # ...
        # ======================================================

        agent_ids = torch.arange(
            NUM_AGENTS,
            device=self.device,
        ).repeat(T)

        # ======================================================
        # Advantage normalization
        # ======================================================

        if NORMALIZE_ADVANTAGES:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

        # ======================================================
        # Value normalization
        #
        # old_values in the buffer are DENORMALIZED (select_action()
        # denormalizes before storage). They must be renormalized with
        # the SAME statistics that were active at rollout time to
        # recover the original normalized critic prediction for PPO
        # value clipping. update(returns) below changes those
        # statistics, so snapshot the normalized old values FIRST
        # (using the pre-update stats), then update, then normalize
        # returns with the new stats. Re-normalizing old values AFTER
        # the update (the old code) compared current vs "old" values
        # in different normalized spaces.
        # ======================================================

        old_values_normalized = None

        if self.value_norm is not None:

            with torch.no_grad():
                old_values_normalized = (
                    self.value_norm.normalize(old_values).detach()
                )

            self.value_norm.update(
                returns
            )

        # ======================================================
        # Statistics
        # ======================================================

        actor_loss_epoch = 0.0

        critic_loss_epoch = 0.0

        entropy_epoch = 0.0

        # ======================================================
        # Dataset size
        # ======================================================

        dataset_size = obs.shape[0]

        # ======================================================
        # PPO epochs
        # ======================================================

        for epoch in range(
            UPDATE_EPOCHS
        ):

            permutation = torch.randperm(
                dataset_size,
                device=self.device,
            )

            # --------------------------------------------------
            # Minibatches
            # --------------------------------------------------

            for start in range(
                0,
                dataset_size,
                MINIBATCH_SIZE,
            ):

                end = (
                    start
                    + MINIBATCH_SIZE
                )

                idx = permutation[
                    start:end
                ]

                # ==================================================
                # Core minibatch
                # ==================================================

                mb_obs = obs[idx]

                mb_global = (
                    global_obs[idx]
                )

                mb_actions = (
                    actions[idx]
                )

                mb_old_log_probs = (
                    old_log_probs[idx]
                )

                mb_returns = (
                    returns[idx]
                )

                mb_advantages = (
                    advantages[idx]
                )

                mb_old_values = (
                    old_values[idx]
                )

                mb_action_masks = (
                    action_masks[idx]
                )

                mb_agent_ids = (
                    agent_ids[idx]
                )
                mb_comm_field_ids = None
                mb_old_comm_fields = None
                if communication_field_ids is not None:
                    comm_ids = (
                        communication_field_ids
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            7,
                        )
                    )
                    mb_comm_field_ids = comm_ids[idx]
                if old_communication_log_probs is not None:
                    # Buffer stores log-probability for each of the
                    # 7 message fields, in mappo field_order:
                    #
                    # [T, N, 7]
                    #
                    # Kept PER-FIELD (not summed): update() builds an
                    # independent PPO ratio per field below, so a small
                    # drift in every field no longer clips as one
                    # saturated joint ratio. The [B, N, 7] layout here
                    # matches new_comm_log_prob_fields from
                    # evaluate_actions() by construction (same order).
                    #
                    # Expand receiver dimension so every receiver
                    # gets the sender log-probabilities.

                    old_comm = (
                        old_communication_log_probs
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            7,
                        )
                    )

                    mb_old_comm_fields = old_comm[idx]

                # ==================================================
                # Value targets
                # ==================================================

                if self.value_norm is not None:

                    normalized_returns = (
                        self.value_norm.normalize(
                            mb_returns
                        )
                    )

                else:

                    normalized_returns = (
                        mb_returns
                    )

                # ==================================================
                # COMMUNICATION SOURCE
                # ==================================================

                mb_source_obs = None
                mb_comm_valid = None

                if communication_source_obs is not None:

                    # ------------------------------------------------------
                    # Buffer:
                    #
                    # communication_source_obs
                    #     [T, N, OBS]
                    #
                    # Expand the receiver dimension:
                    #
                    #     [T, receiver, sender, OBS]
                    #
                    # Then flatten timestep + receiver:
                    #
                    #     [T*N, sender, OBS]
                    #
                    # This now has the SAME ordering as:
                    #
                    # obs
                    # actions
                    # agent_ids
                    # ------------------------------------------------------

                    source = (
                        communication_source_obs
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            OBS_DIM,
                        )
                    )

                    mb_source_obs = (
                        source[idx]
                    )

                    # ------------------------------------------------------
                    # Communication validity
                    # ------------------------------------------------------

                    if communication_valid is not None:

                        valid = (
                            communication_valid
                            .unsqueeze(1)
                            .expand(
                                -1,
                                NUM_AGENTS,
                            )
                            .reshape(
                                T * NUM_AGENTS
                            )
                        )

                        mb_comm_valid = (
                            valid[idx]
                        )

                # ==================================================
                # Fallback communication
                # ==================================================

                mb_received_messages = None

                if (
                    mb_source_obs is None
                    and received_messages is not None
                ):

                    mb_received_messages = (
                        received_messages[idx]
                    )

                # ==================================================
                # Trust
                # ==================================================

                mb_trust_weights = None

                if trust_weights is not None:

                    mb_trust_weights = (
                        trust_weights[idx]
                    )

                # ==================================================
                # Host validity
                #
                # Receiver-side: same [T*N, ...] indexing as mb_obs.
                # Sender-side: expanded across the receiver dimension
                # exactly like mb_source_obs above, only actually
                # built when communication_source_obs is in play
                # (otherwise there's no differentiable reconstruction
                # to feed it to).
                # ==================================================

                mb_host_active_mask = None

                if host_active_mask_flat is not None:

                    mb_host_active_mask = (
                        host_active_mask_flat[idx]
                    )

                mb_comm_host_active_mask = None

                if (
                    communication_source_obs is not None
                    and host_active_mask_raw is not None
                ):

                    comm_host_mask = (
                        host_active_mask_raw
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            NUM_HQ_SUBNETS,
                            MAX_HOSTS,
                        )
                    )

                    mb_comm_host_active_mask = (
                        comm_host_mask[idx]
                    )

                # ==================================================
                # HOST-target validity
                #
                # Same receiver-expansion as mb_source_obs above:
                # every receiver row carries all NUM_AGENTS senders'
                # per-episode HOST validity, replayed through the
                # CURRENT decoder in evaluate_actions().
                # ==================================================

                mb_comm_host_valid = None

                if communication_host_valid is not None:

                    comm_valid_mask = (
                        communication_host_valid
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            self.num_host_targets,
                        )
                    )

                    mb_comm_host_valid = (
                        comm_valid_mask[idx]
                    )

                # ==================================================
                # SUBNET-target validity
                #
                # Same receiver-expansion as HOST above.
                # ==================================================

                mb_comm_subnet_valid = None

                if communication_subnet_valid is not None:

                    comm_subnet_mask = (
                        communication_subnet_valid
                        .unsqueeze(1)
                        .expand(
                            -1,
                            NUM_AGENTS,
                            -1,
                            -1,
                        )
                        .reshape(
                            T * NUM_AGENTS,
                            NUM_AGENTS,
                            self.num_subnet_targets,
                        )
                    )

                    mb_comm_subnet_valid = (
                        comm_subnet_mask[idx]
                    )

                # ==================================================
                # Forward
                # ==================================================

                (
                    new_log_probs,
                    entropy,
                    values,
                    new_comm_log_prob_fields,
                    communication_entropy,
                ) = self.evaluate_actions(
                    mb_obs,
                    mb_global,
                    mb_actions,
                    mb_agent_ids,
                    mb_action_masks,
                    received_messages=(
                        mb_received_messages
                    ),
                    communication_field_ids=mb_comm_field_ids,
                    trust_weights=(
                        mb_trust_weights
                    ),

                    communication_source_obs=(
                        mb_source_obs
                    ),

                    communication_valid=(
                        mb_comm_valid
                    ),

                    host_active_mask=mb_host_active_mask,

                    communication_host_active_mask=(
                        mb_comm_host_active_mask
                    ),

                    communication_host_valid=(
                        mb_comm_host_valid
                    ),

                    communication_subnet_valid=(
                        mb_comm_subnet_valid
                    ),
                )

                # ==================================================
                # PPO ratio
                # ==================================================

                ratio = torch.exp(
                    new_log_probs
                    - mb_old_log_probs
                )

                # --------------------------------------------------
                # Communication PPO term. new_comm_log_prob_fields
                # / mb_old_comm_fields can legitimately both be
                # None (communication_field_ids absent from this
                # buffer/minibatch -- e.g. a run with communication
                # disabled), since both are sourced via
                # batch.get(key, None). Guard against that instead of
                # crashing: the communication loss terms simply
                # contribute nothing when there's no communication
                # data to train on.
                # --------------------------------------------------

                have_communication = (
                    new_comm_log_prob_fields is not None
                    and mb_old_comm_fields is not None
                )

                # --------------------------------------------------
                # Mask (a) invalid communication rows (episode-start
                # placeholder field-ids / anything communication_valid
                # =False), AND (b) self-communication -- sender ==
                # receiver never actually reaches this receiver's
                # decision (it's zeroed out of received_messages in
                # _reconstruct_received_messages), so it shouldn't be
                # policy-gradient-updated using this row's advantage
                # either.
                #
                # These are HARD validity masks only -- they decide
                # which rows are real samples for averaging purposes
                # (actor_valid_count below). base_comm_mask covers
                # every reachable pair (entropy); actor_comm_mask
                # additionally excludes empty messages (actor loss).
                # Per-sender CREDIT (how much of this row's advantage
                # each sender should get) is a separate, soft weighting
                # applied afterward -- see credit_weight.
                #
                # mb_agent_ids: [B]        -> this row's receiver id
                # comm_surrogate*: [B, N, 7] -> N = sender dim, 7 fields
                # --------------------------------------------------

                sender_index = torch.arange(
                    NUM_AGENTS,
                    device=self.device,
                ).view(1, -1)

                not_self_mask = (
                    sender_index
                    != mb_agent_ids.view(-1, 1)
                ).to(dtype=mb_advantages.dtype)

                # base_comm_mask: every (receiver, sender) pair whose
                # message actually reaches the receiver's decision
                # (episode-start placeholders excluded via
                # communication_valid, self-messages excluded via
                # not_self_mask). Used for the entropy bonus.
                if mb_comm_valid is not None:

                    base_comm_mask = (
                        not_self_mask
                        * mb_comm_valid
                        .to(dtype=mb_advantages.dtype)
                        .view(-1, 1)
                    )

                else:

                    base_comm_mask = not_self_mask

                # actor_comm_mask: base mask MINUS empty (NONE-event)
                # messages. Saying nothing carries no claim, so there
                # is nothing to reinforce with the receiver's advantage
                # -- this mirrors train.py, which already skips
                # is_empty() messages for trust updates.
                # mb_comm_field_ids is [B, N, 7] in mappo.py's own
                # field order (index 0 == event_type, NONE == 0).
                # Entropy deliberately keeps base_comm_mask (see
                # below): exploration must survive silence, or an
                # all-NONE policy could never recover.
                if mb_comm_field_ids is not None:
                    non_empty = (
                        mb_comm_field_ids[:, :, 0] != 0
                    ).to(dtype=mb_advantages.dtype)
                    actor_comm_mask = base_comm_mask * non_empty
                else:
                    actor_comm_mask = base_comm_mask

                actor_valid_count = actor_comm_mask.sum().clamp(min=1.0)

                if have_communication:

                    # Per-FIELD ratios, NOT one joint exp(sum) ratio:
                    # new/mb_old carry [B, N, 7] log-probs in the same
                    # field order. A joint ratio saturates the shared
                    # clip ~7x too readily (small drift in every field
                    # multiplies into a large joint drift); independent
                    # per-field ratios with the same PPO_CLIP keep each
                    # field's trust region honest. Aggregated with MEAN
                    # (not sum) so the term stays on the same scale as
                    # the single-action surrogate. NOTE: NONE-target
                    # rows contribute ratio exactly 1 (constant, zero
                    # gradient) on the target_id field -- a harmless
                    # offset only, no learning signal either way.
                    communication_ratio = torch.exp(
                        new_comm_log_prob_fields
                        - mb_old_comm_fields
                    )
                    comm_surrogate1 = (
                        communication_ratio
                        * mb_advantages.view(-1, 1, 1)
                    )

                    comm_surrogate2 = (
                        torch.clamp(
                            communication_ratio,
                            1.0 - PPO_CLIP,
                            1.0 + PPO_CLIP,
                        )
                        * mb_advantages.view(-1, 1, 1)
                    )

                    # ----------------------------------------------
                    # Per-(sender, receiver) credit weight.
                    #
                    # Without this, every sender a receiver heard
                    # from gets the SAME receiver-level advantage,
                    # so a positive outcome reinforces A->B, C->B,
                    # D->B identically even if only one of them
                    # actually helped. mb_trust_weights[b, sender] is
                    # receiver b's own current trust in that sender
                    # (already per-sender-per-receiver -- see
                    # communication/trust.py and the receiver-
                    # relevance fix in evaluator.py/train.py), reused
                    # here as a soft contribution weight on the
                    # advantage magnitude. It does NOT change
                    # actor_valid_count above, so low-trust senders
                    # are still trained on every valid row, just with
                    # a smaller gradient contribution rather than
                    # being excluded. Falls back to uniform (the
                    # previous behavior) if no trust signal is
                    # available this run.
                    # ----------------------------------------------

                    if mb_trust_weights is not None:

                        credit_weight = mb_trust_weights.to(
                            dtype=comm_surrogate1.dtype
                        ).clamp(0.0, 1.0)

                    else:

                        credit_weight = torch.ones_like(
                            comm_surrogate1[..., 0]
                        )

                    weighted_comm_mask = (
                        actor_comm_mask * credit_weight
                    )

                    communication_actor_loss = (
                        -(
                            torch.min(
                                comm_surrogate1,
                                comm_surrogate2,
                            ).mean(dim=-1)
                            * weighted_comm_mask
                        ).sum()
                        / actor_valid_count
                    )

                    # ------------------------------------------------
                    # Entropy. Deliberately uses the BASE mask (valid
                    # x not-self, INCLUDING empty messages), not
                    # actor_comm_mask and not weighted_comm_mask:
                    # exploration is what a currently-silent or
                    # low-trust sender needs in order to start
                    # communicating / earn trust, so excluding empties
                    # here (or down-weighting like the credit) would
                    # make an all-NONE policy unrecoverable.
                    # ------------------------------------------------

                    comm_entropy_mask = base_comm_mask.to(
                        dtype=communication_entropy.dtype
                    )

                    comm_entropy_valid_count = (
                        comm_entropy_mask.sum().clamp(min=1.0)
                    )

                    communication_entropy_mean = (
                        (communication_entropy * comm_entropy_mask).sum()
                        / comm_entropy_valid_count
                    )

                else:

                    communication_actor_loss = torch.zeros(
                        (), device=self.device, dtype=mb_advantages.dtype
                    )

                    communication_entropy_mean = torch.zeros(
                        (), device=self.device, dtype=mb_advantages.dtype
                    )

                # ==================================================
                # Clipped objective
                # ==================================================

                surrogate1 = (
                    ratio
                    * mb_advantages
                )

                surrogate2 = (
                    torch.clamp(
                        ratio,
                        1.0 - PPO_CLIP,
                        1.0 + PPO_CLIP,
                    )
                    * mb_advantages
                )

                # ==================================================
                # Actor loss
                # ==================================================

                actor_loss = (
                    -torch.min(
                        surrogate1,
                        surrogate2,
                    ).mean()
                )

                # ==================================================
                # Critic loss
                # ==================================================

                if VALUE_CLIP:

                    # `values` (current critic forward pass) and
                    # `normalized_returns` live in the value_norm
                    # NORMALIZED space. The buffer's stored old
                    # values are DENORMALIZED (select_action() calls
                    # value_norm.denormalize() before returning them
                    # for storage/logging), so they have to be
                    # renormalized before the clip range is
                    # meaningful -- otherwise PPO_CLIP is comparing
                    # deltas across two different scales. Crucially,
                    # the renormalization must use the PRE-UPDATE
                    # statistics active at rollout time (snapshotted
                    # as old_values_normalized before update() --
                    # see above), not the post-update statistics:
                    # re-normalizing mb_old_values here would compare
                    # current vs "old" values in different spaces.
                    if self.value_norm is not None:

                        mb_old_values_for_clip = (
                            old_values_normalized[idx]
                        )

                    else:

                        mb_old_values_for_clip = mb_old_values

                    value_clipped = (
                        mb_old_values_for_clip
                        + torch.clamp(
                            values - mb_old_values_for_clip,
                            -PPO_CLIP,
                            PPO_CLIP,
                        )
                    )

                    value_loss_unclipped = (
                        values - normalized_returns
                    ).pow(2)

                    value_loss_clipped = (
                        value_clipped - normalized_returns
                    ).pow(2)

                    critic_loss = (
                        0.5
                        * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    )

                else:

                    critic_loss = F.mse_loss(
                        values,
                        normalized_returns,
                    )

                # ==================================================
                # Entropy (separate action vs communication bonuses)
                #
                # The joint message log-prob sums 7 fields, so comm
                # entropy lives on a different scale from the single
                # env action -- sharing one coefficient lets one bonus
                # drown out the other. action_entropy_mean and
                # communication_entropy_mean are kept separate below;
                # entropy_loss (unweighted sum) is retained only for
                # logging continuity.
                # ==================================================

                action_entropy_mean = entropy.mean()

                entropy_loss = (
                    action_entropy_mean + communication_entropy_mean
                )

                # ==================================================
                # Statistics
                # ==================================================

                actor_loss_epoch += (
                    actor_loss.item()
                )

                critic_loss_epoch += (
                    critic_loss.item()
                )

                entropy_epoch += (
                    entropy_loss.item()
                )

                # ==================================================
                # Total loss
                # ==================================================

                total_loss = (
                    actor_loss
                    + COMM_POLICY_COEF * communication_actor_loss
                    + VALUE_LOSS_COEF * critic_loss
                    - ENTROPY_COEF * action_entropy_mean
                    - COMM_ENTROPY_COEF * communication_entropy_mean
                )

                # ==================================================
                # Zero gradients
                # ==================================================

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                # ==================================================
                # Backpropagation
                # ==================================================

                total_loss.backward()

                # ==================================================
                # Gradient clipping
                # ==================================================

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm,
                )

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm,
                )

                # ==================================================
                # Optimizer step
                # ==================================================

                self.actor_optimizer.step()

                self.critic_optimizer.step()

        # ======================================================
        # Average statistics
        # ======================================================

        num_updates = (
            UPDATE_EPOCHS
            * (
                (
                    dataset_size
                    + MINIBATCH_SIZE
                    - 1
                )
                // MINIBATCH_SIZE
            )
        )

        training_stats = {

            "actor_loss":
                actor_loss_epoch
                / num_updates,

            "critic_loss":
                critic_loss_epoch
                / num_updates,

            "entropy":
                entropy_epoch
                / num_updates,
        }

        return training_stats