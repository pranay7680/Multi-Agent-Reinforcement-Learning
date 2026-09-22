"""
network_attention.py -- DEPRECATED / NOT ACTIVE.

mappo.py imports MAPPOModel from gnn_attention.py, NOT from this file.
This module is kept only for reference/history and must not be imported
by training code. If you are tempted to switch back, note that the
authoritative communication-attention semantics is trust-as-attention-
bias ONLY (see gnn_attention.py::SharedActor._apply_received_communication
and the note below): trust must NOT scale values AND bias logits.

Neural networks for MAPPO with structured communication (legacy).

Architecture
------------

Shared Actor:

    Local Observation
            |
            v
    Entity Embeddings
            |
            v
    Self-Attention
            |
            v
      Local Hidden
       /         \
      /           \
     v             v
Policy Path    Communication Path
                  |
                  v
              Decoder
                  |
                  v
        Structured Message
                  |
                  v
               Encoder
                  |
                  v
        Communication Vector
                  |
                  v
        Other Blue Agents
                  |
                  v
        Trust-Weighted
        Communication
                  |
                  v
      Communication Attention
                  |
                  v
       Action Representation
                  |
                  v
            Action Logits


Central Critic:

    Global Observation
            |
            v
      Per-Agent Tokens
            |
            v
      Cross-Agent Attention
            |
            v
       Value per Agent


IMPORTANT
---------
The original CC4 observation still contains the native 8-bit message
block.

The new structured communication mechanism is separate from that
native message representation.

The old observation layout is preserved so the environment interface
does not change.

The new learned communication vector is generated from the actor's
internal hidden representation and passed between Blue agents by
MAPPO's rollout/training code.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    OBS_DIM,
    ACTION_DIM,
    EMBED_DIM,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_AGENTS,
)

from CybORG.Agents.Wrappers.BlueFlatWrapper import (
    NUM_SUBNETS,
    NUM_HQ_SUBNETS,
    MAX_HOSTS,
)

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
    NUM_MESSAGES,
    MESSAGE_LENGTH,
)

from .communication.structured_communication import (
    StructuredCommunication,
)


# ==========================================================
# Communication Configuration
# ==========================================================

# Final learned communication vector.
#
# This replaces the conceptual role of the old 8-bit message
# representation inside our neural communication mechanism.
COMMUNICATION_DIM = 128

# Latent size expected by MessageDecoder.
COMMUNICATION_LATENT_DIM = 256

# Hidden size used when attending over received messages.
COMMUNICATION_ATTENTION_DIM = EMBED_DIM


# ==========================================================
# Entity layout
# ==========================================================

MISSION_DIM = 1

# subnet one-hot
# + blocked-subnets mask
# + communication-policy mask
# + process alerts
# + connection alerts
SUBNET_BLOCK_DIM = (
    3 * NUM_SUBNETS
    + 2 * MAX_HOSTS
)

MESSAGE_DIM = NUM_MESSAGES * MESSAGE_LENGTH

NUM_ENTITY_TOKENS = (
    1
    + NUM_HQ_SUBNETS
)

_EXPECTED_OBS_DIM = (
    MISSION_DIM
    + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
    + MESSAGE_DIM
)

assert _EXPECTED_OBS_DIM == OBS_DIM, (
    f"Entity split ({_EXPECTED_OBS_DIM}) does not match "
    f"OBS_DIM ({OBS_DIM}) -- "
    "BlueFlatWrapper's per-subnet block size probably isn't "
    "a flat MAX_HOSTS per subnet at runtime. "
    "Print actual per-subnet host counts from "
    "BlueFlatWrapper.observation_change / self.hosts(agent_name) "
    "and adjust SUBNET_BLOCK_DIM."
)


# ==========================================================
# Utility
# ==========================================================

def build_mlp(
    input_dim: int,
    output_dim: int,
):
    """
    Build the standard MAPPO MLP.
    """

    layers = []

    current = input_dim

    for _ in range(NUM_HIDDEN_LAYERS):

        layers.append(
            nn.Linear(
                current,
                HIDDEN_DIM,
            )
        )

        layers.append(
            nn.ReLU()
        )

        current = HIDDEN_DIM

    layers.append(
        nn.Linear(
            current,
            output_dim,
        )
    )

    return nn.Sequential(*layers)


# ==========================================================
# Shared Actor
# ==========================================================

class SharedActor(nn.Module):
    """
    One policy shared by ALL Blue agents.

    Local observation:

        [mission]
        [subnet_0]
        [subnet_1]
        [subnet_2]
        [native CC4 messages]

    The observation is decomposed into semantic entities.

    The native 8-bit message block is retained in the input for
    backwards compatibility, but the new structured communication
    mechanism does NOT depend on it.

    Structured communication:

        local hidden
            |
            +--> MessageDecoder
            |
            +--> MessageEncoder
            |
            +--> communication vector

    Received communication:

        received vectors
              +
        trust weights
              |
              v
        communication attention
              |
              v
        communication context
              |
              v
        policy representation
    """

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        communication_dim: int = COMMUNICATION_DIM,
        communication_latent_dim: int = COMMUNICATION_LATENT_DIM,
        num_targets: Optional[int] = None,
    ):
        super().__init__()

        self.num_agents = num_agents

        self.communication_dim = communication_dim

        self.communication_latent_dim = (
            communication_latent_dim
        )

        # ------------------------------------------------------
        # Local observation embeddings
        # ------------------------------------------------------

        self.mission_embed = nn.Linear(
            MISSION_DIM,
            EMBED_DIM,
        )

        self.subnet_embed = nn.Linear(
            SUBNET_BLOCK_DIM,
            EMBED_DIM,
        )

        # Native CC4 message embedding.
        #
        # This remains only because OBS_DIM still contains the
        # original message block.
        #
        # It is NOT the new structured communication mechanism.
        # self.message_embed = nn.Linear(
        #     MESSAGE_DIM,
        #     EMBED_DIM,
        # )

        # ------------------------------------------------------
        # Positional embeddings
        # ------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                NUM_ENTITY_TOKENS,
                EMBED_DIM,
            )
        )

        nn.init.normal_(
            self.pos_embed,
            std=0.02,
        )

        # ------------------------------------------------------
        # Existing observation attention
        # ------------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            EMBED_DIM
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM * 4,
            ),
            nn.ReLU(),
            nn.Linear(
                EMBED_DIM * 4,
                EMBED_DIM,
            ),
        )

        self.norm2 = nn.LayerNorm(
            EMBED_DIM
        )

        # ------------------------------------------------------
        # Local representation
        # ------------------------------------------------------

        self.local_projection = nn.Sequential(
            nn.Linear(
                NUM_ENTITY_TOKENS * EMBED_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(
                communication_latent_dim
            ),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Structured communication module
        # ------------------------------------------------------

        self.communication = StructuredCommunication(
            input_dim=communication_latent_dim,
            message_dim=communication_dim,
            num_agents=num_agents,
            num_targets=num_targets,
        )

        # ------------------------------------------------------
        # Communication attention
        # ------------------------------------------------------

        self.communication_query = nn.Linear(
            communication_latent_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_key = nn.Linear(
            communication_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_value = nn.Linear(
            communication_dim,
            COMMUNICATION_ATTENTION_DIM,
        )

        self.communication_attention = nn.MultiheadAttention(
            embed_dim=COMMUNICATION_ATTENTION_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.communication_norm = nn.LayerNorm(
            COMMUNICATION_ATTENTION_DIM
        )

        # ------------------------------------------------------
        # Combine local representation with communication
        # ------------------------------------------------------

        self.policy_input_projection = nn.Sequential(
            nn.Linear(
                communication_latent_dim
                + COMMUNICATION_ATTENTION_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(
                communication_latent_dim
            ),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Policy head
        # ------------------------------------------------------

        self.policy_head = build_mlp(
            communication_latent_dim,
            ACTION_DIM,
        )

        # ------------------------------------------------------
        # Debug / visualization state
        # ------------------------------------------------------

        self.last_attention = None

        self.last_communication_attention = None

        self.last_outgoing_message = None

        self.last_local_hidden = None

    # ======================================================
    # Observation splitting
    # ======================================================

    def _split_entities(
        self,
        observation: torch.Tensor,
    ):
        """
        Split:

            [mission]
            [subnets]
            [native messages]

        from the flat CC4 observation.

        Returns
        -------

        mission:
            [B, 1]

        subnets:
            [B, NUM_HQ_SUBNETS, SUBNET_BLOCK_DIM]

        messages:
            [B, MESSAGE_DIM]
        """

        subnets_start = MISSION_DIM

        subnets_end = (
            MISSION_DIM
            + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
        )

        mission = observation[
            :,
            :MISSION_DIM,
        ]

        subnets = observation[
            :,
            subnets_start:subnets_end,
        ].view(
            -1,
            NUM_HQ_SUBNETS,
            SUBNET_BLOCK_DIM,
        )

        messages = observation[
            :,
            subnets_end:subnets_end + MESSAGE_DIM,
        ]

        return (
            mission,
            subnets,
            messages,
        )

    # ======================================================
    # Entity encoding
    # ======================================================

    def _encode_entities(
        self,
        observation: torch.Tensor,
    ):
        """
        Convert local observation into attended entity tokens.
        """

        (
            mission,
            subnets,
            messages,
        ) = self._split_entities(
            observation
        )

        # --------------------------------------------------
        # Mission token
        # --------------------------------------------------

        mission_tok = torch.relu(
            self.mission_embed(
                mission
            )
        ).unsqueeze(1)

        # --------------------------------------------------
        # Subnet tokens
        # --------------------------------------------------

        subnet_tok = torch.relu(
            self.subnet_embed(
                subnets
            )
        )

        # --------------------------------------------------
        # Native CC4 message token
        #
        # This is retained for observation compatibility.
        # It is NOT our new structured communication vector.
        # --------------------------------------------------

        # message_tok = torch.relu(
        #     self.message_embed(
        #         messages
        #     )
        # ).unsqueeze(1)

        # --------------------------------------------------
        # Build entity sequence
        # --------------------------------------------------

        # tokens = torch.cat(
        #     [
        #         mission_tok,
        #         subnet_tok,
        #         message_tok,
        #     ],
        #     dim=1,
        # )

        tokens = torch.cat(
            [
                mission_tok,
                subnet_tok,
            ],
            dim=1,
        )
        tokens = (
            tokens
            + self.pos_embed
        )

        # --------------------------------------------------
        # Self attention across entities
        # --------------------------------------------------

        attn_out, attention_weights = (
            self.attention(
                tokens,
                tokens,
                tokens,
            )
        )

        self.last_attention = (
            attention_weights.detach()
        )

        # --------------------------------------------------
        # Residual block
        # --------------------------------------------------

        x = self.norm1(
            tokens
            + attn_out
        )

        # --------------------------------------------------
        # Feed-forward block
        # --------------------------------------------------

        ff = self.ffn(x)

        # --------------------------------------------------
        # Second residual
        # --------------------------------------------------

        x = self.norm2(
            x + ff
        )

        return x

    # ======================================================
    # Local hidden representation
    # ======================================================

    def _get_local_hidden(
        self,
        observation: torch.Tensor,
    ):
        """
        Convert local observation into the latent representation
        used by both policy and communication.
        """

        x = self._encode_entities(
            observation
        )

        batch_size = x.shape[0]

        flat = x.reshape(
            batch_size,
            -1,
        )

        local_hidden = self.local_projection(
            flat
        )

        self.last_local_hidden = (
            local_hidden.detach()
        )

        return local_hidden

    # ======================================================
    # Communication generation
    # ======================================================

    def generate_communication(
        self,
        local_hidden: torch.Tensor,
    ):
        """
        Generate outgoing structured communication.

        Returns
        -------
        field_ids:
            Sampled message field IDs.

        log_probs:
            Log-probabilities of sampled message fields.

        entropies:
            Message-field entropies.

        communication_vector:
            Encoded communication vector.
        """

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.communication.generate_message(
            local_hidden
        )

        self.last_outgoing_message = (
            communication_vector.detach()
        )

        return (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        )

    # ======================================================
    # Received communication
    # ======================================================

    def _prepare_received_messages(
        self,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Normalize the different communication input formats.

        Supported:

        1. No communication:

            received_messages=None

        2. Messages without explicit trust:

            [B, N-1, D]

        3. Messages with all N agents:

            [B, N, D]

        4. Single-agent batch:

            [N-1, D]

        Trust weights:

            [B, N-1]

        or:

            [B, N]

        The training loop will eventually provide the messages
        from the previous timestep.
        """

        if received_messages is None:
            return None, None

        messages = received_messages

        if messages.dim() == 2:
            messages = messages.unsqueeze(0)

        if messages.dim() != 3:
            raise ValueError(
                "received_messages must have shape "
                "[B, N, D] or [B, N-1, D]."
            )

        messages = messages.to(
            device=device,
            dtype=dtype,
        )

        # --------------------------------------------------
        # Trust weights
        # --------------------------------------------------

        if trust_weights is not None:

            if trust_weights.dim() == 1:
                trust_weights = (
                    trust_weights.unsqueeze(0)
                )

            trust_weights = trust_weights.to(
                device=device,
                dtype=dtype,
            )

            if trust_weights.dim() != 2:
                raise ValueError(
                    "trust_weights must have shape "
                    "[B, N] or [B, N-1]."
                )

            if trust_weights.shape[0] == 1 and batch_size > 1:
                trust_weights = trust_weights.expand(
                    batch_size,
                    -1,
                )

        return (
            messages,
            trust_weights,
        )

    # ======================================================
    # Communication attention
    # ======================================================

    def _apply_received_communication(
        self,
        local_hidden: torch.Tensor,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor] = None,
    ):
        """
        Attend over received communication.

        local_hidden:
            [B, communication_latent_dim]

        received_messages:
            [B, NUM_SENDERS, communication_dim]

        trust_weights:
            [B, NUM_SENDERS]

        Returns:

            communication_context:
                [B, EMBED_DIM]
        """

        batch_size = (
            local_hidden.shape[0]
        )

        device = local_hidden.device

        dtype = local_hidden.dtype

        if received_messages is None:

            return torch.zeros(
                batch_size,
                COMMUNICATION_ATTENTION_DIM,
                device=device,
                dtype=dtype,
            )

        messages, trust = (
            self._prepare_received_messages(
                received_messages,
                trust_weights,
                batch_size,
                device,
                dtype,
            )
        )

        # --------------------------------------------------
        # Message keys and values
        # --------------------------------------------------

        keys = self.communication_key(
            messages
        )

        values = self.communication_value(
            messages
        )

        # --------------------------------------------------
        # Trust weighting
        #
        # Trust does NOT alter the message representation
        # itself. It controls how much the receiver should
        # attend to each sender.
        # --------------------------------------------------

        trust_bias = None

        if trust is not None:

            if trust.shape[1] != messages.shape[1]:
                raise ValueError(
                    "trust_weights and received_messages "
                    "must contain the same number of senders."
                )





# here , remove / comment below to disable or enable trust :-


            
            trust_safe = torch.clamp(
                trust,
                min=1e-4,
                max=1.0,
            )

            # NOTE (deprecated file): do NOT re-add values scaling here.
            # Trust is bias-only (see gnn_attention.py). The old
            # 'values * trust' line applied trust twice and is removed.

            # Actually feed the bias into attention (previously computed
            # but never used): log(trust) changes the mixture weights,
            # which is what survives the LayerNorm below.
            trust_bias = torch.log(trust_safe).unsqueeze(1).to(
                dtype=local_hidden.dtype
            )
            trust_bias = trust_bias.repeat_interleave(
                self.communication_attention.num_heads, dim=0
            )

        # --------------------------------------------------
        # Receiver query
        # --------------------------------------------------

        query = self.communication_query(
            local_hidden
        ).unsqueeze(1)

        # --------------------------------------------------
        # Attention over messages
        # --------------------------------------------------

        context, communication_weights = (
            self.communication_attention(
                query,
                keys,
                values,
                need_weights=True,
                attn_mask=trust_bias,
            )
        )

        self.last_communication_attention = (
            communication_weights.detach()
        )

        context = context.squeeze(1)

        context = self.communication_norm(
            context
        )

        return context

    # ======================================================
    # Main forward
    # ======================================================

    def forward(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
    ):
        """
        Forward pass through the actor.

        Parameters
        ----------
        observation:
            Local CC4 observation.

        received_messages:
            Structured communication vectors from other Blue
            agents.

            Expected:

                [B, N-1, COMMUNICATION_DIM]

            or:

                [B, N, COMMUNICATION_DIM]

        trust_weights:
            Trust values corresponding to received messages.

            Expected:

                [B, N-1]

            or:

                [B, N]

        return_communication:
            If False:

                returns action logits only.

            If True:

                returns:

                    logits,
                    communication_vector

        Notes
        -----
        The default behavior remains:

            actor(observation)

        so the existing MAPPO code continues to work while
        communication integration is being added.
        """

        squeeze_output = (
            observation.dim() == 1
        )

        if squeeze_output:
            observation = (
                observation.unsqueeze(0)
            )

        # --------------------------------------------------
        # Local observation processing
        # --------------------------------------------------

        local_hidden = (
            self._get_local_hidden(
                observation
            )
        )

        # --------------------------------------------------
        # Generate outgoing structured communication
        # --------------------------------------------------

        (
            field_ids,
            message_log_probs,
            message_entropies,
            outgoing_message,
        ) = self.generate_communication(
            local_hidden
        )

        # --------------------------------------------------
        # Process incoming communication
        # --------------------------------------------------

        communication_context = (
            self._apply_received_communication(
                local_hidden=local_hidden,
                received_messages=received_messages,
                trust_weights=trust_weights,
            )
        )

        # --------------------------------------------------
        # Combine local information and communication
        # --------------------------------------------------

        policy_representation = (
            torch.cat(
                [
                    local_hidden,
                    communication_context,
                ],
                dim=-1,
            )
        )

        policy_representation = (
            self.policy_input_projection(
                policy_representation
            )
        )

        # --------------------------------------------------
        # Action logits
        # --------------------------------------------------

        logits = self.policy_head(
            policy_representation
        )

        # --------------------------------------------------
        # Preserve old API
        # --------------------------------------------------

        if squeeze_output:

            logits = logits.squeeze(0)

            outgoing_message = (
                outgoing_message.squeeze(0)
            )

        if return_communication:
            return (
                logits,
                outgoing_message,
                field_ids,
                message_log_probs,
                message_entropies,
            )

        return logits

    # ======================================================
    # Communication-only forward
    # ======================================================

    def get_outgoing_message(
        self,
        observation: torch.Tensor,
    ):
        """
        Generate outgoing communication for rollout.
        """

        squeeze_output = (
            observation.dim() == 1
        )

        if squeeze_output:
            observation = observation.unsqueeze(0)

        local_hidden = self._get_local_hidden(
            observation
        )

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.generate_communication(
            local_hidden
        )

        if squeeze_output:
            communication_vector = (
                communication_vector.squeeze(0)
            )

            field_ids = {
                k: v.squeeze(0)
                for k, v in field_ids.items()
            }

            log_probs = {
                k: v.squeeze(0)
                for k, v in log_probs.items()
            }

            entropies = {
                k: v.squeeze(0)
                for k, v in entropies.items()
            }

        return (
            communication_vector,
            field_ids,
            log_probs,
            entropies,
        )

    # ======================================================
    # Full actor state
    # ======================================================

    def get_local_hidden(
        self,
        observation: torch.Tensor,
    ):
        """
        Expose the actor's local latent representation.

        Mainly useful for debugging and communication experiments.
        """

        squeeze_output = (
            observation.dim() == 1
        )

        if squeeze_output:
            observation = (
                observation.unsqueeze(0)
            )

        hidden = self._get_local_hidden(
            observation
        )

        if squeeze_output:
            hidden = hidden.squeeze(0)

        return hidden


# ==========================================================
# Central Critic
# ==========================================================

class CentralCritic(nn.Module):
    """
    Centralized critic.

    Receives:

        concat(
            obs0,
            obs1,
            obs2,
            obs3,
            obs4
        )

    and reshapes it into:

        [batch, NUM_AGENTS, OBS_DIM]

    Cross-agent self-attention allows the critic to model
    dependencies between Blue agents.

    Output:

        one value per agent
    """

    def __init__(self):

        super().__init__()

        self.agent_embed = nn.Linear(
            OBS_DIM,
            EMBED_DIM,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            EMBED_DIM
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                EMBED_DIM,
                EMBED_DIM * 4,
            ),
            nn.ReLU(),
            nn.Linear(
                EMBED_DIM * 4,
                EMBED_DIM,
            ),
        )

        self.norm2 = nn.LayerNorm(
            EMBED_DIM
        )

        self.value_head = build_mlp(
            EMBED_DIM,
            1,
        )

    def forward(
        self,
        global_state: torch.Tensor,
    ):

        squeeze_output = (
            global_state.dim() == 1
        )

        if squeeze_output:
            global_state = (
                global_state.unsqueeze(0)
            )

        batch_size = (
            global_state.shape[0]
        )

        # --------------------------------------------------
        # Global state -> per-agent tokens
        # --------------------------------------------------

        per_agent_obs = global_state.view(
            batch_size,
            NUM_AGENTS,
            OBS_DIM,
        )

        tokens = torch.relu(
            self.agent_embed(
                per_agent_obs
            )
        )

        # --------------------------------------------------
        # Cross-agent attention
        # --------------------------------------------------

        attn_out, attention_weights = (
            self.attention(
                tokens,
                tokens,
                tokens,
            )
        )

        self.last_attention = (
            attention_weights.detach()
        )

        # --------------------------------------------------
        # Residual block
        # --------------------------------------------------

        x = self.norm1(
            tokens
            + attn_out
        )

        # --------------------------------------------------
        # Feed-forward block
        # --------------------------------------------------

        ff = self.ffn(x)

        # --------------------------------------------------
        # Second residual
        # --------------------------------------------------

        x = self.norm2(
            x + ff
        )

        # --------------------------------------------------
        # Value prediction
        # --------------------------------------------------

        values = (
            self.value_head(x)
            .squeeze(-1)
        )

        if squeeze_output:
            values = values.squeeze(0)

        return values


# ==========================================================
# MAPPO Model
# ==========================================================

class MAPPOModel(nn.Module):
    """
    Complete MAPPO model.

    Contains:

        SharedActor
        CentralCritic
    """

    def __init__(
        self,
        num_targets: Optional[int] = None,
    ):

        super().__init__()

        self.actor = SharedActor(
            num_agents=NUM_AGENTS,
            communication_dim=COMMUNICATION_DIM,
            communication_latent_dim=COMMUNICATION_LATENT_DIM,
            num_targets=num_targets,
        )

        self.critic = CentralCritic()

    # ======================================================
    # Action
    # ======================================================

    def act(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
    ):
        """
        Actor forward pass.

        Backwards compatible:

            model.act(observation)

        Communication-enabled:

            model.act(
                observation,
                received_messages,
                trust_weights,
                return_communication=True,
            )
        """

        return self.actor(
            observation,
            received_messages=received_messages,
            trust_weights=trust_weights,
            return_communication=return_communication,
        )

    # ======================================================
    # Critic
    # ======================================================

    def evaluate(
        self,
        global_state: torch.Tensor,
    ):
        """
        Return critic value estimate per agent.

        Shape:

            [batch, NUM_AGENTS]
        """
        return self.critic(
            global_state
        )
