"""
gnn_attention.py

Neural networks for MAPPO with hierarchical GNN + attention and
structured communication.

Architecture
------------

Shared Actor:

    Local Observation
            |
            v
    Split into entities:
        mission            (1 node)
        subnet context     (NUM_HQ_SUBNETS nodes)
        per-host alerts     (NUM_HQ_SUBNETS * MAX_HOSTS nodes)
            |
            v
    Hierarchical Graph Message Passing
        host      <-> own subnet
        host      <-> other hosts, same subnet
        subnet    <-> subnet            (real CC4 topology, resolved
                                          per-sample from the
                                          observation -- see
                                          "Subnet graph construction"
                                          below)
        subnet    <-> mission
            |
            v
    Multi-Head Attention (over ALL graph nodes, padding masked out)
            |
            v
    Pool back down to [mission, subnet_0 .. subnet_N] tokens
    (host-level detail has already been propagated upward by
    the GNN, so we don't need to carry raw host tokens forward)
            |
            v
        Local Hidden
        /            \
       /              \
      v                v
 Policy Path      Communication Path
                        |
                        v
                    MessageDecoder
                        |
                        v
                 Structured Message
                        |
                        v
                    MessageEncoder
                        |
                        v
               Communication Vector
                        |
                        v
                Other Blue Agents
                        |
                        v
              Trust-Weighted Communication
                        |
                        v
              Communication Attention
                        |
                        v
               Action Representation
                        |
                        v
                   Action Logits


Central Critic (unchanged):

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


WHY THIS IS DIFFERENT FROM A "GNN" OVER MISSION+SUBNET NODES ONLY
-------------------------------------------------------------------
A GNN over ~4-10 fully-connected mission/subnet nodes with a fixed,
uniform adjacency is functionally close to a no-op sitting in front
of MultiheadAttention: attention already learns adaptive pairwise
weights over that same fully-connected node set, so a fixed uniform
message-passing layer on top of it adds parameters without adding
real inductive bias.

The observation already contains host-level signal
(`process_alerts`, `connection_alerts`, each length MAX_HOSTS,
packed into every subnet block) that the old code flattened into an
opaque per-subnet vector. This version promotes that host-level
signal into real graph nodes with a real (non-fully-connected)
topology: each host only connects to its own subnet and to other
hosts in that same subnet, and subnets connect to mission. That
gives the GNN actual structure to propagate over before attention
re-summarizes it -- the two modules are now doing different jobs
instead of the same job twice.

Subnet graph construction
--------------------------
Every agent's observation has exactly NUM_HQ_SUBNETS (=3) subnet
slots, but only some of those slots hold a real subnet:

    Agent 4 (HQ)     -> 3 real slots: admin, office, public_access
    Agents 0-3       -> 1 real slot (their assigned subnet),
                         the other 2 slots are padding

BlueFlatWrapper writes a NUM_SUBNETS-length one-hot into the start
of every subnet slot's context block identifying *which* real
subnet (if any) occupies that slot; an all-zero one-hot means the
slot is padding. Because this identity is only knowable from the
observation itself -- and because the shared policy sees agents
with different real/padding layouts in the same batch -- the
subnet<->subnet edges (and which nodes are masked out as padding)
are resolved **per sample, at forward time**, in
`SharedActor._build_batch_adjacency`, rather than being baked into
a single fixed adjacency buffer at __init__ time. The physical CC4
subnet-to-subnet topology (which slots are allowed to connect to
which *if both are real*) is still fixed and is precomputed once
into `real_topology`; only the resolution of "which real subnet is
in slot i for this sample" and "is slot i even real" is dynamic.

Padding is kept harmless in two places:
  1. GNN adjacency: every edge touching a padding node (mission<->
     padding, padding<->padding subnet edges, host<->padding) is
     zeroed before row-normalization, so padding nodes neither send
     nor receive messages.
  2. Attention: padding nodes are passed to `nn.MultiheadAttention`
     via `key_padding_mask` so they can't be attended to either.

If you want to override the physical topology (e.g. for an ablation
or a different scenario), pass `subnet_edges` to `SharedActor` /
`MAPPOModel` as a list of (i, j) index pairs into
`SUBNET_NAME_ORDER` -- the same interface as before. `None` now
means "no subnet-subnet edges at all" (fully isolated subnets)
rather than the old fully-connected fallback, since a fully-connected
default silently misrepresents the real topology.

Host-level padding
--------------------
Every active subnet slot always has MAX_HOSTS (=16) host nodes, but
CC4 randomizes 1-6 servers / 3-10 users per zone at reset, so most
episodes have fewer than MAX_HOSTS real hosts in a given subnet --
the remaining slots are padding, exactly like the padding subnet
slots above. Unlike subnet identity, though, this is NOT recoverable
from the observation vector: BlueFlatWrapper ANDs "host exists" and
"host currently has an alert" into a single bit before it's written,
so an absent host and a present-but-quiet host are bit-for-bit
identical (both read 0) at any given timestep. `SharedActor` accepts
an optional `host_active_mask` ([B, NUM_HQ_SUBNETS, MAX_HOSTS] bool)
through `forward`/`get_local_hidden`/`get_outgoing_message`/`act` for
exactly this -- when supplied, per-host padding is isolated the same
way per-subnet padding is (adjacency edges zeroed, node_active
excludes it from attention and from `local_projection`'s input). It
defaults to None, which treats every host slot in an active subnet
as real -- the same assumption every call site already made before
this parameter existed. That default is NOT a fix by itself: closing
this gap for real requires a caller to source the true per-episode
host count from the environment (fixed at reset, not observation-
derivable) and pass it in. That plumbing lives outside this file.

IMPORTANT
---------
The model's PRE-EXISTING interface (observation / received_messages
/ trust_weights / return_communication) and all output shapes are
unchanged. `host_active_mask` above is a genuinely NEW, optional
keyword argument, added specifically to make per-host padding
isolation possible -- every existing call site keeps working
unchanged (it defaults to None), but actually protecting against
fake host nodes requires updating the caller to supply it, sourced
from the environment. That is the one exception to the "no changes
required elsewhere" claim below; everything else still holds:

No other changes are required in:

    mappo.py
    buffer.py
    train.py
    env.py
    encoder.py
    decoder.py
    schema.py
    structured_communication.py

`local_projection`'s input dimension is still
NUM_ENTITY_TOKENS * EMBED_DIM where NUM_ENTITY_TOKENS = 1 +
NUM_HQ_SUBNETS, exactly as in the pre-GNN version, so
COMMUNICATION_LATENT_DIM and everything downstream of it (decoder,
encoder, trust, evaluator) is untouched.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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

COMMUNICATION_DIM = 128

COMMUNICATION_LATENT_DIM = 256

COMMUNICATION_ATTENTION_DIM = EMBED_DIM


# ==========================================================
# Entity layout
#
# Each subnet block (SUBNET_BLOCK_DIM) is, in order:
#
#   [ subnet one-hot          | NUM_SUBNETS ]
#   [ blocked-subnets mask    | NUM_SUBNETS ]
#   [ communication-policy    | NUM_SUBNETS ]
#   [ process alerts          | MAX_HOSTS   ]
#   [ connection alerts       | MAX_HOSTS   ]
#
# The first three chunks are context shared by the whole subnet.
# The last two chunks are genuinely per-host and are what we turn
# into host graph nodes below. The subnet one-hot is also what
# tells us, per sample, which real subnet (if any) occupies this
# slot -- see "Subnet graph construction" in the module docstring.
# ==========================================================

MISSION_DIM = 1

SUBNET_CONTEXT_DIM = 3 * NUM_SUBNETS

HOST_FEATURE_DIM = 2  # [process_alert_i, connection_alert_i]

SUBNET_BLOCK_DIM = (
    SUBNET_CONTEXT_DIM
    + 2 * MAX_HOSTS
)

MESSAGE_DIM = NUM_MESSAGES * MESSAGE_LENGTH

# Tokens kept AFTER pooling, fed into local_projection.
# Identical to the pre-GNN version -- this is what keeps every
# downstream module's input shape unchanged.
NUM_ENTITY_TOKENS = (
    1
    + NUM_HQ_SUBNETS
)

# Full graph size used DURING message passing / attention, before
# pooling back down to NUM_ENTITY_TOKENS.
NUM_HOST_NODES = NUM_HQ_SUBNETS * MAX_HOSTS

NUM_GRAPH_NODES = (
    NUM_ENTITY_TOKENS
    + NUM_HOST_NODES
)


_EXPECTED_OBS_DIM = (
    MISSION_DIM
    + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
    + MESSAGE_DIM
)

assert _EXPECTED_OBS_DIM == OBS_DIM, (
    f"Entity split ({_EXPECTED_OBS_DIM}) does not match "
    f"OBS_DIM ({OBS_DIM}) -- "
    "BlueFlatWrapper's per-subnet block size probably "
    "doesn't match the configured observation dimensions."
)


# ==========================================================
# Canonical subnet identity table
# ==========================================================
#
# BlueFlatWrapper builds each subnet slot's one-hot from
# `sorted(state.subnet_name_to_cidr.items())` -- alphabetical order
# over the 9 CC4 subnet names. This is the same order used by
# Marl/gnn/wrapper/globals.py::ROUTERS (minus the "_router" suffix),
# which is where these names are sourced from. If the scenario's
# subnet names ever change, update this list to match.
SUBNET_NAME_ORDER: List[str] = [
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
]

assert len(SUBNET_NAME_ORDER) == NUM_SUBNETS, (
    f"SUBNET_NAME_ORDER has {len(SUBNET_NAME_ORDER)} entries but "
    f"NUM_SUBNETS is {NUM_SUBNETS} -- BlueFlatWrapper's subnet "
    "one-hot layout has changed; update this table to match."
)

_SUBNET_NAME_TO_IDX = {name: i for i, name in enumerate(SUBNET_NAME_ORDER)}

# Real, physical CC4 subnet-to-subnet topology (independent of which
# agent is observing). Cross-checked against both BlueFlatWrapper's
# `_build_comms_policy_network` and Marl/gnn/wrapper/globals.py's
# `ACCESSABLE_OFFLINE`:
#
#          Public_Access
#           /         \
#          /           \
#       Admin -------- Office
#
#   Restricted_A <-> Operational_A
#   Restricted_B <-> Operational_B
REAL_SUBNET_TOPOLOGY_EDGES: List[Tuple[str, str]] = [
    ("restricted_zone_a_subnet", "operational_zone_a_subnet"),
    ("restricted_zone_b_subnet", "operational_zone_b_subnet"),
    ("public_access_zone_subnet", "admin_network_subnet"),
    ("public_access_zone_subnet", "office_network_subnet"),
    ("admin_network_subnet", "office_network_subnet"),
]

# Default `subnet_edges` for SharedActor / MAPPOModel: the real
# topology above, expressed as index pairs into SUBNET_NAME_ORDER
# (i.e. into the universe of 9 real subnets -- NOT into an agent's
# 3 padded observation slots, which are resolved per-sample from
# the observation itself; see `SharedActor._build_batch_adjacency`).
SUBNET_EDGES: List[Tuple[int, int]] = [
    (_SUBNET_NAME_TO_IDX[a], _SUBNET_NAME_TO_IDX[b])
    for a, b in REAL_SUBNET_TOPOLOGY_EDGES
]


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

        layers.append(nn.Linear(current, HIDDEN_DIM))
        layers.append(nn.ReLU())

        current = HIDDEN_DIM

    layers.append(nn.Linear(current, output_dim))

    return nn.Sequential(*layers)


# ==========================================================
# Hierarchical adjacency
# ==========================================================

def _build_structural_adjacency(
    num_subnets: int,
    max_hosts: int,
) -> torch.Tensor:
    """
    Build the FIXED (data-independent) part of the hierarchical
    graph over:

        node 0                        = mission
        node 1 .. num_subnets         = subnet slots
        node (1+num_subnets) ..       = hosts, grouped by slot

    Edges:

        mission <-> every subnet slot
        subnet slot  <-> its own hosts
        host    <-> other hosts in the SAME slot only

    Subnet-slot <-> subnet-slot edges are intentionally NOT built
    here. Which slots may be connected depends on which real
    subnets occupy them for a given observation (and whether a slot
    is even real, or padding) -- that is agent- and sample-
    dependent, so it's resolved per-sample at forward time in
    `SharedActor._build_batch_adjacency` instead of being baked
    into this static buffer.

    Not row-normalized: normalization happens per-sample, after
    the dynamic subnet-subnet edges are added and padding nodes are
    masked out, so that a padding node's zeroed row doesn't get
    divided by a nonzero degree computed before masking.
    """

    num_nodes = 1 + num_subnets + num_subnets * max_hosts

    adjacency = torch.zeros(num_nodes, num_nodes)

    mission_idx = 0
    subnet_start = 1
    host_start = 1 + num_subnets

    def host_idx(subnet_i: int, host_j: int) -> int:
        return host_start + subnet_i * max_hosts + host_j

    # mission <-> subnet slots
    for s in range(num_subnets):
        si = subnet_start + s
        adjacency[mission_idx, si] = 1.0
        adjacency[si, mission_idx] = 1.0

    # subnet slot <-> own hosts, host <-> host (same slot)
    for s in range(num_subnets):
        si = subnet_start + s
        for h in range(max_hosts):
            hi = host_idx(s, h)
            adjacency[si, hi] = 1.0
            adjacency[hi, si] = 1.0
            for h2 in range(max_hosts):
                if h2 != h:
                    hi2 = host_idx(s, h2)
                    adjacency[hi, hi2] = 1.0

    return adjacency


def _build_real_topology_matrix(
    subnet_edges: Optional[List[Tuple[int, int]]],
    num_subnets: int,
) -> torch.Tensor:
    """
    Build the [num_subnets, num_subnets] adjacency over the
    universe of *real* CC4 subnets (indexed by SUBNET_NAME_ORDER),
    independent of any agent's observation layout.

    `subnet_edges=None` means no subnet-subnet edges at all
    (fully isolated subnets) -- NOT a fully-connected fallback.
    A fully-connected default would silently misrepresent the real
    topology, which is exactly the bug this module fixes.
    """

    matrix = torch.zeros(num_subnets, num_subnets)

    if subnet_edges is None:
        return matrix

    for i, j in subnet_edges:
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0

    return matrix


# ==========================================================
# Graph Neural Network Layer
# ==========================================================

class GraphMessagePassing(nn.Module):
    """
    Gated graph message-passing layer over a (possibly per-sample)
    adjacency.

    Input:  x          [B, N, D]
            adjacency   [B, N, N] or [N, N] -- already row-
                        normalized, and with any padding-node edges
                        already zeroed out by the caller.
    Output: [B, N, D]
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.dim = dim

        self.self_linear = nn.Linear(dim, dim)
        self.neighbor_linear = nn.Linear(dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:

        # torch.matmul broadcasts a [N, N] adjacency across the
        # batch, or applies a [B, N, N] per-sample adjacency
        # directly -- both are supported.
        neighbor_info = torch.matmul(adjacency, x)

        self_info = self.self_linear(x)
        neighbor_info = self.neighbor_linear(neighbor_info)

        gate_input = torch.cat([self_info, neighbor_info], dim=-1)
        gate = self.gate(gate_input)

        updated = self_info + gate * neighbor_info
        updated = self.norm(x + updated)

        return updated


# ==========================================================
# Shared Actor
# ==========================================================

class SharedActor(nn.Module):
    """
    One policy shared by ALL Blue agents.
    """

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        communication_dim: int = COMMUNICATION_DIM,
        communication_latent_dim: int = COMMUNICATION_LATENT_DIM,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
        use_host_graph: bool = True,
        subnet_edges: Optional[List[Tuple[int, int]]] = SUBNET_EDGES,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.communication_dim = communication_dim
        self.communication_latent_dim = communication_latent_dim
        self.use_host_graph = use_host_graph and MAX_HOSTS > 0

        # ------------------------------------------------------
        # Local observation embeddings
        # ------------------------------------------------------

        self.mission_embed = nn.Linear(MISSION_DIM, EMBED_DIM)
        self.subnet_embed = nn.Linear(SUBNET_CONTEXT_DIM, EMBED_DIM)

        if self.use_host_graph:
            self.host_embed = nn.Linear(HOST_FEATURE_DIM, EMBED_DIM)
            num_graph_nodes = NUM_GRAPH_NODES
        else:
            self.host_embed = None
            num_graph_nodes = NUM_ENTITY_TOKENS

        self.num_graph_nodes = num_graph_nodes

        # ------------------------------------------------------
        # Positional embeddings (over the FULL graph, pre-pooling)
        # ------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_graph_nodes, EMBED_DIM)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # ------------------------------------------------------
        # GNN
        #
        # `structural_adjacency` is the fixed mission/subnet/host
        # skeleton (no subnet-subnet edges). `real_topology` is the
        # fixed physical CC4 subnet graph over the universe of 9
        # real subnets. Neither depends on which agent produced a
        # given observation -- they're combined with per-sample
        # subnet identity/padding info at forward time in
        # `_build_batch_adjacency`.
        # ------------------------------------------------------

        if self.use_host_graph:
            self.register_buffer(
                "structural_adjacency",
                _build_structural_adjacency(
                    num_subnets=NUM_HQ_SUBNETS,
                    max_hosts=MAX_HOSTS,
                ),
            )
            self.register_buffer(
                "real_topology",
                _build_real_topology_matrix(
                    subnet_edges=subnet_edges,
                    num_subnets=NUM_SUBNETS,
                ),
            )
            self.gnn1 = GraphMessagePassing(EMBED_DIM)
            self.gnn2 = GraphMessagePassing(EMBED_DIM)
        else:
            self.gnn1 = None
            self.gnn2 = None

        # ------------------------------------------------------
        # Multi-Head Attention (over the full graph)
        # ------------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        # ------------------------------------------------------
        # Local representation
        #
        # Pooled back down to NUM_ENTITY_TOKENS (mission + subnets)
        # -- same shape as the pre-GNN version.
        # ------------------------------------------------------

        self.local_projection = nn.Sequential(
            nn.Linear(
                NUM_ENTITY_TOKENS * EMBED_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(communication_latent_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Structured communication module
        #
        # num_targets -> num_host_targets/num_subnet_targets: target_id
        # is one field (schema.py's MESSAGE_FIELDS is unchanged), but
        # its vocabulary depends on target_type -- HOST and SUBNET are
        # separate, independently-sized vocabularies now (see
        # communication/encoder.py and communication/decoder.py's dual
        # host/subnet embedding tables and heads). Passed straight
        # through; StructuredCommunication is expected to forward these
        # to MessageEncoder.build_target_embeddings(...) /
        # MessageDecoder.build_target_heads(...) using the same two
        # names.
        # ------------------------------------------------------

        self.communication = StructuredCommunication(
            input_dim=communication_latent_dim,
            message_dim=communication_dim,
            num_agents=num_agents,
            num_host_targets=num_host_targets,
            num_subnet_targets=num_subnet_targets,
        )

        # ------------------------------------------------------
        # Communication attention (unchanged)
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
        # Policy fusion (unchanged)
        # ------------------------------------------------------

        self.policy_input_projection = nn.Sequential(
            nn.Linear(
                communication_latent_dim + COMMUNICATION_ATTENTION_DIM,
                communication_latent_dim,
            ),
            nn.LayerNorm(communication_latent_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------
        # Policy head (unchanged)
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
        self.last_gnn_output = None

    # ======================================================
    # Observation splitting
    # ======================================================

    def _split_entities(self, observation: torch.Tensor):
        """
        Split [mission | subnets | native messages] from the
        flat CC4 observation. Unchanged from the pre-GNN version.
        """

        subnets_start = MISSION_DIM
        subnets_end = (
            MISSION_DIM
            + NUM_HQ_SUBNETS * SUBNET_BLOCK_DIM
        )

        mission = observation[:, :MISSION_DIM]

        subnets = observation[
            :, subnets_start:subnets_end
        ].view(-1, NUM_HQ_SUBNETS, SUBNET_BLOCK_DIM)

        messages = observation[
            :, subnets_end:subnets_end + MESSAGE_DIM
        ]

        return mission, subnets, messages

    # ======================================================
    # Per-sample subnet graph construction
    # ======================================================

    def _build_batch_adjacency(
        self,
        subnet_one_hot: torch.Tensor,
        host_active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the fixed mission/subnet/host skeleton with
        per-sample subnet-subnet edges drawn from the real CC4
        topology, then isolate padding subnet slots (and their host
        blocks) so they neither send nor receive GNN messages.

        Args:
            subnet_one_hot: [B, NUM_HQ_SUBNETS, NUM_SUBNETS] -- the
                "subnet one-hot" sub-vector BlueFlatWrapper writes
                at the start of every subnet slot's context block.
                A slot whose one-hot sums to 0 is padding (no real
                subnet occupies it in this observation).

            host_active_mask: optional [B, NUM_HQ_SUBNETS, MAX_HOSTS]
                bool. True = this host slot is a real host THIS
                EPISODE, within a subnet slot that is itself real
                (ignored for slots `subnet_one_hot` already marks as
                padding). CC4 randomizes 1-6 servers / 3-10 users per
                zone at reset, fixed for the episode, but that count
                is NOT recoverable from the observation vector: per
                BlueFlatWrapper.observation_change,
                `process_subvector`/`connection_subvector` are built
                as `h in state.hosts and has_alert(h)` -- host
                existence and "has an active alert" are ANDed into a
                single bit before the flat vector is built, so an
                absent host and a present-but-currently-quiet host
                are bit-for-bit identical (both read 0) at any single
                timestep. This mask therefore cannot be derived
                in here; it must come from the caller, sourced from
                the environment's per-episode host layout. If None
                (the default -- every existing call site still omits
                it), every host slot in an active subnet is treated
                as real, i.e. the behavior before this parameter
                existed. That is a real gap, not a fix, until a
                caller supplies the true mask -- but it is a
                deliberately inert default: it never fabricates a
                presence signal from the alert bits, which would
                silently mask out (delete) real, currently-quiet
                hosts rather than genuine padding.

        Returns:
            adjacency:   [B, N, N] row-normalized adjacency, ready
                         for GraphMessagePassing.
            node_active: [B, N] bool -- True for real (non-padding)
                         nodes. Mission is always True. Also usable
                         directly as the inverse of an
                         `nn.MultiheadAttention` key_padding_mask.
        """

        batch_size, num_slots, _ = subnet_one_hot.shape
        device = subnet_one_hot.device

        if host_active_mask is not None:
            expected_shape = (batch_size, num_slots, MAX_HOSTS)
            if tuple(host_active_mask.shape) != expected_shape:
                raise ValueError(
                    f"host_active_mask must have shape {expected_shape} "
                    f"(B, NUM_HQ_SUBNETS, MAX_HOSTS), got "
                    f"{tuple(host_active_mask.shape)}."
                )
            host_active_mask = host_active_mask.to(
                device=device, dtype=torch.bool
            )

        # Which slots hold a real subnet this step, and which real
        # subnet (by index into SUBNET_NAME_ORDER) each active slot
        # holds. For padding slots slot_id is meaningless (masked
        # out below), so any value is fine there.
        slot_active = subnet_one_hot.sum(dim=-1) > 0.5           # [B, S]
        slot_id = subnet_one_hot.argmax(dim=-1)                  # [B, S]

        # Gather the real-world subnet-subnet topology restricted
        # to whichever real subnets occupy these slots this step:
        #   subnet_subnet[b, i, j] = real_topology[slot_id[b,i], slot_id[b,j]]
        topology = self.real_topology.to(device)
        rows = topology[slot_id]                                  # [B, S, NUM_SUBNETS]
        gather_index = slot_id.unsqueeze(1).expand(-1, num_slots, -1)
        subnet_subnet = torch.gather(rows, dim=2, index=gather_index)

        # An edge only exists if BOTH slots are real for this sample.
        slot_pair_active = slot_active.unsqueeze(2) & slot_active.unsqueeze(1)
        subnet_subnet = subnet_subnet * slot_pair_active.to(subnet_subnet.dtype)

        subnet_start = 1
        subnet_end = subnet_start + num_slots
        host_start = subnet_end

        adjacency = self.structural_adjacency.to(device)
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1).clone()
        adjacency[:, subnet_start:subnet_end, subnet_start:subnet_end] = subnet_subnet

        # Full-graph active-node mask: mission is always real; a
        # subnet slot and its entire host block share the same
        # active/padding status.
        node_active = torch.ones(
            batch_size, adjacency.shape[1], dtype=torch.bool, device=device
        )
        node_active[:, subnet_start:subnet_end] = slot_active
        for s in range(num_slots):
            h0 = host_start + s * MAX_HOSTS
            h1 = h0 + MAX_HOSTS

            if host_active_mask is not None:
                # A host node is only active if its subnet slot is
                # real AND this specific host slot is a real host
                # within it -- the fake/padded-host fix.
                node_active[:, h0:h1] = (
                    slot_active[:, s : s + 1] & host_active_mask[:, s, :]
                )
            else:
                node_active[:, h0:h1] = slot_active[:, s : s + 1]

        # Zero every edge touching a padding node -- no mission<->
        # padding, no padding<->padding subnet edges, no host<->
        # padding (whether the whole subnet slot is padding, or just
        # this specific host slot within a real subnet), in either
        # direction.
        edge_active = node_active.unsqueeze(2) & node_active.unsqueeze(1)
        adjacency = adjacency * edge_active.to(adjacency.dtype)

        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adjacency = adjacency / degree

        return adjacency, node_active

    # ======================================================
    # Padding isolation
    # ======================================================

    @staticmethod
    def _mask_inactive_nodes(
        x: torch.Tensor,
        node_active: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out every padding node's token in-place-equivalent
        (returns a new tensor).

        `key_padding_mask` on `nn.MultiheadAttention` only stops
        padding nodes from being attended to as keys/values -- it
        does NOT stop them from acting as queries. A padding node's
        query still attends over the real nodes' values, so its
        post-attention token absorbs real information. Since padding
        subnet tokens sit inside `entity_tokens` (pooled straight
        into `local_projection`), that leakage would otherwise reach
        the policy/communication path. Explicitly re-zeroing padding
        nodes after every stage that could introduce such leakage
        (GNN layers, and the attention + FFN block) guarantees they
        stay inert regardless of what happens inside those stages.

        Args:
            x:           [B, N, D]
            node_active: [B, N] bool -- True for real (non-padding)
                         nodes, as returned by
                         `_build_batch_adjacency`.
        """

        return x * node_active.unsqueeze(-1).to(dtype=x.dtype)

    # ======================================================
    # Entity encoding
    # ======================================================

    def _encode_entities(
        self,
        observation: torch.Tensor,
        host_active_mask: Optional[torch.Tensor] = None,
    ):
        """
        Flat observation -> hierarchical graph -> GNN -> attention
        -> pooled entity tokens (mission + subnets).

        host_active_mask: see `_build_batch_adjacency` -- optional
        per-host validity, defaults to "every host slot in an active
        subnet is real" when not supplied. Unused in the
        `not self.use_host_graph` branch (no host nodes exist there).
        """

        mission, subnets, _messages = self._split_entities(
            observation
        )

        batch_size = observation.shape[0]

        # Every branch below needs to know which of the
        # NUM_HQ_SUBNETS slots are real vs. padding for this batch.
        subnet_one_hot = subnets[..., :NUM_SUBNETS]
        slot_active = subnet_one_hot.sum(dim=-1) > 0.5

        # --------------------------------------------------
        # Mission node
        # --------------------------------------------------

        mission_tok = torch.relu(
            self.mission_embed(mission)
        ).unsqueeze(1)

        # --------------------------------------------------
        # Subnet context nodes (one-hot + blocked + comms mask
        # only -- host alerts are split off separately below)
        # --------------------------------------------------

        subnet_context = subnets[..., :SUBNET_CONTEXT_DIM]

        subnet_tok = torch.relu(
            self.subnet_embed(subnet_context)
        )

        if not self.use_host_graph:
            node_active = torch.cat(
                [
                    torch.ones(
                        batch_size, MISSION_DIM, dtype=torch.bool,
                        device=slot_active.device,
                    ),
                    slot_active,
                ],
                dim=1,
            )

            tokens = torch.cat([mission_tok, subnet_tok], dim=1)
            tokens = tokens + self.pos_embed

            attn_out, attention_weights = self.attention(
                tokens, tokens, tokens,
                key_padding_mask=~node_active,
            )
            self.last_attention = attention_weights.detach()

            x = self.norm1(tokens + attn_out)
            x = self.norm2(x + self.ffn(x))

            # See `_mask_inactive_nodes`: key_padding_mask alone
            # doesn't stop a padding query from absorbing real
            # values, so explicitly zero padding tokens before they
            # reach local_projection.
            x = self._mask_inactive_nodes(x, node_active)

            return x

        # --------------------------------------------------
        # Host nodes: real per-host alert signal, previously
        # flattened into an opaque per-subnet vector.
        # --------------------------------------------------

        process_alerts = subnets[
            ..., SUBNET_CONTEXT_DIM:SUBNET_CONTEXT_DIM + MAX_HOSTS
        ]
        connection_alerts = subnets[
            ...,
            SUBNET_CONTEXT_DIM + MAX_HOSTS:
            SUBNET_CONTEXT_DIM + 2 * MAX_HOSTS,
        ]

        host_features = torch.stack(
            [process_alerts, connection_alerts], dim=-1
        )  # [B, NUM_HQ_SUBNETS, MAX_HOSTS, 2]

        host_features = host_features.reshape(
            batch_size, NUM_HOST_NODES, HOST_FEATURE_DIM
        )

        host_tok = torch.relu(
            self.host_embed(host_features)
        )

        # --------------------------------------------------
        # Full graph: [mission | subnets | hosts]
        # --------------------------------------------------

        tokens = torch.cat(
            [mission_tok, subnet_tok, host_tok], dim=1
        )
        tokens = tokens + self.pos_embed

        # --------------------------------------------------
        # Per-sample subnet graph: real topology + padding isolation
        # --------------------------------------------------

        adjacency, node_active = self._build_batch_adjacency(
            subnet_one_hot, host_active_mask=host_active_mask
        )

        # --------------------------------------------------
        # Hierarchical GNN message passing
        #
        # The masked adjacency already stops padding nodes from
        # receiving real neighbor messages, but each layer's
        # self_linear(x) term still runs on whatever a padding
        # node's own token happens to be (embedding-layer bias,
        # etc). Re-zeroing after every layer guarantees padding
        # nodes carry no information forward regardless of that,
        # rather than relying solely on the adjacency mask.
        # --------------------------------------------------

        tokens = self.gnn1(tokens, adjacency)
        tokens = F.gelu(tokens)
        tokens = self._mask_inactive_nodes(tokens, node_active)

        tokens = self.gnn2(tokens, adjacency)
        tokens = F.gelu(tokens)
        tokens = self._mask_inactive_nodes(tokens, node_active)

        self.last_gnn_output = tokens.detach()

        # --------------------------------------------------
        # Attention over the full graph (padding nodes excluded)
        # --------------------------------------------------

        attn_out, attention_weights = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=~node_active,
        )
        self.last_attention = attention_weights.detach()

        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))

        # `key_padding_mask` only stops padding nodes from being
        # attended to as keys/values -- a padding node's own query
        # still attends over the real nodes' values, so its
        # post-attention/FFN token can absorb real information.
        # Since entity_tokens (below) is pooled straight into
        # local_projection, re-zero padding nodes here so that
        # leak can't reach the policy/communication path.
        x = self._mask_inactive_nodes(x, node_active)

        # --------------------------------------------------
        # Pool back down to mission + subnet tokens. Host-level
        # detail has already been propagated into these via the
        # GNN + attention above, so we don't carry raw host
        # tokens into local_projection.
        # --------------------------------------------------

        entity_tokens = x[:, :NUM_ENTITY_TOKENS, :]

        return entity_tokens

    # ======================================================
    # Local hidden representation
    # ======================================================

    def _get_local_hidden(
        self,
        observation: torch.Tensor,
        host_active_mask: Optional[torch.Tensor] = None,
    ):

        x = self._encode_entities(
            observation, host_active_mask=host_active_mask
        )

        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)

        local_hidden = self.local_projection(flat)

        self.last_local_hidden = local_hidden.detach()

        return local_hidden

    # ======================================================
    # Communication generation (unchanged)
    # ======================================================

    def generate_communication(self, local_hidden: torch.Tensor):

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.communication.generate_message(local_hidden)

        self.last_outgoing_message = communication_vector.detach()

        return field_ids, log_probs, entropies, communication_vector

    # ======================================================
    # Received communication (unchanged)
    # ======================================================

    def _prepare_received_messages(
        self,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):

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

        messages = messages.to(device=device, dtype=dtype)

        if trust_weights is not None:

            if trust_weights.dim() == 1:
                trust_weights = trust_weights.unsqueeze(0)

            trust_weights = trust_weights.to(device=device, dtype=dtype)

            if trust_weights.dim() != 2:
                raise ValueError(
                    "trust_weights must have shape "
                    "[B, N] or [B, N-1]."
                )

            if trust_weights.shape[0] == 1 and batch_size > 1:
                trust_weights = trust_weights.expand(batch_size, -1)

        return messages, trust_weights

    # ======================================================
    # Communication attention (unchanged)
    # ======================================================

    def _apply_received_communication(
        self,
        local_hidden: torch.Tensor,
        received_messages: Optional[torch.Tensor],
        trust_weights: Optional[torch.Tensor] = None,
        communication_valid: Optional[torch.Tensor] = None,
    ):
        """
        Attend over received communication.

        communication_valid (optional [B] bool): False marks episode-start
        rows with no previous message. Invalid rows return an exact zero
        context -- identical to the ``received_messages is None`` rollout
        path -- regardless of what (zeroed) message/trust tensors the PPO
        reconstruction supplies. Without this, zero message tensors still
        flow through Linear biases + attention + LayerNorm and produce a
        NONZERO context, contaminating the PPO ratio for episode-start
        transitions (old_log_prob used no communication, new_log_prob
        would use a communication branch that technically exists).
        """

        batch_size = local_hidden.shape[0]
        device = local_hidden.device
        dtype = local_hidden.dtype

        if received_messages is None:
            return torch.zeros(
                batch_size,
                COMMUNICATION_ATTENTION_DIM,
                device=device,
                dtype=dtype,
            )

        messages, trust = self._prepare_received_messages(
            received_messages, trust_weights, batch_size, device, dtype
        )

        keys = self.communication_key(messages)
        values = self.communication_value(messages)

        trust_bias = None
        if trust is not None:

            if trust.shape[1] != messages.shape[1]:
                raise ValueError(
                    "trust_weights and received_messages must "
                    "contain the same number of senders."
                )

            # Clamped so log(trust) below stays finite even for
            # zero/near-zero trust (log(1e-4) ~= -9.21); trust == 1.0
            # maps to a zero bias (no down-weighting).
            trust_safe = torch.clamp(trust, min=1e-4, max=1.0)
            # Bias attention logits: log(trust) down-weights *which*
            # sender gets attended to. This changes the mixture
            # *weights* (direction), which survives the LayerNorm below.
            # Deliberately NO direct values-magnitude scaling here: it
            # was largely erased by that LayerNorm while doubly
            # suppressing low-trust senders together with the PPO
            # credit weight in mappo.py. Trust reaches the sender-side
            # update through that credit weight instead.
            trust_bias = torch.log(trust_safe).unsqueeze(1).to(
                dtype=local_hidden.dtype
            )
            trust_bias = trust_bias.repeat_interleave(
                self.communication_attention.num_heads, dim=0
            )

        query = self.communication_query(local_hidden).unsqueeze(1)

        context, communication_weights = self.communication_attention(
            query, keys, values, need_weights=True, attn_mask=trust_bias
        )

        self.last_communication_attention = communication_weights.detach()

        context = context.squeeze(1)
        context = self.communication_norm(context)

        # Episode-start rows (communication_valid=False) must match the
        # rollout's exact-zero context (received_messages=None path above).
        # The reconstructed zero message tensor would otherwise produce a
        # nonzero context via Linear biases + attention + LayerNorm.
        if communication_valid is not None:
            valid = torch.as_tensor(
                communication_valid, device=device, dtype=torch.bool
            ).reshape(-1)
            if valid.shape[0] != batch_size:
                raise ValueError(
                    "communication_valid must have shape [B]. Got "
                    f"{tuple(communication_valid.shape) if hasattr(communication_valid, 'shape') else '?'} "
                    f"for batch_size={batch_size}."
                )
            context = context * valid.to(dtype=context.dtype).view(-1, 1)

        return context

    # ======================================================
    # Main forward (unchanged interface)
    # ======================================================

    def forward(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
        host_active_mask: Optional[torch.Tensor] = None,
        communication_valid: Optional[torch.Tensor] = None,
    ):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        local_hidden = self._get_local_hidden(
            observation, host_active_mask=host_active_mask
        )

        # Only sample/encode an outgoing message when the caller
        # actually wants it (return_communication=True). Plain policy
        # forward calls -- e.g. actor_forward() during PPO's
        # evaluate_actions() -- previously generated and discarded a
        # full message every call, roughly doubling forward cost for
        # no effect on `logits`.
        if return_communication:
            (
                field_ids,
                message_log_probs,
                message_entropies,
                outgoing_message,
            ) = self.generate_communication(local_hidden)

        # communication_valid is only meaningful for the batched path;
        # for the squeezed single-observation path it is reshaped to [1]
        # inside _apply_received_communication.
        communication_context = self._apply_received_communication(
            local_hidden=local_hidden,
            received_messages=received_messages,
            trust_weights=trust_weights,
            communication_valid=communication_valid,
        )

        policy_representation = torch.cat(
            [local_hidden, communication_context], dim=-1
        )
        policy_representation = self.policy_input_projection(
            policy_representation
        )

        logits = self.policy_head(policy_representation)

        if squeeze_output:
            logits = logits.squeeze(0)

        if return_communication:
            if squeeze_output:
                outgoing_message = outgoing_message.squeeze(0)

            return (
                logits,
                outgoing_message,
                field_ids,
                message_log_probs,
                message_entropies,
            )

        return logits

    # ======================================================
    # Communication-only forward (unchanged)
    # ======================================================

    def get_outgoing_message(
        self,
        observation: torch.Tensor,
        host_active_mask: Optional[torch.Tensor] = None,
    ):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        local_hidden = self._get_local_hidden(
            observation, host_active_mask=host_active_mask
        )

        (
            field_ids,
            log_probs,
            entropies,
            communication_vector,
        ) = self.generate_communication(local_hidden)

        if squeeze_output:
            communication_vector = communication_vector.squeeze(0)
            field_ids = {k: v.squeeze(0) for k, v in field_ids.items()}
            log_probs = {k: v.squeeze(0) for k, v in log_probs.items()}
            entropies = {k: v.squeeze(0) for k, v in entropies.items()}

        return communication_vector, field_ids, log_probs, entropies

    # ======================================================
    # Local hidden (unchanged)
    # ======================================================

    def get_local_hidden(
        self,
        observation: torch.Tensor,
        host_active_mask: Optional[torch.Tensor] = None,
    ):

        squeeze_output = observation.dim() == 1

        if squeeze_output:
            observation = observation.unsqueeze(0)

        hidden = self._get_local_hidden(
            observation, host_active_mask=host_active_mask
        )

        if squeeze_output:
            hidden = hidden.squeeze(0)

        return hidden


# ==========================================================
# Central Critic (unchanged)
# ==========================================================

class CentralCritic(nn.Module):
    """
    Centralized critic. Unchanged -- cross-agent attention already
    models inter-agent dependency directly, which is a different
    graph (agent-level) than the local host/subnet graph above.
    """

    def __init__(self):

        super().__init__()

        self.agent_embed = nn.Linear(OBS_DIM, EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.value_head = build_mlp(EMBED_DIM, 1)

    def forward(self, global_state: torch.Tensor):

        squeeze_output = global_state.dim() == 1

        if squeeze_output:
            global_state = global_state.unsqueeze(0)

        batch_size = global_state.shape[0]

        per_agent_obs = global_state.view(
            batch_size, NUM_AGENTS, OBS_DIM
        )

        tokens = torch.relu(self.agent_embed(per_agent_obs))

        attn_out, attention_weights = self.attention(
            tokens, tokens, tokens
        )
        self.last_attention = attention_weights.detach()

        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))

        values = self.value_head(x).squeeze(-1)

        if squeeze_output:
            values = values.squeeze(0)

        return values


# ==========================================================
# MAPPO Model (unchanged)
# ==========================================================

class MAPPOModel(nn.Module):
    """
    Complete MAPPO model. Contains SharedActor + CentralCritic.

    num_targets -> num_host_targets/num_subnet_targets: see
    SharedActor's matching constructor comment. Both are optional and
    independent -- either may be None if that vocabulary isn't
    configured yet.
    """

    def __init__(
        self,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
        use_host_graph: bool = True,
        subnet_edges: Optional[List[Tuple[int, int]]] = SUBNET_EDGES,
    ):

        super().__init__()

        self.actor = SharedActor(
            num_agents=NUM_AGENTS,
            communication_dim=COMMUNICATION_DIM,
            communication_latent_dim=COMMUNICATION_LATENT_DIM,
            num_host_targets=num_host_targets,
            num_subnet_targets=num_subnet_targets,
            use_host_graph=use_host_graph,
            subnet_edges=subnet_edges,
        )

        self.critic = CentralCritic()

    def act(
        self,
        observation: torch.Tensor,
        received_messages: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None,
        return_communication: bool = False,
        host_active_mask: Optional[torch.Tensor] = None,
    ):

        return self.actor(
            observation,
            received_messages=received_messages,
            trust_weights=trust_weights,
            return_communication=return_communication,
            host_active_mask=host_active_mask,
        )

    def evaluate(self, global_state: torch.Tensor):

        return self.critic(global_state)