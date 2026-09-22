"""
decoder.py (v3 -- dual HOST/SUBNET target heads, single target_id field)

Key change from v2: target_id used to come from ONE shared `target_head`,
sized by whatever single `num_targets` the caller passed in -- but
target_id's MEANING depends on target_type (see schema.py's module
comment on StructuredMessage.target_id): HOST indexes the host
vocabulary, SUBNET indexes a completely separate subnet vocabulary. A
shared head made host index 3 and subnet index 3 indistinguishable to
the network.

Fixed here to match encoder.py's dual-table design, WITHOUT changing the
message format: target_id stays ONE field (schema.py's 7-field
MESSAGE_FIELDS is unchanged, and mappo.py's field_order / buffer.py
storage do not need to change). Internally, TWO heads exist --
`host_target_head` / `subnet_target_head`, configured via
build_target_heads(num_host_targets, num_subnet_targets) -- and which
one actually produces the single `target_id` value (and its log_prob/
entropy) for a given row is decided by that SAME row's `target_type`:

    target_type == HOST   -> target_id, its log_prob and entropy come
                              ONLY from host_target_head.
    target_type == SUBNET -> target_id, its log_prob and entropy come
                              ONLY from subnet_target_head.
    target_type == NONE   -> target_id is a placeholder 0; log_prob and
                              entropy are exactly 0.0 -- NONE never
                              trains a target_id, since there was no
                              target to name.

Both heads' distributions are computed for the whole batch (cheap linear
layers), then combined per-row with `torch.where` keyed on target_type --
mirroring encoder.py's mask-and-select pattern for the exact same
selection problem. This keeps everything batched and differentiable
without any Python-level branching per example, while guaranteeing a
HOST row's target_id/log_prob/entropy never comes from the subnet head
and vice versa, and a NONE row's target_id never receives any policy
gradient (log_prob==0 for that row's contribution to the joint message
log-prob sum -- there is nothing to reinforce or penalize about an
unused, placeholder id).

Changes carried over from v2:

1. confidence is a small categorical head (ConfidenceLevel, 4 buckets)
   instead of a raw sigmoid scalar. A bare continuous point-estimate has
   no valid log_prob under a policy-gradient training scheme -- everything
   transmitted needs to come from a sampleable distribution so it can be
   trained the same way as the rest of PPO.

2. sample_message() / evaluate_message() are the actual training
   interface. forward() is kept because sample/evaluate call it
   internally, but it's no longer meant to be used directly by training
   code.

sample_message() is used at rollout time (no_grad): draws a discrete id
per field and returns the log_prob of that draw under the CURRENT policy.

evaluate_message() is used at PPO-update time: given field ids that were
sampled and stored during rollout, recomputes their log_prob under the
policy being updated. This is the exact same role Categorical(logits=...)
.log_prob(stored_action) plays for the regular environment action -- the
PPO ratio needs old vs. new log_prob of the SAME sampled value, not a new
sample. This is why evaluate_message() must select the SAME head
sample_message() would have used for that row -- i.e. keyed on the
STORED target_type, never re-decided from scratch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .schema import (
    ConfidenceLevel,
    EventType,
    HostStatus,
    Priority,
    StructuredMessage,
    TargetType,
    ThreatLevel,
    confidence_level_to_value,
)

# Fields sampled the same way every time: (field_name, output_dict_key, enum_type)
# target_id is handled separately below (see host_target_head /
# subnet_target_head / _sample_target_id / _evaluate_target_id), since
# which head it comes from depends on this same call's target_type.
_CATEGORICAL_FIELDS = (
    ("event_type", "event_logits", EventType),
    ("target_type", "target_type_logits", TargetType),
    ("threat_level", "threat_logits", ThreatLevel),
    ("status", "status_logits", HostStatus),
    ("priority", "priority_logits", Priority),
    ("confidence", "confidence_logits", ConfidenceLevel),
)


class MessageDecoder(nn.Module):
    """Decodes an agent latent representation into structured message field logits."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.event_head = nn.Linear(hidden_dim, len(EventType))
        self.target_type_head = nn.Linear(hidden_dim, len(TargetType))
        self.threat_head = nn.Linear(hidden_dim, len(ThreatLevel))
        self.status_head = nn.Linear(hidden_dim, len(HostStatus))
        self.priority_head = nn.Linear(hidden_dim, len(Priority))

        # Confidence: categorical bucket, not a sigmoid scalar. See module docstring.
        self.confidence_head = nn.Linear(hidden_dim, len(ConfidenceLevel))

        # target_id depends on the CC4 host/subnet mapping, configured
        # later -- TWO separate heads (matching encoder.py's two separate
        # embedding tables), since HOST and SUBNET target_id are two
        # separate vocabularies (see module docstring). The single
        # `target_head` from v2 is removed entirely -- there is no
        # shared-head code path left anywhere in this file.
        self.host_target_head: Optional[nn.Linear] = None
        self.subnet_target_head: Optional[nn.Linear] = None

    def build_target_heads(
        self,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
    ) -> None:
        """
        Configure either or both target vocabularies. Mirrors
        encoder.py's build_target_embeddings() -- HOST and SUBNET are
        independent here, so you may configure just one (e.g. a
        scenario with no subnet targets yet) without forcing the other
        into existence.
        """

        if num_host_targets is not None:

            if num_host_targets <= 0:
                raise ValueError("num_host_targets must be greater than zero")

            self.host_target_head = nn.Linear(self.hidden_dim, num_host_targets)

        if num_subnet_targets is not None:

            if num_subnet_targets <= 0:
                raise ValueError("num_subnet_targets must be greater than zero")

            self.subnet_target_head = nn.Linear(self.hidden_dim, num_subnet_targets)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw logits for every field. Internal use by sample_message/evaluate_message."""
        if hidden.ndim != 2:
            raise ValueError(f"Expected [batch, input_dim], got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {hidden.shape[-1]}")

        z = self.shared(hidden)

        outputs = {
            "event_logits": self.event_head(z),
            "target_type_logits": self.target_type_head(z),
            "threat_logits": self.threat_head(z),
            "status_logits": self.status_head(z),
            "priority_logits": self.priority_head(z),
            "confidence_logits": self.confidence_head(z),
            "host_target_logits": (
                self.host_target_head(z) if self.host_target_head is not None else None
            ),
            "subnet_target_logits": (
                self.subnet_target_head(z) if self.subnet_target_head is not None else None
            ),
        }
        return outputs

    # ------------------------------------------------------------------
    # Conditional target_id selection (shared by sample/evaluate)
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_host_logits(
        host_logits: torch.Tensor,
        host_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask per-episode-invalid HOST ids out of the target distribution
        BEFORE sampling/scoring (never sample-then-discard).

        host_valid_mask ([B] or [B, H] bool) uses the STABLE host order
        (env.STABLE_HOST_LIST, the same mapping as ground truth): True
        = the host exists in this episode inside the sender's own slots.
        Out-of-zone hostnames, routers (no observation slots), and
        nonexistent hosts are False.

        A row with zero valid ids falls back to unmasked (mirroring the
        action-mask Sleep safety net) so sampling can never hit an
        all -inf row; in practice every zone always has >=4 real hosts.
        """

        valid = host_valid_mask.to(dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[-1] != host_logits.shape[-1]:
            raise ValueError(
                "host_valid_mask last dim "
                f"{valid.shape[-1]} does not match host vocabulary "
                f"{host_logits.shape[-1]} -- mask and head disagree on "
                "the stable host order."
            )
        if valid.shape[0] == 1 and host_logits.shape[0] > 1:
            valid = valid.expand(host_logits.shape[0], -1)
        if valid.shape[0] != host_logits.shape[0]:
            raise ValueError(
                "host_valid_mask batch "
                f"{valid.shape[0]} does not match logits batch "
                f"{host_logits.shape[0]}."
            )
        row_has_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = valid | ~row_has_valid
        return host_logits.masked_fill(~safe_valid, -1e10)

    @staticmethod
    def _masked_subnet_logits(
        subnet_logits: torch.Tensor,
        subnet_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask unobservable SUBNET ids out of the target distribution
        BEFORE sampling/scoring (never sample-then-discard).

        subnet_valid_mask ([B] or [B, S] bool) uses the STABLE subnet
        order (env.STABLE_SUBNET_LIST, the same mapping as ground truth
        subnet_status): True = the subnet is observable by this sender
        (in its own zone -- see env.get_subnet_valid_mask). A sender has
        no observation basis for any other subnet, and such a claim
        grades as wrong, so it is kept out of the distribution entirely.

        A row with zero valid ids falls back to unmasked (same safety
        net as the host mask) so sampling can never hit an all -inf
        row; in practice every agent observes >=1 subnet.
        """
        valid = subnet_valid_mask.to(dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[-1] != subnet_logits.shape[-1]:
            raise ValueError(
                "subnet_valid_mask last dim "
                f"{valid.shape[-1]} does not match subnet vocabulary "
                f"{subnet_logits.shape[-1]} -- mask and head disagree on "
                "the stable subnet order."
            )
        if valid.shape[0] == 1 and subnet_logits.shape[0] > 1:
            valid = valid.expand(subnet_logits.shape[0], -1)
        if valid.shape[0] != subnet_logits.shape[0]:
            raise ValueError(
                "subnet_valid_mask batch "
                f"{valid.shape[0]} does not match logits batch "
                f"{subnet_logits.shape[0]}."
            )
        row_has_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = valid | ~row_has_valid
        return subnet_logits.masked_fill(~safe_valid, -1e10)

    def _sample_target_id(
        self,
        outputs: Dict[str, torch.Tensor],
        target_type_ids: torch.Tensor,
        host_valid_mask: torch.Tensor = None,
        subnet_valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample target_id for a whole batch, per-row selecting HOST vs
        SUBNET vs NONE based on `target_type_ids` (that same row's
        ALREADY-sampled target_type -- this never re-decides target_type
        itself, only which head answers target_id given it).

        NONE rows get target_id=0 (placeholder, never used downstream --
        see schema.py: target_id is ignored when target_type is NONE)
        and log_prob=entropy=0.0 -- NONE never trains a target_id.

        host_valid_mask (optional) restricts the HOST head to
        per-episode-valid ids BEFORE sampling -- see
        _masked_host_logits(). The mask must use the STABLE host
        order; SUBNET/NONE rows are unaffected.

        subnet_valid_mask (optional) restricts the SUBNET head to the
        sender's observable subnets BEFORE sampling -- see
        _masked_subnet_logits() and env.get_subnet_valid_mask(). The
        mask must use the STABLE subnet order; HOST/NONE rows are
        unaffected.
        """

        batch_shape = target_type_ids.shape
        device = target_type_ids.device

        target_id = torch.zeros(batch_shape, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch_shape, dtype=torch.float32, device=device)

        is_host = target_type_ids == int(TargetType.HOST)
        is_subnet = target_type_ids == int(TargetType.SUBNET)

        if outputs["host_target_logits"] is not None:
            host_logits = outputs["host_target_logits"]
            if host_valid_mask is not None:
                host_logits = self._masked_host_logits(
                    host_logits, host_valid_mask
                )
            host_dist = Categorical(logits=host_logits)
            host_sample = host_dist.sample()
            target_id = torch.where(is_host, host_sample, target_id)
            log_prob = torch.where(is_host, host_dist.log_prob(host_sample), log_prob)
            entropy = torch.where(is_host, host_dist.entropy(), entropy)

        if outputs["subnet_target_logits"] is not None:
            subnet_logits = outputs["subnet_target_logits"]
            if subnet_valid_mask is not None:
                subnet_logits = self._masked_subnet_logits(
                    subnet_logits, subnet_valid_mask
                )
            subnet_dist = Categorical(logits=subnet_logits)
            subnet_sample = subnet_dist.sample()
            target_id = torch.where(is_subnet, subnet_sample, target_id)
            log_prob = torch.where(is_subnet, subnet_dist.log_prob(subnet_sample), log_prob)
            entropy = torch.where(is_subnet, subnet_dist.entropy(), entropy)

        return target_id, log_prob, entropy

    def _evaluate_target_id(
        self,
        outputs: Dict[str, torch.Tensor],
        target_type_ids: torch.Tensor,
        target_id: torch.Tensor,
        host_valid_mask: torch.Tensor = None,
        subnet_valid_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute target_id's log_prob/entropy for a whole batch of
        STORED (target_type, target_id) pairs, selecting HOST vs SUBNET
        vs NONE the same way _sample_target_id() does -- keyed on the
        STORED target_type, since that is what determined which head
        actually produced the stored target_id at rollout time.

        NONE rows contribute log_prob=entropy=0.0, exactly matching
        _sample_message()'s convention -- so a NONE row's PPO ratio for
        this field is log(1)-log(1)=0, i.e. genuinely no gradient, not
        an arbitrary head's opinion about an id nobody used.

        target_id is clamped into range before each head's log_prob
        lookup purely to avoid an IndexError on a stale/out-of-range
        value (e.g. replaying an older buffer, or a manually-constructed
        message) -- rows where that head isn't the one actually selected
        by target_type are masked out immediately after regardless of
        what this defensive dummy lookup computes.

        host_valid_mask (optional) applies the IDENTICAL validity
        semantics as sampling (see _masked_host_logits): the stored id
        was sampled under this same episode's mask, so replaying under
        it keeps old/new log-probs comparable for the PPO ratio.

        subnet_valid_mask (optional) applies the IDENTICAL validity
        semantics as sampling for the SUBNET head (see
        _masked_subnet_logits).
        """

        batch_shape = target_type_ids.shape
        device = target_type_ids.device

        log_prob = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch_shape, dtype=torch.float32, device=device)

        is_host = target_type_ids == int(TargetType.HOST)
        is_subnet = target_type_ids == int(TargetType.SUBNET)

        if outputs["host_target_logits"] is not None:
            host_logits = outputs["host_target_logits"]
            if host_valid_mask is not None:
                host_logits = self._masked_host_logits(
                    host_logits, host_valid_mask
                )
            host_dist = Categorical(logits=host_logits)
            safe_target_id = target_id.clamp(0, host_dist.logits.shape[-1] - 1)
            log_prob = torch.where(is_host, host_dist.log_prob(safe_target_id), log_prob)
            entropy = torch.where(is_host, host_dist.entropy(), entropy)

        if outputs["subnet_target_logits"] is not None:
            subnet_logits = outputs["subnet_target_logits"]
            if subnet_valid_mask is not None:
                subnet_logits = self._masked_subnet_logits(
                    subnet_logits, subnet_valid_mask
                )
            subnet_dist = Categorical(logits=subnet_logits)
            safe_target_id = target_id.clamp(0, subnet_dist.logits.shape[-1] - 1)
            log_prob = torch.where(is_subnet, subnet_dist.log_prob(safe_target_id), log_prob)
            entropy = torch.where(is_subnet, subnet_dist.entropy(), entropy)

        return log_prob, entropy

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def sample_message(
        self,
        hidden: torch.Tensor,
        host_valid_mask: torch.Tensor = None,
        subnet_valid_mask: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Sample a discrete structured message from the current policy.

        target_id (a single field, matching schema.py's 7-field
        MESSAGE_FIELDS) is populated conditionally on this SAME call's
        sampled target_type -- see _sample_target_id().

        host_valid_mask (optional [B] or [B, H] bool, STABLE host
        order) masks per-episode-invalid HOST ids BEFORE sampling --
        see _masked_host_logits(). None preserves the legacy unmasked
        behavior.

        subnet_valid_mask (optional [B] or [B, S] bool, STABLE subnet
        order) masks unobservable SUBNET ids BEFORE sampling -- see
        _masked_subnet_logits(). None preserves the legacy unmasked
        behavior.

        Returns
        -------
        field_ids : dict[str, LongTensor[batch]]
        log_probs : dict[str, FloatTensor[batch]]
        entropies : dict[str, FloatTensor[batch]]
        """
        outputs = self.forward(hidden)

        field_ids: Dict[str, torch.Tensor] = {}
        log_probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}

        for field, logits_key, _enum in _CATEGORICAL_FIELDS:
            dist = Categorical(logits=outputs[logits_key])
            sample = dist.sample()
            field_ids[field] = sample
            log_probs[field] = dist.log_prob(sample)
            entropies[field] = dist.entropy()

        target_id, target_log_prob, target_entropy = self._sample_target_id(
            outputs,
            field_ids["target_type"],
            host_valid_mask,
            subnet_valid_mask,
        )
        field_ids["target_id"] = target_id
        log_probs["target_id"] = target_log_prob
        entropies["target_id"] = target_entropy

        return field_ids, log_probs, entropies

    def evaluate_message(
        self,
        hidden: torch.Tensor,
        field_ids: Dict[str, torch.Tensor],
        host_valid_mask: torch.Tensor = None,
        subnet_valid_mask: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Recompute log_prob/entropy of a STORED sample under current parameters.
        Used inside the PPO update -- same role as
        Categorical(logits=...).log_prob(stored_action) for the env action.

        target_id's log_prob/entropy are recomputed via the head selected
        by the STORED target_type -- see _evaluate_target_id().

        host_valid_mask (optional [B] or [B, H] bool, STABLE host order)
        must carry the IDENTICAL validity semantics as sampling (same
        episode's mask): the stored id was sampled under it, so replaying
        under it keeps old/new log-probs comparable. None preserves the
        legacy unmasked behavior.

        subnet_valid_mask (optional [B] or [B, S] bool, STABLE subnet
        order) carries the IDENTICAL semantics for the SUBNET head.
        """
        outputs = self.forward(hidden)

        log_probs: Dict[str, torch.Tensor] = {}
        entropies: Dict[str, torch.Tensor] = {}

        for field, logits_key, _enum in _CATEGORICAL_FIELDS:
            dist = Categorical(logits=outputs[logits_key])
            log_probs[field] = dist.log_prob(field_ids[field])
            entropies[field] = dist.entropy()

        if "target_id" in field_ids:
            target_log_prob, target_entropy = self._evaluate_target_id(
                outputs,
                field_ids["target_type"],
                field_ids["target_id"],
                host_valid_mask,
                subnet_valid_mask,
            )
            log_probs["target_id"] = target_log_prob
            entropies["target_id"] = target_entropy

        return log_probs, entropies

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> StructuredMessage:
        """Hard argmax decode for inference/debugging/logging only. Not used in training."""
        if hidden.ndim != 1:
            raise ValueError("decode() expects a single vector with shape [input_dim]")

        outputs = self.forward(hidden.unsqueeze(0))

        event_type = EventType(torch.argmax(outputs["event_logits"][0]).item())
        target_type = TargetType(torch.argmax(outputs["target_type_logits"][0]).item())
        threat_level = ThreatLevel(torch.argmax(outputs["threat_logits"][0]).item())
        status = HostStatus(torch.argmax(outputs["status_logits"][0]).item())
        priority = Priority(torch.argmax(outputs["priority_logits"][0]).item())

        confidence_level = ConfidenceLevel(torch.argmax(outputs["confidence_logits"][0]).item())
        # Canonical bucket -> [0,1] mapping, shared with
        # structured_communication.py.structured_message_from_ids() --
        # see schema.py's CONFIDENCE_LEVEL_VALUES docstring for why this
        # must not be a separately-hardcoded conversion.
        confidence = confidence_level_to_value(confidence_level)

        # Pick target_id from whichever vocabulary target_type actually
        # selected -- HOST uses ONLY host_target_logits, SUBNET uses
        # ONLY subnet_target_logits, NONE never touches either head.
        if target_type == TargetType.HOST and outputs["host_target_logits"] is not None:
            target_id = torch.argmax(outputs["host_target_logits"][0]).item()
        elif target_type == TargetType.SUBNET and outputs["subnet_target_logits"] is not None:
            target_id = torch.argmax(outputs["subnet_target_logits"][0]).item()
        else:
            target_id = 0

        return StructuredMessage(
            event_type=event_type,
            target_type=target_type,
            target_id=target_id,
            threat_level=threat_level,
            confidence=confidence,
            status=status,
            priority=priority,
        )


# ============================================================================
# Basic validation / self-test
# ============================================================================

def _run_self_test() -> None:
    """
    Minimal smoke test covering HOST, SUBNET, and NONE target-id
    handling in both the sample and evaluate paths, plus decode()'s
    argmax path. Run directly:
    `python -m Marl.mappo.communication.decoder` (needs torch).
    """

    torch.manual_seed(0)

    input_dim = 16
    hidden_dim = 8
    num_host_targets = 6
    num_subnet_targets = 3

    decoder = MessageDecoder(input_dim=input_dim, hidden_dim=hidden_dim)
    decoder.build_target_heads(num_host_targets, num_subnet_targets)
    decoder.eval()

    batch = 6
    hidden = torch.randn(batch, input_dim)

    with torch.no_grad():
        outputs = decoder.forward(hidden)

        # Force one row of each type (bypassing the randomly-sampled
        # target_type) so the conditional head selection can be checked
        # deterministically for all three cases.
        target_type_ids = torch.tensor(
            [
                int(TargetType.HOST),
                int(TargetType.HOST),
                int(TargetType.SUBNET),
                int(TargetType.SUBNET),
                int(TargetType.NONE),
                int(TargetType.NONE),
            ]
        )

        # ---- sample_message's conditional selection, via the shared helper ----
        target_id, log_prob, entropy = decoder._sample_target_id(outputs, target_type_ids)

        assert (target_id[:2] < num_host_targets).all(), "HOST rows must sample within host vocab"
        assert (target_id[2:4] < num_subnet_targets).all(), "SUBNET rows must sample within subnet vocab"
        assert torch.all(log_prob[4:6] == 0.0), "NONE rows must have log_prob == 0.0"
        assert torch.all(entropy[4:6] == 0.0), "NONE rows must have entropy == 0.0"
        assert torch.all(log_prob[:4] != 0.0), "HOST/SUBNET rows must have a real (nonzero) log_prob"

        # ---- evaluate_message's conditional selection must match, for
        # the SAME stored (target_type, target_id) pairs ----
        eval_log_prob, eval_entropy = decoder._evaluate_target_id(
            outputs, target_type_ids, target_id
        )
        assert torch.allclose(log_prob, eval_log_prob), (
            "evaluate path must reproduce the same log_prob as sample path "
            "for identical (target_type, target_id) under the same params"
        )
        assert torch.all(eval_log_prob[4:6] == 0.0), "NONE rows must stay log_prob == 0.0 on replay"

        # ---- decode()'s argmax path must never mix vocabularies ----
        for _ in range(50):
            single_hidden = torch.randn(input_dim)
            message = decoder.decode(single_hidden)
            if message.target_type == TargetType.HOST:
                assert 0 <= message.target_id < num_host_targets
            elif message.target_type == TargetType.SUBNET:
                assert 0 <= message.target_id < num_subnet_targets
            else:
                assert message.target_id == 0

    print("decoder.py self-test passed: HOST/SUBNET/NONE target_id "
          "selection is conditional and consistent between sample_message, "
          "evaluate_message, and decode().")


if __name__ == "__main__":
    _run_self_test()