"""
structured_communication.py

Main structured communication module for CC4 MARL.

This module connects:

    Agent latent representation
            |
            v
    MessageDecoder
            |
            v
    Structured message representation
            |
            v
    MessageEncoder
            |
            v
    Communication vector
            |
            v
    Receiver
            |
            v
    DynamicTrust
            |
            v
    Trust-weighted communication

The module is deliberately independent of CybORG.

CybORG interaction remains in env.py.

Trust feedback is supplied externally by evaluator.py.

Communication direction:

    sender -> receiver

Trust direction:

    receiver trusts sender

Therefore:

    trust[sender][receiver]

represents the receiver's trust in the sender.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .decoder import MessageDecoder
from .encoder import MessageEncoder
from .schema import StructuredMessage, confidence_level_to_value
from .trust import DynamicTrust


class StructuredCommunication(nn.Module):
    """
    Complete structured communication system.

    Components
    ----------
    decoder:
        Converts an agent latent representation into structured
        message fields.

    encoder:
        Converts structured message fields into a fixed-size
        communication vector.

    trust:
        Maintains dynamic directed trust between agents.

    Parameters
    ----------
    input_dim : int
        Dimension of the latent representation received from the
        actor.

    message_dim : int
        Dimension of the final communication vector.

    decoder_hidden_dim : int
        Hidden dimension used by the decoder.

    encoder_embedding_dim : int
        Embedding dimension used by the encoder.

    encoder_hidden_dim : int
        Hidden dimension used by the encoder.

    num_agents : int
        Number of Blue agents.

    num_host_targets : int, optional
        Number of possible HOST targets. See decoder.py /
        encoder.py's module docstrings -- HOST and SUBNET are
        separate, independently-sized vocabularies now; there is no
        longer a single combined `num_targets`.

    num_subnet_targets : int, optional
        Number of possible SUBNET targets. Independent of
        `num_host_targets` above.

    trust_decay : float
        Forgetting factor used by DynamicTrust.

    Notes
    -----
    The default architecture assumes:

        actor latent
            -> 256
            -> structured decoder
            -> structured encoder
            -> 128-dimensional message
    """

    def __init__(
        self,
        input_dim: int = 256,
        message_dim: int = 128,
        decoder_hidden_dim: int = 128,
        encoder_embedding_dim: int = 16,
        encoder_hidden_dim: int = 128,
        num_agents: int = 5,
        num_host_targets: Optional[int] = None,
        num_subnet_targets: Optional[int] = None,
        trust_decay: float = 0.995,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.message_dim = message_dim
        self.num_agents = num_agents

        # ---------------------------------------------------------------
        # Structured message decoder
        # ---------------------------------------------------------------

        self.decoder = MessageDecoder(
            input_dim=input_dim,
            hidden_dim=decoder_hidden_dim,
        )

        # Configure target vocabularies if available. HOST and SUBNET
        # are independent -- either, both, or neither may be
        # configured at construction time (see decoder.py's
        # build_target_heads).
        if num_host_targets is not None or num_subnet_targets is not None:
            self.decoder.build_target_heads(
                num_host_targets,
                num_subnet_targets,
            )

        # ---------------------------------------------------------------
        # Structured message encoder
        # ---------------------------------------------------------------

        self.encoder = MessageEncoder(
            message_dim=message_dim,
            embedding_dim=encoder_embedding_dim,
            hidden_dim=encoder_hidden_dim,
            num_host_targets=num_host_targets,
            num_subnet_targets=num_subnet_targets,
        )

        # ---------------------------------------------------------------
        # Dynamic trust
        #
        # Trust is stateful and intentionally kept outside the neural
        # computation graph.
        # ---------------------------------------------------------------

        self.trust = DynamicTrust(
            num_agents=num_agents,
            decay=trust_decay,
        )

    # ==================================================================
    # Sender side
    # ==================================================================

    def generate_message(
            self,
            hidden: torch.Tensor,
            host_valid_mask: torch.Tensor = None,
            subnet_valid_mask: torch.Tensor = None,
        ) -> tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ]:
            """
            Sample a structured communication message and encode it.

            host_valid_mask (optional [B] or [B, H] bool, STABLE host
            order) masks per-episode-invalid HOST ids BEFORE sampling --
            see MessageDecoder._masked_host_logits(). None preserves the
            legacy unmasked behavior.

            subnet_valid_mask (optional [B] or [B, S] bool, STABLE subnet
            order) masks unobservable SUBNET ids BEFORE sampling -- see
            MessageDecoder._masked_subnet_logits(). None preserves the
            legacy unmasked behavior.

            Returns
            -------
            field_ids:
                Sampled categorical message fields.

            log_probs:
                Log-probability of each sampled message field.

            entropies:
                Entropy of each message-field distribution.

            communication_vector:
                Encoded communication vector.
            """

            if hidden.ndim == 1:
                hidden = hidden.unsqueeze(0)

            # ------------------------------------------------------------
            # V2 decoder (mask shapes are validated/expanded inside
            # MessageDecoder._masked_host_logits/_masked_subnet_logits)
            # ------------------------------------------------------------

            field_ids, log_probs, entropies = (
                self.decoder.sample_message(
                    hidden, host_valid_mask, subnet_valid_mask
                )
            )

            # ------------------------------------------------------------
            # V2 encoder
            # ------------------------------------------------------------

            communication_vector = (
                self.encoder.encode_from_ids(field_ids)
            )

            return (
                field_ids,
                log_probs,
                entropies,
                communication_vector,
            )
    # ==================================================================
    # Inference/debugging
    # ==================================================================
    def evaluate_message(
            self,
            hidden: torch.Tensor,
            field_ids: Dict[str, torch.Tensor],
            host_valid_mask: torch.Tensor = None,
            subnet_valid_mask: torch.Tensor = None,
        ) -> tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
        ]:
            """
            Re-evaluate a stored message under the current decoder.

            Used during PPO updates.

            host_valid_mask (optional [B] or [B, H] bool, STABLE host
            order) must carry the IDENTICAL validity semantics as
            sampling (same episode's mask) so old/new log-probs stay
            comparable. None preserves the legacy unmasked behavior.

            subnet_valid_mask (optional [B] or [B, S] bool, STABLE subnet
            order) carries the IDENTICAL semantics for the SUBNET head.

            Returns
            -------
            log_probs:
                Current log-probabilities of the stored message fields.

            entropies:
                Current entropy of each message-field distribution.
            """

            if hidden.ndim == 1:
                hidden = hidden.unsqueeze(0)
            return self.decoder.evaluate_message(
                hidden,
                field_ids,
                host_valid_mask,
                subnet_valid_mask,
            )

    # ==================================================================
    # Build a StructuredMessage from already-sampled field_ids
    # ==================================================================

    def structured_message_from_ids(
        self,
        field_ids: Dict[str, torch.Tensor],
        index: int = 0,
    ) -> StructuredMessage:
        """
        Build a StructuredMessage directly from field_ids that were
        already sampled (e.g. by generate_message()), instead of
        re-decoding `hidden` through the decoder a second time.

        This matters: decoder.decode(hidden) is a SEPARATE forward
        pass and is not guaranteed to reproduce the exact categories
        sample_message() already sampled (e.g. if decode() is
        greedy/argmax while sample_message() is stochastic). Building
        the StructuredMessage from the same field_ids used to build
        the transmitted communication_vector (via encode_from_ids)
        guarantees the message a receiver/evaluator sees always
        matches what was actually encoded and sent -- there is no
        second, independent decoding that can silently disagree with
        it.

        Parameters
        ----------
        field_ids:
            Dict of category-index tensors, each shape [B], as
            returned by generate_message()/sample_message().

        index:
            Which batch row to build a StructuredMessage for.
        """

        return StructuredMessage(
            event_type=int(field_ids["event_type"][index].item()),
            target_type=int(field_ids["target_type"][index].item()),
            target_id=int(field_ids["target_id"][index].item()),
            threat_level=int(field_ids["threat_level"][index].item()),
            confidence=confidence_level_to_value(
                field_ids["confidence"][index].item()
            ),
            status=int(field_ids["status"][index].item()),
            priority=int(field_ids["priority"][index].item()),
        )

    @torch.no_grad()
    def generate_hard_message(
        self,
        hidden: torch.Tensor,
    ) -> tuple[
        StructuredMessage,
        torch.Tensor,
    ]:
        """
        Generate a concrete StructuredMessage.

        Intended for evaluation, debugging and logging.
        """

        if hidden.ndim != 1:
            raise ValueError(
                "generate_hard_message() expects a single "
                "[input_dim] latent vector."
            )

        field_ids, _, _ = self.decoder.sample_message(
            hidden.unsqueeze(0)
        )

        communication_vector = (
            self.encoder.encode_from_ids(field_ids)
        )

        # Built from the SAME field_ids used above -- see
        # structured_message_from_ids()'s docstring for why this
        # replaces a separate decoder.decode(hidden) call.
        decoded = self.structured_message_from_ids(field_ids, index=0)

        return (
            decoded,
            communication_vector.squeeze(0),
        )

    # ==================================================================
    # Communication routing
    # ==================================================================

    def create_message_matrix(
        self,
        communication_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create the sender -> receiver communication matrix.

        Parameters
        ----------
        communication_vectors : torch.Tensor

            Shape:

                [num_agents, message_dim]

        Returns
        -------
        torch.Tensor

            Shape:

                [num_agents, num_agents, message_dim]

        Interpretation:

            result[receiver][sender]

        contains the message sent by ``sender`` to ``receiver``.

        Every sender message is initially available to every other
        Blue agent. Trust determines how strongly the receiver
        uses it.

        Self-communication is set to zero.
        """

        if communication_vectors.ndim != 2:
            raise ValueError(
                "communication_vectors must have shape "
                "[num_agents, message_dim]."
            )

        if communication_vectors.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agents, "
                f"got {communication_vectors.shape[0]}."
            )

        if communication_vectors.shape[1] != self.message_dim:
            raise ValueError(
                f"Expected message dimension "
                f"{self.message_dim}, "
                f"got {communication_vectors.shape[1]}."
            )

        # Every receiver gets every sender's vector.
        #
        # [N, D]
        #   ->
        # [N, N, D]
        #
        # First dimension = receiver
        # Second dimension = sender
        messages = communication_vectors.unsqueeze(0).expand(
            self.num_agents,
            -1,
            -1,
        ).clone()

        # Remove self communication.
        indices = torch.arange(
            self.num_agents,
            device=communication_vectors.device,
        )

        messages[
            indices,
            indices,
        ] = 0.0

        return messages

    # ==================================================================
    # Trust-aware communication
    # ==================================================================

    def apply_trust(
        self,
        communication_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the current trust scores to communication vectors.

        Parameters
        ----------
        communication_vectors : torch.Tensor

            Shape:

                [num_agents, message_dim]

        Returns
        -------
        torch.Tensor

            Shape:

                [num_agents, num_agents, message_dim]

            First dimension:
                receiver

            Second dimension:
                sender

        Formula:

            weighted_message[receiver][sender]
                =
            trust(sender -> receiver)
                *
            message[sender]
        """

        messages = self.create_message_matrix(
            communication_vectors
        )

        trust_matrix = self.trust.get_trust_matrix()

        # trust matrix is stored as:
        #
        #     trust[sender][receiver]
        #
        # Convert to:
        #
        #     trust[receiver][sender]
        #
        # so it matches the message matrix.
        trust_receiver_sender = (
            trust_matrix.transpose(0, 1)
        ).to(
            device=communication_vectors.device,
            dtype=communication_vectors.dtype,
        )

        weighted_messages = (
            messages
            * trust_receiver_sender.unsqueeze(-1)
        )

        return weighted_messages

    # ==================================================================
    # Receiver aggregation
    # ==================================================================

    def aggregate_messages(
        self,
        communication_vectors: torch.Tensor,
        receiver: int,
    ) -> torch.Tensor:
        """
        Aggregate all trusted messages received by one agent.

        Parameters
        ----------
        communication_vectors : torch.Tensor

            Shape:

                [num_agents, message_dim]

        receiver : int
            Receiving agent ID.

        Returns
        -------
        torch.Tensor

            Shape:

                [message_dim]

        Notes
        -----
        This is deliberately a simple weighted aggregation.

        Later, when integrating with network_attention.py, we can
        replace this with the actual communication-attention mechanism.

        The current aggregation is:

            sum(
                trust(sender -> receiver)
                * message(sender)
            )
        """

        if not 0 <= receiver < self.num_agents:
            raise IndexError(
                f"receiver must be in "
                f"[0, {self.num_agents - 1}]"
            )

        weighted_messages = self.apply_trust(
            communication_vectors
        )

        received = weighted_messages[
            receiver
        ]

        return received.sum(
            dim=0
        )

    def aggregate_all(
        self,
        communication_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate communication for every receiver.

        Parameters
        ----------
        communication_vectors : torch.Tensor

            Shape:

                [num_agents, message_dim]

        Returns
        -------
        torch.Tensor

            Shape:

                [num_agents, message_dim]

        Each row contains the trust-weighted communication context
        received by that agent.
        """

        weighted_messages = self.apply_trust(
            communication_vectors
        )

        return weighted_messages.sum(
            dim=1
        )

    # ==================================================================
    # Trust feedback
    # ==================================================================

    def update_trust(
        self,
        sender: int,
        receiver: int,
        message_quality: float,
        weight: float = 1.0,
    ) -> float:
        """
        Update trust using feedback from MessageEvaluator.

        Parameters
        ----------
        sender : int
            Agent that sent the message.

        receiver : int
            Agent that received the message.

        message_quality : float
            Quality score produced by MessageEvaluator.

            1.0 -> completely useful/correct
            0.0 -> completely incorrect/useless

        weight : float
            Importance of this feedback.

        Returns
        -------
        float
            Updated trust score.

        Example
        -------
        quality = 0.9

        communication.update_trust(
            sender=0,
            receiver=1,
            message_quality=quality,
        )
        """

        return self.trust.update(
            sender=sender,
            receiver=receiver,
            correctness=message_quality,
            weight=weight,
        )

    def update_trust_matrix(
        self,
        evaluations: Iterable[dict],
    ) -> torch.Tensor:
        """
        Update multiple trust relationships.

        Parameters
        ----------
        evaluations : iterable of dict

            Each dictionary must contain:

                sender
                receiver
                quality

            Optional:

                weight

        Example
        -------

            evaluations = [
                {
                    "sender": 0,
                    "receiver": 1,
                    "quality": 0.9,
                },
                {
                    "sender": 2,
                    "receiver": 1,
                    "quality": 0.2,
                },
            ]

        Returns
        -------
        torch.Tensor
            Updated trust matrix.
        """

        for evaluation in evaluations:

            sender = evaluation["sender"]
            receiver = evaluation["receiver"]
            quality = evaluation["quality"]

            weight = evaluation.get(
                "weight",
                1.0,
            )

            self.update_trust(
                sender=sender,
                receiver=receiver,
                message_quality=quality,
                weight=weight,
            )

        return self.trust.get_trust_matrix()

    # ==================================================================
    # Trust inspection
    # ==================================================================

    def get_trust(
        self,
        sender: int,
        receiver: int,
    ) -> float:
        """
        Get the current trust of sender as perceived by receiver.
        """

        return self.trust.get_trust(
            sender,
            receiver,
        )

    def get_trust_matrix(
        self,
    ) -> torch.Tensor:
        """
        Return the current directed trust matrix.

        matrix[sender][receiver]

        means:

            receiver's trust in sender.
        """

        return self.trust.get_trust_matrix()

    # ==================================================================
    # Reset
    # ==================================================================

    def reset_trust(self) -> None:
        """
        Reset all trust relationships to their initial prior.
        """

        self.trust.reset()

    # ==================================================================
    # Checkpoint support
    # ==================================================================

    def get_trust_state(self) -> dict:
        """
        Return trust state for checkpointing.
        """

        return self.trust.state_dict()

    def load_trust_state(
        self,
        state: dict,
    ) -> None:
        """
        Restore trust state from a checkpoint.
        """

        self.trust.load_state_dict(
            state
        )

    # ==================================================================
    # Utility
    # ==================================================================

    def get_message_dimension(self) -> int:
        """Return the communication vector dimension."""

        return self.message_dim

    def get_num_agents(self) -> int:
        """Return the number of communicating agents."""

        return self.num_agents