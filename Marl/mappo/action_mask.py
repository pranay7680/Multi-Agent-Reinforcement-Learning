"""
action_mask.py

Adaptive Action Mask (AAM) for MAPPO on CAGE Challenge 4 (CC4).

Why this exists
----------------

CybORG's own BlueFixedActionWrapper already produces a *structural* mask
(`env.action_mask(agent)`): True where an action's target host/subnet
exists this episode, False for the padding/no-op slots. That mask is
computed once per `env.reset()` and never changes for the rest of the
episode -- it says nothing about whether an action makes sense *right
now*.

The previous masking in this project (`train.build_action_mask`) was even
coarser: it only marked the first `real_action_dim` slots True, ignoring
CybORG's own structural mask entirely and relying on the network to learn,
purely from reward, that most "[Invalid] <action> <host>" entries are
functionally Sleep.

The Adaptive Action Mask (AAM) adds a second, *contextual* layer on top
of CybORG's structural mask, recomputed fresh every timestep:

    structural mask                (CybORG, episode-static)
            |
            v
    + contextual gate               (this module, per-timestep)
            |
            v
    + safety net (Sleep fallback)
            |
            v
    final per-timestep action mask

Action risk tiers
------------------

Blue's fixed action set (see EnterpriseScenarioGenerator.blue_actions) is
split into four tiers:

    SAFE            Sleep, Monitor
                    No target host. Always structurally valid, never
                    contextually gated.

    INVESTIGATIVE   Analyse, DeployDecoy
                    Low-risk / informational. Left to the structural
                    mask only -- an agent should always be free to look
                    or lay a decoy.

    DISRUPTIVE      Restore, Remove
                    Costly, host-impacting actions (Restore forces a
                    reimage/downtime; Remove kills a session). Only kept
                    valid on hosts that are currently *flagged* -- i.e.
                    show a malicious-process or suspicious-connection
                    event. This is exactly the signal already encoded in
                    the agent's own observation vector
                    (BlueFlatWrapper.observation_change ->
                    `malicious_processes` / `network_connections`), so
                    the gate uses information the agent legitimately has,
                    not privileged red-team state.

    NETWORK_CONTROL BlockTrafficZone, AllowTrafficZone
                    BlockTrafficZone is suppressed only when its source
                    zone (`from_subnet`) is both visible to this agent
                    AND currently quiet (no flagged host). A source zone
                    outside the agent's own observable zone is left to
                    the structural mask: the agent has no signal there,
                    so suppressing would permanently disable every
                    cross-zone Block. AllowTrafficZone (undoing a block)
                    is always left to the structural mask: restoring
                    connectivity is never unsafe.

Safety net
----------

If every action in an agent's real action space would end up False
(e.g. very early in an episode, or a pathological edge case), Sleep is
forced back to True. `Categorical(logits=...)` over an all -1e10 row
returns NaNs, so this guarantee is not optional.

Usage
-----

    from .action_mask import compute_padded_mask

    mask = compute_padded_mask(env, agent_name)
    action, log_prob, value, entropy = ppo.select_action(
        observation=obs, action_mask=mask, ...
    )

`compute_padded_mask` is a function of `(env, agent_name)` plus the
single config flag `config.USE_AAM` -- it holds no other state of its
own, so it is automatically correct across the `env = CC4Env(...)`
re-creation that happens at every episode boundary in train.py /
evaluate.py. No extra bookkeeping is required by callers. Rollout masks
are stored in the buffer and replayed verbatim during PPO
update/re-evaluation, so all stages observe identical USE_AAM behavior.
"""

from __future__ import annotations

import numpy as np

from .config import ACTION_DIM, USE_AAM


# ==========================================================
# Action risk tiers
# ==========================================================

SAFE_ACTIONS = {
    "Sleep",
    "Monitor",
}

INVESTIGATIVE_ACTIONS = {
    "Analyse",
    "DeployDecoy",
}

DISRUPTIVE_ACTIONS = {
    "Restore",
    "Remove",
}

NETWORK_CONTROL_ACTIONS = {
    "BlockTrafficZone",
    "AllowTrafficZone",
}


# ==========================================================
# Per-index action metadata
# ==========================================================

def describe_actions(env, agent_name):
    """
    Build per-index action metadata for one agent.

    Recomputed fresh on every call. This is cheap (~action_dim simple
    attribute reads) and, importantly, always correct even though `env`
    is replaced with a brand-new CC4Env instance at every episode
    boundary -- there is nothing here to go stale.

    Returns
    -------
    list[tuple[str, Optional[str], Optional[str]]]
        One (action_type, target_host, source_zone) tuple per action
        index, in the same order as `env.action_mask(agent_name)`.

        target_host is set for host-targeted actions (Restore, Remove,
        Analyse, DeployDecoy). source_zone is set for BlockTrafficZone /
        AllowTrafficZone (the `from_subnet`). Both are None otherwise.
    """

    meta = []

    for action in env.actions(agent_name):

        action_type = type(action).__name__

        target_host = getattr(action, "hostname", None)
        source_zone = getattr(action, "from_subnet", None)

        meta.append((action_type, target_host, source_zone))

    return meta


# ==========================================================
# Core adaptive mask
# ==========================================================

def compute_adaptive_mask(env, agent_name):
    """
    Compute this agent's real-sized (unpadded) Adaptive Action Mask
    for the CURRENT timestep.

    Parameters
    ----------
    env : CC4Env
    agent_name : str

    Returns
    -------
    np.ndarray[bool], shape (real_action_dim,)
        True  -> structurally valid AND contextually sensible right now.
        False -> invalid, or suppressed for this timestep.
    """

    structural = np.asarray(env.action_mask(agent_name), dtype=bool)
    meta = describe_actions(env, agent_name)

    host_flags = env.get_host_alert_flags(agent_name)
    zone_flags = env.get_zone_alert_flags(agent_name)

    mask = structural.copy()
    sleep_index = None

    for i, (action_type, target_host, source_zone) in enumerate(meta):

        if action_type == "Sleep":
            sleep_index = i

        if not mask[i]:
            # Already structurally invalid -- nothing to add.
            continue

        if action_type in DISRUPTIVE_ACTIONS:
            if not host_flags.get(target_host, False):
                mask[i] = False

        elif action_type == "BlockTrafficZone":
            # Only suppress when the source zone is *known quiet*:
            # the agent can see this source zone (it is one of its
            # own) and nothing there is flagged. A source zone the
            # agent cannot observe at all (outside its own zone) is
            # left to the structural mask -- suppressing it would
            # permanently disable every cross-zone Block, since an
            # agent's alert flags only ever cover its own zone.
            source_key = (
                str(source_zone).lower()
                if source_zone is not None
                else None
            )
            if (
                source_key is not None
                and source_key in zone_flags
                and not zone_flags[source_key]
            ):
                mask[i] = False

        # SAFE_ACTIONS, INVESTIGATIVE_ACTIONS and AllowTrafficZone are
        # left exactly as the structural mask set them.

    # --------------------------------------------------------
    # Safety net -- never return an all-False mask.
    # --------------------------------------------------------

    if not mask.any():

        if sleep_index is not None:
            mask[sleep_index] = True
        else:
            mask[0] = True  # extremely defensive fallback

    return mask


def pad_mask(mask, action_dim=ACTION_DIM):
    """Pad an agent's real-sized mask up to the shared-policy action_dim."""

    padded = np.zeros(action_dim, dtype=bool)
    padded[: len(mask)] = mask

    return padded


def compute_structural_mask(env, agent_name):
    """
    Structural mask only (CybORG episode-static validity + Sleep safety
    net), WITHOUT the contextual Restore/Remove/BlockTrafficZone gating
    in compute_adaptive_mask().

    Use for the MAPPO + structural-mask-only ablation: compare against
    compute_adaptive_mask() under identical seeds to measure how much of
    the result comes from the hand-written heuristic vs learned policy.
    """
    structural = np.asarray(env.action_mask(agent_name), dtype=bool).copy()
    meta = describe_actions(env, agent_name)
    sleep_index = None
    for i, (action_type, _t, _s) in enumerate(meta):
        if action_type == "Sleep":
            sleep_index = i
            break
    if not structural.any():
        if sleep_index is not None:
            structural[sleep_index] = True
        else:
            structural[0] = True
    return structural


def compute_padded_mask(env, agent_name, action_dim=ACTION_DIM):
    """
    Single entry point for train.py / evaluate.py / PPO rollout.

    Single source of truth for AAM enable/disable is config.USE_AAM:

      USE_AAM = True  -> compute_adaptive_mask() (structural mask +
                          contextual AAM gating + Sleep safety net).
      USE_AAM = False -> compute_structural_mask() only (structural /
                          environment-validity mask + Sleep safety net,
                          AAM completely disabled).

    Returns a boolean mask of length `action_dim` (the shared-policy
    dimension, 242 by default), ready to pass straight into
    `ppo.select_action(..., action_mask=mask)`.

    The structural mask is never removed or weakened in either path:
    AAM only further restricts structurally-valid actions, and the
    stored rollout masks are replayed verbatim during PPO
    update/re-evaluation, so rollout, PPO update, training, and
    evaluation all observe identical USE_AAM behavior.
    """
    if USE_AAM:
        return pad_mask(
            compute_adaptive_mask(env, agent_name),
            action_dim=action_dim,
        )
    return pad_mask(
        compute_structural_mask(env, agent_name),
        action_dim=action_dim,
    )


# ==========================================================
# Explainability hook (optional)
# ==========================================================

def explain_mask(env, agent_name):
    """
    Same computation as compute_adaptive_mask, but also returns a
    human-readable reason for every action that the contextual layer
    suppressed this timestep.

    Not used in the hot training loop (it builds label strings, which
    the rollout loop doesn't need), but handy for evaluate.py logging,
    a dashboard, or feeding the LLM explanation module described in the
    project architecture ("why couldn't the agent Restore host X?").

    Returns
    -------
    mask : np.ndarray[bool]
        Same as compute_adaptive_mask.
    reasons : dict[int, str]
        {action_index: reason} for every index the contextual layer
        turned from structurally-valid to False.
    """

    structural = np.asarray(env.action_mask(agent_name), dtype=bool)
    meta = describe_actions(env, agent_name)
    labels = env.action_labels(agent_name)

    host_flags = env.get_host_alert_flags(agent_name)
    zone_flags = env.get_zone_alert_flags(agent_name)

    mask = structural.copy()
    reasons = {}
    sleep_index = None

    for i, (action_type, target_host, source_zone) in enumerate(meta):

        if action_type == "Sleep":
            sleep_index = i

        if not mask[i]:
            continue

        if action_type in DISRUPTIVE_ACTIONS:
            if not host_flags.get(target_host, False):
                mask[i] = False
                reasons[i] = (
                    f"{labels[i]}: suppressed -- host '{target_host}' shows "
                    f"no active malicious-process or connection alert."
                )

        elif action_type == "BlockTrafficZone":
            source_key = (
                str(source_zone).lower()
                if source_zone is not None
                else None
            )
            if (
                source_key is not None
                and source_key in zone_flags
                and not zone_flags[source_key]
            ):
                mask[i] = False
                reasons[i] = (
                    f"{labels[i]}: suppressed -- source zone '{source_zone}' "
                    f"shows no flagged host."
                )

    if not mask.any():
        if sleep_index is not None:
            mask[sleep_index] = True
        else:
            mask[0] = True

    return mask, reasons
