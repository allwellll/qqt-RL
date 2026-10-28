"""Reward-package adapter for the preregistered isolated JAX ablation."""
from __future__ import annotations

import jax.numpy as jnp


DELIVERY_REWARD = 2.0
KILL_REWARD = 16.0
DEATH_PENALTY = 12.0
STRICT_PBRS_COMPONENT = "route_shaped_reward"
CONTROL_COMPONENTS = (
    "control_entry_reward",
    "control_residual_reward",
)
KEEP_COMPONENTS = frozenset((
    "global_safety",
    "settled_kill",
    "transport_capture",
    *CONTROL_COMPONENTS,
    STRICT_PBRS_COMPONENT,
))


def output_component_names(component_names):
    """Validate the frozen source schema; no labels are added or reordered."""
    names = tuple(component_names)
    if not names or names[-1] != "combined_total":
        raise ValueError("reward component schema must end in combined_total")
    if len(set(names)) != len(names):
        raise ValueError("reward component schema contains duplicate labels")
    return names


def _component_indices(component_names):
    names = output_component_names(component_names)
    indices = {name: index for index, name in enumerate(names)}
    required = ("global_safety", "settled_kill", "transport_capture",
                *CONTROL_COMPONENTS, STRICT_PBRS_COMPONENT, "combined_total")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"reward schema missing required components: {missing}")
    pbrs_channels = [name for name in names
                     if name in {"route_shaped_reward", "strict_pbrs_reward",
                                 "potential_shaped_reward"}]
    if pbrs_channels != [STRICT_PBRS_COMPONENT]:
        raise ValueError("treatment must use exactly one strict PBRS reward channel")
    return names, indices


def select_reward(arm, legacy_reward, info, parts, component_names, *,
                  kill_event=None, coefficient=1.0):
    """Select the legacy reward or the preregistered minimal event package.

    Minimal coefficients reuse the existing retention event scales: capture +2,
    surviving kill +16, death -12, and causal control entry/residual. The
    route channel is the already scaled strict PBRS delta; all other shaping and
    item-payment channels are zeroed. A zero coefficient is an exact legacy
    fallback so the adapter cannot perturb the control arm.
    """
    if arm not in {"legacy", "minimal"}:
        raise ValueError(f"unknown reward arm: {arm}")
    if arm == "legacy" or coefficient == 0.0:
        zero = jnp.zeros_like(legacy_reward)
        return legacy_reward, {"total": zero}, parts

    names, indices = _component_indices(component_names)
    component_values = jnp.asarray(parts)
    if component_values.shape[:-1] != jnp.asarray(legacy_reward).shape:
        raise ValueError("reward component tensor must match the reward shape")
    if component_values.shape[-1] != len(names):
        raise ValueError("reward component tensor width must match its fixed schema")
    for required in ("capture", "death"):
        if required not in info:
            raise ValueError(f"native event info is missing {required}")
    if kill_event is None:
        if "surviving_kill" not in info:
            raise ValueError("native event info is missing surviving_kill")
        kill_event = info["surviving_kill"]

    dtype = jnp.result_type(legacy_reward, component_values)
    delivery = jnp.asarray(info["capture"], dtype=dtype) * DELIVERY_REWARD
    kill = jnp.asarray(kill_event, dtype=dtype) * KILL_REWARD
    control = sum(jnp.asarray(component_values[..., indices[name]], dtype=dtype)
                  for name in CONTROL_COMPONENTS)
    death = -jnp.asarray(info["death"], dtype=dtype) * DEATH_PENALTY
    pbrs = jnp.asarray(component_values[..., indices[STRICT_PBRS_COMPONENT]], dtype=dtype)
    selected = (delivery + kill + control + death + pbrs) * coefficient

    # Preserve the fixed component schema while preventing excluded legacy
    # reward terms from appearing as if they contributed to the treatment.
    selected_parts = jnp.zeros_like(component_values)
    for name in KEEP_COMPONENTS:
        if name in indices and name not in {"combined_total"}:
            value = component_values[..., indices[name]]
            if name == "global_safety":
                value = death
            elif name == "settled_kill":
                value = kill
            elif name == "transport_capture":
                value = delivery
            elif name == "route_shaped_reward":
                value = pbrs
            selected_parts = selected_parts.at[..., indices[name]].set(value * coefficient)
    selected_parts = selected_parts.at[..., indices["combined_total"]].set(selected)
    terms = dict(delivery=delivery * coefficient, kill=kill * coefficient,
                 control=control * coefficient, death=death * coefficient,
                 pbrs=pbrs * coefficient, total=selected)
    return selected, terms, selected_parts


def bind_reward_arm(ops_module, control_credit_module, arm):
    """Install an arm only on an isolated runner; return the prior callable."""
    parent = ops_module.combined_reward

    def combined_reward(states, next_states, info, aux, settled,
                        coefficient=1.0, bridge_config=None):
        legacy, new_aux, parts, metrics = parent(
            states, next_states, info, aux, settled, coefficient, bridge_config)
        if arm == "legacy":
            return legacy, new_aux, parts, metrics
        kill_events = control_credit_module.parent_reward_events(states, info)
        selected, terms, selected_parts = select_reward(
            arm, legacy, info, parts, ops_module.COMPONENTS,
            kill_event=kill_events["surviving_kill"], coefficient=coefficient)
        metrics = dict(metrics, expert_reward_ablation_terms=terms)
        return selected, new_aux, selected_parts, metrics

    ops_module.combined_reward = combined_reward
    return parent
