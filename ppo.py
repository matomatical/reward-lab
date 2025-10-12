"""
Reinforcement learning with proximal policy optimisation and generalised
advantage estimation in a simple maze environment.

Needs tuning.
"""

import collections
import functools

import numpy as np
import jax
import jax.numpy as jnp
import einops
import optax
from PIL import Image

from jaxtyping import Array, Float, Bool, PRNGKeyArray

import strux
from potteryshop import Environment, collect_annotated_rollout, AnnotatedTransition
from evaluation import RewardFunction, compute_return
from agent import ActorCriticNetwork


@functools.partial(
    jax.jit,
    static_argnames=["reward_fn","optimiser","num_rollouts","num_env_steps"],
)
def ppo_train_step(
    key: PRNGKeyArray,
    net: ActorCriticNetwork,
    env: Environment,
    reward_fn: RewardFunction,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    num_rollouts: int = 32,
    num_env_steps: int = 64,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
) -> tuple[
    ActorCriticNetwork,
    optax.OptState,
    dict[str, float],
]:
    # collect experience with current policy...
    key_rollouts, key = jax.random.split(key)
    rollouts = jax.vmap(
        collect_annotated_rollout,
        in_axes=(None, 0, None, None),
    )(
        env,
        jax.random.split(key_rollouts, num_rollouts),
        net.policy_value,
        num_env_steps,
    )
    # compute rewards
    rewards = jax.vmap(jax.vmap(reward_fn))(
        rollouts.transitions.state,
        rollouts.transitions.action,
        rollouts.transitions.next_state,
    )
    # estimate advantages on the collected experience...
    advantages = jax.vmap(
        generalised_advantage_estimation,
        in_axes=(0, 0, 0, None, None),
    )(
        rewards,
        rollouts.transitions.value_pred,
        rollouts.final_value_pred,
        eligibility_rate,
        discount_rate,
    )
    # update the policy on the collected experience...
    (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
        net,
        transitions=rollouts.transitions,
        advantages=advantages,
        discount_rate=discount_rate,
        proximity_eps=proximity_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
    )
    updates, optimiser_state = optimiser.update(grads, optimiser_state, net)
    net = optax.apply_updates(net, updates)
    # metrics
    train_metrics = {
        'loss': loss,
        'return': jax.vmap(compute_return, in_axes=(0, None))(
            rewards,
            discount_rate,
        ).mean(),
        **aux,
    }
    return net, optimiser_state, train_metrics


@functools.partial(
    jax.jit,
    static_argnames=["reward_fn", "optimiser", "num_env_steps"],
)
def ppo_train_step_multienv(
    key: PRNGKeyArray,
    net: ActorCriticNetwork,
    envs: Environment["num_rollouts"],
    reward_fn: RewardFunction,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    num_env_steps: int = 64,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
) -> tuple[
    ActorCriticNetwork,
    optax.OptState,
    dict[str, float],
]:
    # collect experience with current policy...
    num_rollouts = jax.tree.leaves(envs)[0].shape[0]
    key_rollouts, key = jax.random.split(key)
    rollouts = jax.vmap(
        collect_annotated_rollout,
        in_axes=(0, 0, None, None),
    )(
        envs,
        jax.random.split(key_rollouts, num_rollouts),
        net.policy_value,
        num_env_steps,
    )
    # compute rewards
    rewards = jax.vmap(jax.vmap(reward_fn))(
        rollouts.transitions.state,
        rollouts.transitions.action,
        rollouts.transitions.next_state,
    )
    # estimate advantages on the collected experience...
    advantages = jax.vmap(
        generalised_advantage_estimation,
        in_axes=(0, 0, 0, None, None),
    )(
        rewards,
        rollouts.transitions.value_pred,
        rollouts.final_value_pred,
        eligibility_rate,
        discount_rate,
    )
    # update the policy on the collected experience...
    (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
        net,
        transitions=rollouts.transitions,
        advantages=advantages,
        discount_rate=discount_rate,
        proximity_eps=proximity_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
    )
    updates, optimiser_state = optimiser.update(grads, optimiser_state, net)
    net = optax.apply_updates(net, updates)
    # metrics
    train_metrics = {
        'loss': loss,
        'return': jax.vmap(compute_return, in_axes=(0, None))(
            rewards,
            discount_rate,
        ).mean(),
        **aux,
    }
    return net, optimiser_state, train_metrics


# # # 
# PPO loss function


@jax.jit
def ppo_loss_fn(
    net: ActorCriticNetwork,
    transitions: AnnotatedTransition["num_envs num_steps"],
    advantages: Float[Array, "num_envs num_steps"],
    discount_rate: float,
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> tuple[float, dict[str, float]]:
    # reshape the data to have one batch dimension
    transitions, advantages = jax.tree.map(
        lambda x: einops.rearrange(
            x,
            "n_envs n_steps ... -> (n_envs n_steps) ...",
        ),
        (transitions, advantages),
    )
    batch_size = advantages.size

    # run network to get latest predictions
    new_action_logits, new_value_preds = jax.vmap(net.policy_value)(
        transitions.obs,
    ) # -> float[batch_size, 7], float[batch_size]

    # actor loss
    new_action_logprobs = jax.nn.log_softmax(
        new_action_logits,
        axis=1,
    )
    new_chosen_logprobs = new_action_logprobs[
        jnp.arange(batch_size),
        transitions.action,
    ]
    old_action_logprobs = jax.nn.log_softmax(
        transitions.action_logits,
        axis=1,
    )
    old_chosen_logprobs = old_action_logprobs[
        jnp.arange(batch_size),
        transitions.action,
    ]
    action_log_ratios = new_chosen_logprobs - old_chosen_logprobs
    action_prob_ratios = jnp.exp(action_log_ratios)
    action_prob_ratios_clipped = jnp.clip(
        action_prob_ratios,
        1 - proximity_eps,
        1 + proximity_eps,
    )
    std_advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    actor_loss = -jnp.minimum(
        std_advantages * action_prob_ratios,
        std_advantages * action_prob_ratios_clipped,
    ).mean()

    # critic loss
    value_diffs = new_value_preds - transitions.value_pred
    value_diffs_clipped = jnp.clip(
        value_diffs,
        -proximity_eps,
        proximity_eps,
    )
    new_value_preds_proximal = transitions.value_pred + value_diffs_clipped
    targets = transitions.value_pred + advantages
    critic_loss = jnp.maximum(
        jnp.square(new_value_preds - targets),
        jnp.square(new_value_preds_proximal - targets),
    ).mean() / 2

    # entropy regularisation term
    per_step_entropy = - jnp.sum(
        jnp.exp(new_action_logprobs) * new_action_logprobs,
        axis=1,
    )
    average_entropy = jnp.mean(per_step_entropy)

    # diagnostics
    actor_clipfrac = jnp.mean(action_prob_ratios_clipped != action_prob_ratios)
    actor_approxkl1 = jnp.mean(-action_log_ratios)
    actor_approxkl3 = jnp.mean((action_prob_ratios - 1) - action_log_ratios)
    critic_clipfrac = jnp.mean(value_diffs != value_diffs_clipped)

    # total loss
    total_loss = (
        actor_loss
        + critic_coeff * critic_loss
        - entropy_coeff * average_entropy
    )
    return (
        total_loss,
        {
            'loss-actor': actor_loss,
            'loss-critic': critic_loss,
            'entropy': average_entropy,
            'actor-clip': actor_clipfrac,
            'critic-clip': critic_clipfrac,
            'actor-kl1': actor_approxkl1,
            'actor-kl2': actor_approxkl3,
        }
    )


# # # 
# Generalised advantage estimation


@jax.jit
def generalised_advantage_estimation(
    rewards: Float[Array, "num_steps"],
    values: Float[Array, "num_steps"],
    final_value: float,
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Array, "num_steps"]:
    # reverse scan through num_steps axis
    initial_gae_and_next_value = (0., final_value)
    transitions = (rewards, values)
    def _gae_reverse_step(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        reward, this_value = transition
        gae = (
            reward
            - this_value
            + discount_rate * (next_value + eligibility_rate * gae)
        )
        return (gae, this_value), gae
    _final_carry, gaes = jax.lax.scan(
        _gae_reverse_step,
        initial_gae_and_next_value,
        transitions,
        reverse=True,
    )
    return gaes


