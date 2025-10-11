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

import matthewplotlib as mp
import tyro

import strux
import environment
import agent


# # # 
# Training loop


def main(
    seed: int = 42,
    # environment parameters
    world_size: int = 4,
    num_trash: int = 1,
    num_vases: int = 1,
    # agent parameters
    net_channels: int = 8,
    net_width: int = 16,
    num_conv_layers: int = 4,
    num_dense_layers: int = 2,
    # rollout parameters
    num_steps_per_rollout: int = 64,
    num_parallel_envs: int = 32,
    # training parameters
    learning_rate: float = 0.001,
    num_updates: int = 1024,
    # PPO loss parameters
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
    max_grad_norm: float = 0.5,
    # visualisation
    num_train_steps_per_vis: int = 8,
    animation_path: str = "animation.gif",
):
    key = jax.random.key(seed)
    key_setup, key = jax.random.split(key)


    print("configuring agent...")
    key_net, key_setup = jax.random.split(key_setup)
    net = agent.ActorCriticNetwork.init(
        key=key_net,
        obs_height=world_size,
        obs_width=world_size,
        obs_channels=4,
        obs_features=2,
        net_channels=net_channels,
        net_width=net_width,
        num_conv_layers=num_conv_layers,
        num_dense_layers=num_dense_layers,
        num_actions=len(environment.Action),
    )


    print("configuring optimiser...")
    optimiser = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )
    opt_state = optimiser.init(net)


    print("defining reward function...")
    reward_fn = environment.reward_simple


    print("define training step...")
    @jax.jit
    def train_step(net, opt_state, key_train):
        # generate environments
        key_generate, key_train = jax.random.split(key_train)
        # key_generate = jax.random.key(seed)
        envs = jax.vmap(
            environment.generate,
            in_axes=(0, None, None, None),
        )(
            jax.random.split(key_generate, num_parallel_envs),
            world_size,
            num_trash,
            num_vases,
        )
        # collect experience with current policy...
        key_rollouts, key_train = jax.random.split(key_train)
        rollouts = jax.vmap(
            environment.collect_rollout,
            in_axes=(0,0,None,None),
        )(
            envs,
            jax.random.split(key_rollouts, num_parallel_envs),
            lambda obs: net.forward(obs.grid, obs.vec),
            num_steps_per_rollout,
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
        updates, opt_state = optimiser.update(grads, opt_state, net)
        net = optax.apply_updates(net, updates)
        # metrics
        train_metrics = {
            'loss': loss,
            'return': jax.vmap(compute_average_return, in_axes=(0, None))(
                rewards,
                discount_rate,
            ).mean(),
            **aux,
        }
        return net, opt_state, key_train, train_metrics
    print("total train steps", num_updates)
    print("env steps per train step", num_parallel_envs * num_steps_per_rollout)
    print(
        "total train steps",
        num_updates * num_parallel_envs * num_steps_per_rollout,
    )
    

    print("run training loop...")
    metrics = collections.defaultdict(list)
    key_train, key = jax.random.split(key)
    plot = vis_metrics(metrics=metrics, total=num_updates)
    print(plot)
    for t in range(num_updates):
        net, opt_state, key_train, train_metrics = train_step(
            net,
            opt_state,
            key_train,
        )
        for name, val in train_metrics.items():
            metrics[name].append(val)
        if (t+1) % num_train_steps_per_vis == 0:
            print(-plot, end="")
            plot = vis_metrics(metrics=metrics, total=num_updates)
            print(plot)


    print("making animation...")
    key_animation, key = jax.random.split(key)
    key_generate, key_animation = jax.random.split(key_animation)
    # key_generate = jax.random.key(seed)
    envs = jax.vmap(environment.generate, in_axes=(0, None, None, None))(
        jax.random.split(key_generate, num_parallel_envs),
        world_size,
        num_trash,
        num_vases,
    )
    key_rollouts, key_animation = jax.random.split(key_animation)
    rollouts = jax.vmap(
        environment.collect_rollout,
        in_axes=(0,0,None,None),
    )(
        envs,
        jax.random.split(key_rollouts, num_parallel_envs),
        lambda obs: net.forward(obs.grid, obs.vec),
        num_steps_per_rollout,
    )
    print("rendering rollouts...")
    frames = environment.animate_rollouts(
        env=jax.tree.map(lambda x: x[0], envs),
        rollouts=rollouts,
        grid_width=8,
    )
    print("saving gif...")
    frames = np.array(frames)
    Image.fromarray(frames[0]).save(
        animation_path,
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=100,
        loop=0,
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


# # # 
# Proximal policy optimisation


@jax.jit
def ppo_loss_fn(
    net: agent.ActorCriticNetwork,
    transitions: environment.Transition["num_envs num_steps"],
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
    new_action_logits, new_value_preds = jax.vmap(net.forward)(
        transitions.obs.grid,
        transitions.obs.vec,
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

    # # diagnostics
    # actor_clipfrac = jnp.mean(action_prob_ratios_clipped != action_prob_ratios)
    # actor_approxkl1 = jnp.mean(-action_log_ratios)
    # actor_approxkl3 = jnp.mean((action_prob_ratios - 1) - action_log_ratios)
    # critic_clipfrac = jnp.mean(value_diffs != value_diffs_clipped)

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
            # 'actor-clip': actor_clipfrac,
            # 'critic-clip': critic_clipfrac,
            # 'actor-kl1': actor_approxkl1,
            # 'actor-kl2': actor_approxkl3,
        }
    )


# # # 
# Evaluating rollouts




# # # 
# Visualisation


def vis_metrics(
    metrics: dict[str, list],
    total: int,
) -> mp.plot:
    if not metrics:
        return mp.blank()
    plots = [
        mp.axes(
            plot=mp.scatter(
                (np.arange(len(data)), np.array(data), "cyan"),
                xrange=(0,total),
                width=34,
                height=12,
            ),
            xlabel="train steps",
            title=key,
        ) for key, data in metrics.items()
    ]
    return mp.wrap(*plots)


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)

