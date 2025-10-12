import functools
import numpy as np
import jax
import jax.numpy as jnp
import einops
from jaxtyping import PRNGKeyArray, Float, UInt8, Array

import tyro
from PIL import Image
import readchar
import matthewplotlib as mp

from potteryshop import Environment, Action, Item
from potteryshop import Observation, Rollout, collect_rollout
from util import animate_rollouts

    
def main():
    tyro.extras.subcommand_cli_from_dict({
        'rollouts': rollouts,
        'play': play,
    })


def rollouts(
    world_size: int = 6,
    num_shards: int = 4,
    num_urns: int = 5,
    episode_horizon: int = 64,
    num_parallel_envs: int = 32,
    animation_path: str = "animation.gif",
    seed: int = 42,
):
    key = jax.random.key(seed=seed)
    print("generating environments...")
    key_generate, key = jax.random.split(key)
    envs = jax.vmap(generate, in_axes=(0, None, None, None))(
        jax.random.split(key, num_parallel_envs),
        world_size,
        num_shards,
        num_urns,
    )
    def random_policy(obs: Observation) -> Float[Array, "num_actions"]:
        return jnp.zeros(len(Action))
    print("collecting rollouts...")
    key_rollouts, key = jax.random.split(key)
    rollouts = jax.vmap(
        collect_rollout,
        in_axes=(0,0,None,None),
    )(
        envs,
        jax.random.split(key_rollouts, num_parallel_envs),
        random_policy,
        episode_horizon,
    )
    print("rendering rollouts...")
    prototypical_env = jax.tree.map(lambda x: x[0], envs)
    frames = animate_rollouts(
        env=prototypical_env,
        rollouts=rollouts,
        grid_width=8,
    )
    print(f"saving gif as {animation_path!r}...")
    frames = np.array(frames)
    Image.fromarray(frames[0]).save(
        animation_path,
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=5,
        loop=0,
    )


def play(
    world_size: int = 8,
    num_shards: int = 8,
    num_urns: int = 8,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)
    env = generate(
        key,
        world_size=world_size,
        num_shards=num_shards,
        num_urns=num_urns,
    )
    
    print("your turn... wasd move / q pickup / e drop / r reset / z quit")
    state = env.reset()
    print(mp.image(env.render(state)))
    while True:
        key = readchar.readkey()
        if key == 'z':
            break
        elif key == 'r':
            state = env.reset()
        else:
            action = {
                'w': Action.UP,
                'a': Action.LEFT,
                's': Action.DOWN,
                'd': Action.RIGHT,
                'q': Action.PICKUP,
                'e': Action.PUTDOWN,
            }[key]
            state = env.step(state, action)
        plot = mp.image(env.render(state))
        print(f"{-plot}{plot}")


# # # 
# Helper functions


@functools.partial(
    jax.jit,
    static_argnames=("world_size", "num_shards", "num_urns"),
)
def generate(
    key: PRNGKeyArray,
    world_size: int,
    num_shards: int,
    num_urns: int,
) -> Environment:
    # fixed goal pos
    bin_pos = jnp.zeros((2,), dtype=jnp.uint8)

    # list of possible item/robot coordinates
    coords = einops.rearrange(
        jnp.indices((world_size, world_size)),
        'c h w -> c (h w)',
    )
    # exclude (0,0) (used for goal)
    coords = coords[:, 1:]

    # sample robot and item positions without replacement
    num_positions = 1 + num_shards + num_urns
    all_positions = jax.random.choice(
        key=key,
        a=coords,
        shape=(num_positions,),
        axis=1,
        replace=False,
    )
    robot_pos = all_positions[:, 0]
    items_pos = all_positions[:, 1:]

    # create item map
    items_map = jnp.zeros((world_size, world_size), dtype=jnp.uint8)
    items_map = items_map.at[
        items_pos[0, :num_shards],
        items_pos[1, :num_shards],
    ].set(Item.SHARDS)
    items_map = items_map.at[
        items_pos[0, num_shards:],
        items_pos[1, num_shards:],
    ].set(Item.URN)
    
    return Environment(
        init_robot_pos=robot_pos,
        init_items_map=items_map,
        bin_pos=bin_pos,
    )


if __name__ == "__main__":
    main()
