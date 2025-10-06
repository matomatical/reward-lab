import enum
import functools

import numpy as np
import jax
import jax.numpy as jnp
import einops
from PIL import Image

from typing import Callable, Self
from jaxtyping import UInt8, Bool, Float, Array, PRNGKeyArray

import strux
import sprites


# # # 
# Environment


class Item(enum.IntEnum):
    EMPTY = 0
    SHARDS = 1
    URN = 2


@strux.struct
class State:
    """
    State of a grid world.
    """
    robot_pos: UInt8[Array, "2"]
    goal_pos: UInt8[Array, "2"]
    items_map: UInt8[Array, "world_size world_size"] # Item[world_size,world_size]
    inventory: UInt8[Array, ""]


@strux.struct
class Observation:
    grid: Bool[Array, "world_size world_size 4"] # world map
    vec: Bool[Array, "2"] # inventory


class Action(enum.IntEnum):
    WAIT = 0 # do nothing
    UP = 1 # move up
    LEFT = 2 # move left
    DOWN = 3 # move down
    RIGHT = 4 # move right
    PICKUP = 5 # pick up item
    PUTDOWN = 6 # drop held item


@strux.struct
class Environment:
    """
    A grid world with a particular layout.
    """
    init_robot_pos: UInt8[Array, "2"]
    init_items_map: UInt8[Array, "world_size world_size"]
    goal_pos: UInt8[Array, "2"]
    
    
    @property
    def world_size(self) -> int:
        ws, ws_ = self.init_items_map.shape
        return ws


    @jax.jit
    def reset(self: Self) -> State:
        """
        Initial state for the given layout.
        """
        return State(
            robot_pos=self.init_robot_pos,
            items_map=self.init_items_map,
            inventory=jnp.array(Item.EMPTY, dtype=jnp.uint8),
            goal_pos=self.goal_pos,
        )

    
    @jax.jit
    def step(self: Self, state: State, action: Action) -> State:
        # move robot
        deltas = jnp.array((
            ( 0,  0), # (do nothing)
            (-1,  0), # move up
            ( 0, -1), # move left
            (+1,  0), # move down
            ( 0, +1), # move right
            ( 0,  0), # (pick up item)
            ( 0,  0), # (drop held item)
        ), dtype=jnp.int8)
        try_robot_pos = state.robot_pos + deltas[action]
        new_robot_pos = jnp.clip(
            try_robot_pos,
            min=0,
            max=self.world_size-1,
        )
        state = state.replace(
            robot_pos=new_robot_pos,
        )
        robot_pos = (state.robot_pos[0], state.robot_pos[1]) # for indexing
        
        # collide with items
        on_item = state.items_map[robot_pos]
        new_item = jnp.where(
            (on_item == Item.URN),
            Item.SHARDS,
            on_item,
        )
        state = state.replace(
            items_map=state.items_map.at[robot_pos].set(new_item),
        )

        # pick up item
        do_pickup = (action == Action.PICKUP) & (state.inventory == Item.EMPTY)
        on_item = state.items_map[robot_pos]
        new_inventory = jnp.where(
            do_pickup,
            on_item,
            state.inventory,
        )
        new_item = jnp.where(
            do_pickup,
            Item.EMPTY,
            on_item,
        )
        state = state.replace(
            inventory=new_inventory,
            items_map=state.items_map.at[robot_pos].set(new_item),
        )

        # put down item
        on_item = state.items_map[robot_pos]
        do_putdown = (action == Action.PUTDOWN) & (on_item == Item.EMPTY)
        new_inventory = jnp.where(
            do_putdown,
            Item.EMPTY,
            state.inventory,
        )
        new_item = jnp.where(
            do_putdown,
            state.inventory,
            on_item,
        )
        state = state.replace(
            inventory=new_inventory,
            items_map=state.items_map.at[robot_pos].set(new_item),
        )

        # dispose of items placed in bin
        state = state.replace(
            items_map=state.items_map.at[
                state.goal_pos[0],
                state.goal_pos[1],
            ].set(Item.EMPTY),
        )

        return state


    @jax.jit
    def observe(self: Self, state: State) -> Observation:
        # grid data (positional stuff)
        grid = jnp.zeros((self.world_size, self.world_size, 4), dtype=bool)
        grid = grid.at[
            state.robot_pos[0],
            state.robot_pos[1],
            0,
        ].set(True)
        grid = grid.at[
            state.goal_pos[0],
            state.goal_pos[1],
            1,
        ].set(True)
        grid = grid.at[:, :, 2].set(state.items_map == Item.SHARDS)
        grid = grid.at[:, :, 2].set(state.items_map == Item.URN)
        # feature data (inventory status)
        vec = jnp.zeros((2,), dtype=bool)
        vec = vec.at[0].set(state.inventory == Item.SHARDS)
        vec = vec.at[1].set(state.inventory == Item.URN)
        # done
        return Observation(grid=grid, vec=vec)


    @jax.jit
    def render(self: Self, state: State) -> UInt8[Array, "height width rgb"]:
        def stack(a, b):
            return jnp.where(b > 0, b, a)
        # choose avatar
        robot_sprite = jnp.stack((
            sprites.ROBOT,
            sprites.ROBOT_SHARDS,
            sprites.ROBOT_URN,
        ))[state.inventory]
        
        # select sprites for other tiles
        tall_sprites = jnp.zeros(
            (self.world_size, self.world_size, 16, 8),
            dtype=jnp.uint8,
        )
        tall_sprites = tall_sprites.at[0, :].set(sprites.FLOOR)
        tall_sprites = tall_sprites.at[1:, :, 8:].set(sprites.FLOOR[8:])
        tall_sprites = jnp.where(
            (state.items_map == Item.SHARDS)[:,:,None,None],
            jnp.where(
                sprites.SHARDS > 0,
                sprites.SHARDS,
                tall_sprites,
            ),
            tall_sprites,
        )
        tall_sprites = jnp.where(
            (state.items_map == Item.URN)[:,:,None,None],
            jnp.where(
                sprites.URN > 0,
                sprites.URN,
                tall_sprites,
            ),
            tall_sprites,
        )
        tall_sprites = tall_sprites.at[
            state.goal_pos[0],
            state.goal_pos[1],
        ].set(jnp.where(
            (sprites.GOAL > 0),
            sprites.GOAL,
            tall_sprites[state.goal_pos[0], state.goal_pos[1]],
        ))
        tall_sprites = tall_sprites.at[
            state.robot_pos[0],
            state.robot_pos[1],
        ].set(jnp.where(
            (robot_sprite > 0),
            robot_sprite,
            tall_sprites[state.robot_pos[0], state.robot_pos[1]],
        ))
        # pack the sprites together
        bottoms = tall_sprites[:, :, 8:, :]
        tops = tall_sprites[:, :, :8, :]
        tiles = jnp.zeros(
            (self.world_size+1, self.world_size, 8, 8),
            dtype=jnp.uint8,
        )
        tiles = tiles.at[1:, :, :, :].set(bottoms)
        tiles = tiles.at[:-1, :, :, :].set(jnp.where(
            tops > 0,
            tops,
            tiles[:-1],
        ))
        # form into 2d image
        image = einops.rearrange(tiles, 'H W h w -> (H h) (W w)')

        # apply color palette
        image = jnp.array(sprites.PALETTE, dtype=jnp.uint8)[image]
        return image


@functools.partial(
    jax.jit,
    static_argnames=("world_size", "num_trash", "num_vases"),
)
def generate(
    key: PRNGKeyArray,
    world_size: int,
    num_trash: int,
    num_vases: int,
) -> Environment:
    """
    Randomly construct an environment layout.
    """
    # fixed goal pos
    goal_pos = jnp.zeros((2,), dtype=jnp.uint8)

    # list of possible item/robot coordinates
    coords = einops.rearrange(
        jnp.indices((world_size, world_size)),
        'c h w -> c (h w)',
    )
    # exclude (0,0) (used for goal)
    coords = coords[:, 1:]

    # sample robot and item positions without replacement
    num_positions = 1 + num_trash + num_vases
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
        items_pos[0, :num_trash],
        items_pos[1, :num_trash],
    ].set(Item.SHARDS)
    items_map = items_map.at[
        items_pos[0, num_trash:],
        items_pos[1, num_trash:],
    ].set(Item.URN)
    
    return Environment(
        init_robot_pos=robot_pos,
        init_items_map=items_map,
        goal_pos=goal_pos,
    )


# # # 
# Rollouts


@strux.struct
class Transition:
    state: State
    obs: Observation
    value_pred: Float[Array, ""]
    action: UInt8[Array, ""]
    action_logits: Float[Array, "num_actions"]
    next_state: State


@strux.struct
class Rollout:
    transitions: Transition["num_steps"]
    final_obs: Observation
    final_value_pred: Float[Array, ""]


@functools.partial(jax.jit, static_argnames=("actor_critic", "num_steps"))
def collect_rollout(
    env: Environment,
    key: PRNGKeyArray,
    actor_critic: Callable[
        [Observation],
        tuple[Float[Array, "num_actions"], Float[Array, ""]]
    ],
    num_steps: int,
) -> Rollout:
    def step(carry, _):
        key, state = carry
        key_step, key = jax.random.split(key)
        obs = env.observe(state)
        action_logits, value_pred = actor_critic(obs)
        action = jax.random.categorical(
            key=key_step,
            logits=action_logits,
        )
        next_state = env.step(state, action)
        transition = Transition(
            state=state,
            obs=obs,
            value_pred=value_pred,
            action=action,
            action_logits=action_logits,
            next_state=next_state,
        )
        return (key, next_state), transition
    (key, final_state), transitions = jax.lax.scan(
        step,
        (key, env.reset()),
        length=num_steps,
    )
    final_obs = env.observe(final_state)
    _, final_value_pred = actor_critic(final_obs)
    return Rollout(
        transitions=transitions,
        final_obs=final_obs,
        final_value_pred=final_value_pred,
    )


# # # 
# Testing


@functools.partial(jax.jit, static_argnames=("grid_width",))
def animate_rollouts(
    env: Environment, # or environments...
    rollouts: Rollout["n"],
    grid_width: int,
) -> UInt8[Array, "num_steps+1 H*h+H+1 W*w+W+1 rgb"]:
    n = jax.tree.leaves(rollouts)[0].shape[0]
    assert (n % grid_width) == 0

    # full state sequence
    all_states = jax.tree.map(
        lambda xs, xs_: jnp.concatenate((xs, xs_[:, [-1]]), axis=1),
        rollouts.transitions.state,
        rollouts.transitions.next_state,
    )
    
    # render images for all states
    images = jax.vmap(jax.vmap(env.render))(all_states)

    # rearrange into a (padded) grid of renders
    images = jnp.pad(
        images,
        pad_width=(
            (0, 0), # env
            (0, 0), # steps
            (0, 1), # height
            (0, 1), # width
            (0, 0), # channel
        ),
    )
    grid = einops.rearrange(
        images,
        '(H W) t h w rgb -> t (H h) (W w) rgb',
        W=grid_width,
    )
    grid = jnp.pad(
        grid,
        pad_width=(
            (0, 4), # time
            (1, 0), # height
            (1, 0), # width
            (0, 0), # channel
        ),
    )
    return grid


def rollouts(
    world_size: int = 8,
    num_trash: int = 8,
    num_vases: int = 8,
    episode_horizon: int = 256,
    num_parallel_envs: int = 32,
    animation_path: str = "animation.gif",
    seed: int = 42,
):
    key = jax.random.key(seed=seed)
    env = generate(
        key,
        world_size=world_size,
        num_trash=num_trash,
        num_vases=num_vases,
    )
    
    print("generating environments...")
    envs = jax.vmap(generate, in_axes=(0,None,None,None))(
        jax.random.split(key, num_parallel_envs),
        world_size,
        num_trash,
        num_vases,
    )
    def actor_critic(obs):
        return jnp.zeros(len(Action)), 0.
    print("collecting rollouts...")
    rollouts = jax.jit(jax.vmap(
        collect_rollout,
        in_axes=(0,0,None,None),
    ))(
        envs,
        jax.random.split(num_parallel_envs),
        actor_critic,
        episode_horizon,
    )
    print("rendering rollouts...")
    frames = animate_rollouts(
        env=env, # TODO: maybe envs?
        rollouts=rollouts,
        grid_width=8,
    )
    print("saving gif...")
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
    num_trash: int = 8,
    num_vases: int = 8,
    seed: int = 42,
):
    import readchar
    import matthewplotlib as mp

    key = jax.random.key(seed=seed)
    env = generate(
        key,
        world_size=world_size,
        num_trash=num_trash,
        num_vases=num_vases,
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


if __name__ == "__main__":
    import tyro
    tyro.extras.subcommand_cli_from_dict({
        'rollouts': rollouts,
        'play': play,
    })
