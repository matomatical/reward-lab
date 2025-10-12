import enum
import functools

import jax
import jax.numpy as jnp
import einops
from PIL import Image

from typing import Callable, Self
from jaxtyping import UInt8, Bool, Float, Array, PRNGKeyArray

import strux


# # # 
# Environment


class Item(enum.IntEnum):
    EMPTY = 0
    SHARDS = 1
    URN = 2


@strux.struct
class State:
    robot_pos: UInt8[Array, "2"]
    bin_pos: UInt8[Array, "2"]
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
    

type PolicyFunction = Callable[
    [Observation],
    Float[Array, "num_actions"]
]
type PolicyValueFunction = Callable[
    [Observation],
    tuple[Float[Array, "num_actions"], Float[Array, ""]]
]


@strux.struct
class Environment:
    init_robot_pos: UInt8[Array, "2"]
    init_items_map: UInt8[Array, "world_size world_size"]
    bin_pos: UInt8[Array, "2"]
    
    
    @property
    def world_size(self) -> int:
        ws, ws_ = self.init_items_map.shape
        return ws


    @jax.jit
    def reset(self: Self) -> State:
        return State(
            robot_pos=self.init_robot_pos,
            items_map=self.init_items_map,
            inventory=jnp.array(Item.EMPTY, dtype=jnp.uint8),
            bin_pos=self.bin_pos,
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
        ).astype(jnp.uint8)
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
                state.bin_pos[0],
                state.bin_pos[1],
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
            state.bin_pos[0],
            state.bin_pos[1],
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
            Sprites.ROBOT,
            Sprites.ROBOT_SHARDS,
            Sprites.ROBOT_URN,
        ))[state.inventory]
        
        # select sprites for other tiles
        tall_sprites = jnp.zeros(
            (self.world_size, self.world_size, 16, 8),
            dtype=jnp.uint8,
        )
        tall_sprites = tall_sprites.at[0, :].set(Sprites.FLOOR)
        tall_sprites = tall_sprites.at[1:, :, 8:].set(Sprites.FLOOR[8:])
        tall_sprites = jnp.where(
            (state.items_map == Item.SHARDS)[:,:,None,None],
            jnp.where(
                Sprites.SHARDS > 0,
                Sprites.SHARDS,
                tall_sprites,
            ),
            tall_sprites,
        )
        tall_sprites = jnp.where(
            (state.items_map == Item.URN)[:,:,None,None],
            jnp.where(
                Sprites.URN > 0,
                Sprites.URN,
                tall_sprites,
            ),
            tall_sprites,
        )
        tall_sprites = tall_sprites.at[
            state.bin_pos[0],
            state.bin_pos[1],
        ].set(jnp.where(
            (Sprites.BIN > 0),
            Sprites.BIN,
            tall_sprites[state.bin_pos[0], state.bin_pos[1]],
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
        image = jnp.array(PALETTE, dtype=jnp.uint8)[image]
        return image


# # # 
# Simple rollouts


@strux.struct
class Transition:
    state: State
    action: UInt8[Array, ""]
    next_state: State


@strux.struct
class Rollout:
    transitions: Transition["num_steps"]


@functools.partial(jax.jit, static_argnames=("policy_fn", "num_steps"))
def collect_rollout(
    env: Environment,
    key: PRNGKeyArray,
    policy_fn: PolicyFunction,
    num_steps: int,
) -> Rollout:
    initial_state = env.reset()
    keys_per_step = jax.random.split(key, num_steps)
    def step(state, key_step):
        obs = env.observe(state)
        action_logits = policy_fn(obs)
        action = jax.random.categorical(
            key=key_step,
            logits=action_logits,
        )
        next_state = env.step(state, action)
        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
        )
        return next_state, transition
    _final_state, transitions = jax.lax.scan(
        step,
        initial_state,
        keys_per_step,
    )
    return Rollout(transitions=transitions)


# # # 
# Annotated rollouts (for RL algorithms)


@strux.struct
class AnnotatedTransition:
    state: State
    obs: Observation
    value_pred: Float[Array, ""]
    action: UInt8[Array, ""]
    action_logits: Float[Array, "num_actions"]
    next_state: State


@strux.struct
class AnnotatedRollout:
    transitions: AnnotatedTransition["num_steps"]
    final_obs: Observation
    final_value_pred: Float[Array, ""]


@functools.partial(jax.jit, static_argnames=("policy_value_fn", "num_steps"))
def collect_annotated_rollout(
    env: Environment,
    key: PRNGKeyArray,
    policy_value_fn: PolicyValueFunction,
    num_steps: int,
) -> AnnotatedRollout:
    initial_state = env.reset()
    keys_per_step = jax.random.split(key, num_steps)
    def step(state, key_step):
        obs = env.observe(state)
        action_logits, value_pred = policy_value_fn(obs)
        action = jax.random.categorical(
            key=key_step,
            logits=action_logits,
        )
        next_state = env.step(state, action)
        transition = AnnotatedTransition(
            state=state,
            obs=obs,
            value_pred=value_pred,
            action=action,
            action_logits=action_logits,
            next_state=next_state,
        )
        return next_state, transition
    final_state, transitions = jax.lax.scan(
        step,
        initial_state,
        keys_per_step,
    )
    final_obs = env.observe(final_state)
    _, final_value_pred = policy_value_fn(final_obs)
    return AnnotatedRollout(
        transitions=transitions,
        final_obs=final_obs,
        final_value_pred=final_value_pred,
    )


# # # 
# Spritesheet

_IMAGE = Image.open("sprites.png")


# palette
_COLORS = {i: rgb for rgb, i in _IMAGE.palette.colors.items()}
PALETTE = jnp.array([_COLORS[i] for i in range(len(_COLORS))])


# sprites
_SPRITESHEET = einops.rearrange(
    jnp.array(_IMAGE),
    '(H h) (W w) -> H W h w',
    h=16,
    w=8,
)


class Sprites:
    FLOOR = _SPRITESHEET[0,0]
    BIN = _SPRITESHEET[0,1]
    SHARDS = _SPRITESHEET[0,2]
    URN = _SPRITESHEET[0,3]
    ROBOT = _SPRITESHEET[0,4]
    ROBOT_SHARDS = _SPRITESHEET[0,5]
    ROBOT_URN = _SPRITESHEET[0,6]


