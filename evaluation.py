import functools
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray

from potteryshop import Environment, State, Action, collect_rollout
from agent import ActorCriticNetwork


type RewardFunction = Callable[[State, Action, State], float]


@jax.jit
def compute_return(
    rewards: Float[Array, "num_steps"],
    discount_rate: float,
) -> float:
    def _accumulate_return(next_step_return, this_step_reward):
        this_step_return = this_step_reward + discount_rate * next_step_return
        return this_step_return, this_step_return
    first_step_return, _per_step_returns = jax.lax.scan(
        _accumulate_return,
        0.0,
        rewards,
        reverse=True,
    )
    return first_step_return


@functools.partial(
    jax.jit,
    static_argnames=["reward_fn", "num_steps", "num_rollouts"],
)
def evaluate_behaviour(
    env: Environment,
    net: ActorCriticNetwork,
    key: PRNGKeyArray,
    reward_fn: RewardFunction,
    num_steps: int = 64,
    num_rollouts: int = 1000,
    discount_rate: float = 0.995,
) -> list[Float[Array, "num_rollouts"]]:
    rollouts = jax.vmap(
        collect_rollout,
        in_axes=(None, 0, None, None),
    )(
        env,
        jax.random.split(key, num_rollouts),
        net.policy,
        num_steps,
    )
    rewards = jax.vmap(jax.vmap(reward_fn))(
        rollouts.transitions.state,
        rollouts.transitions.action,
        rollouts.transitions.next_state,
    )
    returns = compute_return(
        rewards,
        discount_rate,
    )
    return returns
