# # # 
# Rewards


type RewardFunction = Callable[[State, Action, State], float]


def reward_simple(
    state: State,
    action: Action,
    next_state: State,
) -> float:
    pickup_reward = (
        (state.items_map[state.robot_pos[0], state.robot_pos[1]] == Item.TRASH)
        & (state.inventory == Item.EMPTY)
        & (action == Action.PICKUP)
    )
    putdown_reward = (
        (state.goal_pos[0] == state.robot_pos[0])
        & (state.goal_pos[1] == state.robot_pos[1])
        & (state.inventory == Item.TRASH)
        & (action == Action.PUTDOWN)
    )
    return (pickup_reward + putdown_reward).astype(float)


def reward_potential(
    state: State,
    action: Action,
    next_state: State,
    discount_factor: float,
) -> float:
    putdown_reward = (
        (state.goal_pos[0] == state.robot_pos[0])
        & (state.goal_pos[1] == state.robot_pos[1])
        & (state.inventory == Item.TRASH)
        & (action == Action.PUTDOWN)
    )
    potential0 = (state.inventory == Item.TRASH)
    potential1 = (next_state.inventory == Item.TRASH)
    shape_term = discount_factor * potential1 - potential0
    return putdown_reward + shape_term


def reward_no_crash_vase(
    state: State,
    action: Action,
    next_state: State,
) -> float:
    # TODO: Quiz: Are there any situations where it will still be optimal to
    # crash a vase?
    r = (next_state.robot_pos[0], next_state.robot_pos[1])
    crashed_vase = (
        (state.items_map[r] == Item.VASE)
        & (next_state.items_map[r] == Item.TRASH)
    )
    return -crashed_vase


def combined_reward(
    state: State,
    action: Action,
    next_state: State,
    discount_factor: float,
) -> float:
    rp = reward_potential(
        state,
        action,
        next_state,
        discount_factor=discount_factor,
    )
    rv = reward_no_crash_vase(
        state,
        action,
        next_state,
    )
    return rp + rv


# TODO: Yeah I think this is a good exercise. Just have to get over the "if"
# JAX hurdle and finish building out the implementation.


