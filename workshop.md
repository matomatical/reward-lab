---
title: Specification gaming and goal misgeneralisation in grid worlds
author: Matthew Farrugia-Roberts
date: Monday, October 13^th^
---

Welcome to the first lab for
  [AI Safety and Alignment](https://robots.ox.ac.uk/~fazl/aisaa/),
MT 2025.

Structure of today's lab:

0. Installation---in which we install dependencies and introduce JAX.
1. Agents and environments---in which we introduce two core elements of the
   elements of reinforcement learning framework and explore a simple grid-world
   environment.
2. Reward functions---in which we introduce the third element of reinforcement
   learning, and train an agent in our simple environment.
3. Specification gaming---in which we explore the consequences of failing to
   account for 'creative' ways of optimising our reward function.
4. Generalisation---in which we train in a distribution of environments in the
   hope that the agent will learn to behave correctly in situations it has
   never seen.
5. Goal misgeneralisation---in which we explore the consequences of failing to
   account for ambiguity in the specification of our goals.

The supporting code for today's lab can be found at
  [github.com/matomatical/reward-lab](https://github.com/matomatical/reward-lab).

Part 0: Installation
====================

Colab runtime
-------------

For parts 1--3, the default CPU run-time is sufficient. When you reach parts 4
and 5, it may make sense to switch over to a TPU (fastest) or GPU (fast)
run-time and repeat the installation and imports, as training on procedurally
generated grid-worlds requires deeper networks and longer training runs than
individual grid-worlds.

Install dependencies
--------------------

```python
# TODO: !commands for setting up instance.
```

Import libraries
----------------

```python
# TODO: Imports
```

Hello, JAX!
-----------

This workshop uses [JAX](https://jax.readthedocs.io/), a Python deep learning
framework that is a modern alternative to PyTorch and TensorFlow. Much modern
reinforcement learning research, including at Oxford, takes place in JAX,
making it worth learning. However, today's activities don't require much
familiarity with JAX. For our purposes, note the following:

* JAX programs operate on JAX arrays, which are basically immutable NumPy
  arrays. With a few exceptions, where you would write
    `np.function(numpy_array)` or `numpy_array.method()`,
  you can instead write
    `jnp.function(jax_array)` or `jax_array.method()`.

* Due to immutability, JAX has a slightly more verbose way of managing random
  state compared to other frameworks. It is generally necessary to pass around
  a `key` object corresponding to a node in a tree of random states, and to 

* One cool feature of JAX is that, once you write a function with JAX arrays,
  you can use `jax.vmap(function)` to get a version that works with arrays that
  have extra batch dimensions. This means we normally don't have to worry about
  managing batch dimensions ourselves at all. We'll see `jax.vmap` a couple of
  times throughout the code in this notebook.

* Another cool feature of JAX is that, once you write a function with JAX
  arrays, you can use `jax.jit(function)` to get a version that is just-in-time
  compiled and optimised to run efficiently on your specific CPU, GPU, or TPU.
  The resulting speed boost is one of the main selling points of JAX. We will
  see `jax.jit` in action today, giving us faster feedback loops than if we had
  written the notebook using PyTorch.

* JAX's powerful function transformations like `jax.jit` and `jax.vmap` come at
  a cost. In order for functions to be easily batchable and to run well on GPUs
  and TPUs, the computational graph of each function needs to be made "uniform"
  in that the shapes and operations involved don't depend on the values in the
  input arrays. For example:

  * We can't use `if` statements that depend on array values.
  * We can't use `while` loops whose termination condition depends on
  * We can't use some NumPy operations, such as boolean indexing, that result
    in arrays whose shapes depend on the values of other arrays.

  JAX provides its own primitives and patterns for getting around these issues
  and writing the kind of expressive Python programs we are used to.

Throughout this notebook, there are examples and hints on the necessary JAX
knowledge to get you through what limited use of JAX we need for the exercises,
and you can always ask your tutors or an LLM for help if you are facing a weird
JAX error.

After today, if you are interested in learning more about JAX,
  [see](https://github.com/n2cholas/awesome-jax)
  [these](https://far.in.net/hijax)
  [resources](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html).


Part 1: Agents and environments
===============================

In reinforcement learning (RL), we model a sequential decision-making task with
two main entities:

1. An **environment,** which models the state of the world, keeping track of
   how the world changes in response to the actions of an agent (speaking of
   which...)
2. An **agent,** who repeatedly receives information about the current state of
   the environment, chooses an action, and executes that action in the
   environment.

In this part, we'll familiarise ourselves with these concepts.

Note: The framework of reinforcement learning would not be complete without
discussing a third entity, the **reward function.** We'll come to reward
functions in part 2.

Markov decision process formalism
---------------------------------

A popular formalism for defining environments is that of the **Markov decision
process (MDP).** A (rewardless) MDP is a tuple
  $(S, A, \iota, \tau)$
where:

* $S$ is a set of environment states,
* $A$ is the set of actions available to the agent from any state,
* $\iota \in \Delta(S)$ is a distribution of initial states (the states the
  environment starts in before the agent takes any actions), and
* $\tau : S \times A \to \Delta(S)$ is a conditional transition distribution:
  given the current state $s$ and agent action $a$, $\tau(s,a)$ gives the
  probability for the environment to transition into each possible next state.

(Note: This definition omits the reward function and discount factor usually
included in the definition of an MDP. We'll return to defining those in the
next part.)

The Markov decision process earns its name from the fact that the transitions
as defined here satisfy a Markov property, whereby they are independent of the
path taken to get to the state before the transition.

An environment represented by an MDP is often paired with an agent represented
by an action-selection **policy** of the form
  $$
    \pi : S \to \Delta(A),
  $$
where $\pi(s)$ represents the probability that the agent will take each
possible action given the state $s$.
Sometimes, the policy is defined to take as input only a certain subset of
state information (called an *observation*), or perhaps a sequence of
states (or observations), having the effect of imbuing the agent with a memory.

Pottery shop: A simple environment
----------------------------------

Here is a picture of an environment called "pottery shop".

![](environment.png)

Pottery shop is an example of a grid-world environment, where everything plays
out on a finite grid of positions (in this case, a 6 by 6 grid).

This grid world contains various objects:

* Urns---the products of the pottery shop.
* Shards---some urns have been broken, leaving behind piles of shards.
* A bin---there is a bin in the corner that stores shards.
* A robot---there is a small blue robot who can move around the grid, pick up
  shards, carry them around, and drop them (e.g., into the bin).
  If the robot crashes into one of the urns, the urn will break, creating a new
  pile of shards.

The pottery shop environment is implemented in the source file
`environment.py`. Some relevant snippets of code are as follows.

### Environment layout

Pottery shop actually refers to a family of environments of different sizes and
with different initial configurations of objects. We represent such an
environment as an `Environment` dataclass as follows:

```python
@strux.struct
class Environment:
    init_robot_pos: UInt8[Array, "2"]
    init_items_map: UInt8[Array, "world_size world_size"]
    bin_pos: UInt8[Array, "2"]
```

In particular, the fields are as follows:

* `init_robot_pos` contains the row and column grid coordinates of the spawn
  position of the robot.
* `bin_pos` similarly contains the row and column grid coordinates of the spawn
  position of the bin.
* `init_items_map` is an N by N array where N is the size of the grid world.
  The contents of the array map to the presence of shards or urns in the
  respective grid squares.

The following enumeration type explains how to interpret the numbers in
`init_items_map`.

```python
class Item(enum.IntEnum):
    EMPTY = 0
    SHARDS = 1
    URN = 2
```

The coordinates in `init_robot_pos` and `bin_pos` should range from `0` to
`world_size-1`, and the values in `init_items_map` should all be `0`, `1`, or
`2`.

> ### `jaxtyping`
> 
> This code snippet uses type annotations of the form `UInt8[Array, "shape"]`.
> `UInt8` and `Array` were imported from `jaxtyping` above (along with some
> other array types, namely `Float` and `Bool`). These type annotations
> describe the intended type and shape of the JAX arrays used in this code:
> 
> * `init_robot_pos` and `bin_pos` are JAX arrays of unsigned 8-bit integers
>   with shape `(2,)`.
> * `init_items_map` is a JAX array of unsigned 8-bit integers with shape
>   `(world_size, world_size)`.
> 
> Some other examples:
> 
> * `Float[Array, "height width 3"]` indicates an array of floats with shape
>   `(height, width, 3)`, for example for representing some RGB image data.
> * `Bool[Array, ""]` indicates a zero-dimensional array of booleans, that is,
>   a boolean scalar.
> 
> These annotations are not currently type checked, and should be treated like
> comments (in particular, they may contain typos). Nevertheless, we hope they
> array-manipulating code easier to follow. Feel free to include these
> annotations in your own code, too.

> ### `@strux.struct`
> 
> This code snippet uses the `strux.struct` wrapper, defined in `strux.py`.
> Basically, this is equivalent to the built-in Python wrapper
> `dataclasses.dataclass(frozen=True)`, but with some extra functionality to
> make the dataclasses integrate nicely with JAX function transformations.


### Environment state

The environment object encodes the initial configuration of the pottery shop.
But once we start taking actions, the state will change. At any time, the
current state of the grid world is represented by the following dataclass.

```python
@strux.struct
class State:
    robot_pos: UInt8[Array, "2"]
    bin_pos: UInt8[Array, "2"]
    items_map: UInt8[Array, "world_size world_size"]
    inventory: UInt8[Array, ""]
```

The fields `robot_pos`, `bin_pos`, and `items_map` are the dynamic versions of
`init_robot_pos`, `bin_pos`, and `init_items_map` from the environment
configuration.

What's new is `inventory`, an integer scalar that represents what kind of item
the robot is carrying. This is initially `Item.EMPTY`, but changes to
`Item.SHARDS` if the robot picks up a pile of shards (and changes back to
`Item.EMPTY` if the robot drops the shards).

### Agent actions

The agent is responsible for controlling the robot as it moves around the grid.
In each interaction, the agent sees the current state, and then chooses what
the robot should do from the following options.

```python
class Action(enum.IntEnum):
    WAIT = 0 # do nothing
    UP = 1 # move up
    LEFT = 2 # move left
    DOWN = 3 # move down
    RIGHT = 4 # move right
    PICKUP = 5 # pick up item
    PUTDOWN = 6 # drop held item
```

### Environment methods
    
So much for defining the data types involved, the actual implementation of the
environment logic takes place inside the methods of the environment class.
The two most important methods are the following:

* `def reset(self: Environment) -> State`: Initialises a `State`.

* `def step(self: Environment, state: State, action: Action) -> State`: Takes
  the current state and the agent's actions and returns the resulting state of
  the environment (for example, moving the robot, updating its inventory,
  smashing urns).

Task 1: Explore pottery shop
----------------------------

Your first task is simply to instantiate and explore a pottery shop
environment. Complete the following sub-tasks:

1. Create an `Environment` object, including a world size of at least 4, at
   least one pile of shards, and at least one urn.

* Interact with the environment using the built-in simulator
  `InteractivePlayer`.

  
Note: Sub-task 1 involves specifying JAX arrays for each of the arguments of
the `Environment` constructor. To create a JAX array, you can use
`jnp.array(...)` where the input is a list, a nested list, or a numpy array.
Other numpy-like methods for creating arrays should work as well, such as
`jnp.zeros`, `jnp.ones`, `jnp.arange` etc.).

```python
env = # ...
```

```python
InteractivePlayer(env)
```

### Solution

Of course, many environment layouts are permissible. Here is the one depicted
above:

```python
env = Environment(
    init_robot_pos=jnp.array((1,2), dtype=jnp.uint8),
    init_items_map=jnp.array((
      (0,0,0,0,2,2),
      (0,1,0,0,0,2),
      (0,0,0,0,0,0),
      (0,1,1,0,0,2),
      (0,0,0,0,2,2),
      (2,0,1,0,2,2),
    ), dtype=jnp.uint8),
    bin_pos=jnp.array((0,0), dtype=jnp.uint8),
)
```

Bonus task: Framing questions
-----------------------------

If you have time and interest, consider the following questions:

1. Could you write down the pottery shop environment in the MDP formalism? How
   would this compare to the Python implementation? What definitions or methods
   correspond to the set of states, the set of actions, the initial state
   distribution, and the transition distribution?

2. Consider the joint system of you and your computer, while you are clicking
   the buttons that control the pottery shop robot. Decompose this system into
   an environment and an agent. What are the states, and what are the actions?

3. Is the division between agent and environment that you came up with for the
   previous question unique? Are there other ways you could divide up the
   system into an environment and an agent? How many can you think of?

4. When you interacted with the environment, were you acting out a memoryless
   policy, or would representing your policy require histories of environment
   states as inputs? Does it depend on where you draw the boundary between the
   agent and the environment?

Part 2: Reward functions
========================

We now return to our description of the reinforcement learning framework.

Reward functions
----------------

Once we have an environment with a set of states $S$ and an agent with a set of
actions $A$, a **reward function** is defined as a function
$$
  r : S \times A \times S \to \mathbb{R}
$$
that maps individual interactions between the environment and the agent to
scalar reward values.

Note that here the input triple $(s,a,s') \in S \times A \times S$ represents
an environment state ($s$), the agent's choice of action ($a \sim \pi(s)$), and
the resulting successor state ($s' \sim \tau(s,a)$).
Sometimes reward functions are defined on states alone ($r: S \to \mathbb{R}$),
state-action pairs ($r:S \times A \to \mathbb{R}$), or state pairs ($r : S
\times S \to \mathbb{R}$), but the most general form includes the above three
elements.

Maximising expected return
--------------------------

Given a reward function and a **trajectory** of states and actions,
$$
  s_0, a_0, s_1, a_1, \ldots
$$
we define the **return** $R$ as the discounted cumulative sum of rewards:
$$
  R(s_0, a_0, \ldots) = \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t, s_{t+1})
$$
where $\gamma \in (0,1)$ is a discount factor that controls how much more to
value rewards received earlier versus later in time.

A typical formulation of the goal of reinforcement learning algorithms is,
given an environment, a reward function, and a discount factor, find a policy
that **maximises expected return** (taking the expectation over the
stochasticity in the initial state distribution, the transition distribution,
and the policy itself).

The reward hypothesis 
---------------------

Reward functions play a central role in the discipline of reinforcement
learning. Their centrality is driven by the following hypothesis, called the
**reward hypothesis:**

> That all of what we mean by goals and purposes can be well thought of as
> maximization of the expected value of the cumulative sum of a received scalar
> signal (reward).
> 
> ---Richard Sutton, [The reward hypothesis](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/rewardhypothesis.html)

The reward hypothesis essentially claims that, regardless of what kind of
purposes we want a system to fulfil, we can sensibly formulate that behaviour
as the maximum expected return for *some* reward function. Once we find the
right reward function, then we just need to apply a reinforcement learning
algorithm to find a policy with the behaviour we desire.

While there is some debate in the field as to the extent to which the reward
hypothesis is true in its maximal form (that maximisation of scalar returns
covers *all* goals and purposes), it is certainly true that:

* There are some AI problems where finding the right reward function has
  allowed us to specify behaviours that were infeasible to specify imperatively
  (e.g. by programming, as in classical software engineering) or even
  declaratively (e.g. by generating labelled examples, as in supervised
  learning), with the most notable example being computer Go playing and recent
  examples of frontier reasoning models.

* This possibility has driven a substantial portion of interest in the field of
  reinforcement learning, with some seeing reinforcement learning as a key
  component of the likely path to human-level and super-human AI in the future.

Of course, it remains to find the right reward function!

Task 2: Interpreting a reward function
--------------------------------------

Your second task is to study a reward function and consider the behaviour it
incentivises. Here is the reward function:

```python
def reward1(state: State, action: Action, next_state: State) -> float:
    item_below_robot = state.items_map[
        state.robot_pos[0],
        state.robot_pos[1],
    ]
    pickup_reward = (
        (item_below_robot == Item.SHARD)
        & (state.inventory == Item.EMPTY)
        & (action == Action.PICKUP)
    ).astype(float)
    dispose_reward = (
        (state.bin_pos[0] == state.robot_pos[0])
        & (state.bin_pos[1] == state.robot_pos[1])
        & (state.inventory == Item.SHARD)
        & (action == Action.PUTDOWN)
    ).astype(float)
    total_reward = pickup_reward + dispose_reward
    return total_reward
```

Note that `&` represents elementwise 'and' for (NumPy or) JAX arrays.

Your task is to answer the following questions:

1. Describe the kinds of transitions for which this reward function returns
   `1.0` vs `0.0`.

2. What kinds of qualitative behaviours do you think a reward designer who came
   up with this reward function is trying to incentivise?

3. What kinds of qualitative behaviours maximise return subject to this reward
   function? Are they the same as the previous answer?

Cleaning up shop
----------------

Next, let's apply apply a reinforcement learning algorithm to see what
behaviours the agent learns given this reward function.

We'll need a neural network parametrisation of a policy we can train by
gradient descent. The following code defines a small CNN-based policy
network.

```python
key = jax.random.key(seed=42)
key_init, key = jax.random.split(key)
net = ActorCriticNetwork.init(
    key=key_init,
    obs_height=env.world_size,
    obs_width=env.world_size,
    obs_features=2,
    net_channels=16,
    net_width=32,
    num_conv_layers=2,
    num_dense_layers=1,
    num_actions=len(Action),
)
```

The library `ppo.py` provides a function `train_agent` that implements a
reinforcement learning algorithm (proximal policy optimisation) to train a
policy network given an environment and a reward function, plus various other
hyper-parameters.

Let's call that function on the initialised policy network, your
manually-instantiated environment from task 1, and the above reward function,
to train the policy for a small number of steps.

```
key_train, key = jax.random.split(key)
net = train_agent(
  net=net,
  env=env,
  reward=reward1,
  key=key_train,
  num_updates=512,
)
```

> ### `jax.random.key` and `jax.random.split`
> 
> The above code is our first use of JAX's quirky PRNG state management
> pattern.
> 
> In other frameworks, the first step to drawing reproducible random samples
> would be to initialise a PRNG object, to be passed into each random function.
> Each of these functions would draw samples from the PRNG whenever needed,
> mutating the internal state of the PRNG so that every additional sample is
> independent.
> 
> Mutability complicates the kinds of function transformations JAX specialises
> in. So in JAX, we need a way to manage a PRNG's state without mutating an
> object. The JAX solution is essentially for the user to explicitly create a
> copy of the PRNG (with a modified state) and pass different parts of the
> state around to each function. Actually, instead of advancing the state
> linearly, it's more accurate to say that we *split* the state into multiple
> independent child states. Each function gets its own independent branch of
> the phylogenetic tree of PRNG states, which it can continue to split and pass
> around to its liking.
> 
> We see this in the above examples. First, `key = jax.random.key(seed=42)`
> initialises a root random state. Then,
>   `key_init, key = jax.random.split(key)`
> forks it into two children:
> 
> * `key_init`---a branch for passing into the network initialisation function.
> * `key` (again)---since we don't need the root state after we have split it,
>   it is idiomatic to reassign one of the children to the old variable if we
>   want to split again later in the current scope.
> 
> In the next cell, we repeat this pattern, sending `key_train` into the
> training function, and reassigning `key` to a fresh child for subsequent use.
> 
> One thing to watch out for when using JAX in notebooks is that the JAX idiom
> of reassigning to `key` and the notebook idiom of repeated execution of cells
> do not mix well. If you re-run the previous code block, you will get
> different training calls each time as `key` is reassigned to the output of
> the split inside the block. This is fine for today, but for robust
> reproducibility, one should either use fresh names for each child key or have
> each cell restart from its own seed.


Task 3: Interpreting an agent's behaviour
-----------------------------------------

The next task is to study the behaviour of the agent we learned, and see to
what extent it matches the behaviour we intended or expected when we designed
this reward function. The following code samples and animates some trajectories
from the learned network. Your task is to run the code and study the policies,
discerning if there are any difference between:

1. The behaviour the designer of the reward function likely intended;
2. The behaviour you expected after studying the reward function; and
3. The actual behaviour observed.

```python
key_rollout = jax.random.key(seed=1)
rollout = collect_rollout(
    env=env,
    key=key_rollout,
    actor_critic=net.forward,
    num_steps=64,
)
display_rollout(rollout)
```

Note: Change the seed above to see trajectories with different random results when
sampling actions from the policy.

Questions:

1. What qualitative behaviours do you observe?

2. Are there discrepancies between the behaviour you observe and the
   intended/expected behaviour? If so, list them.

Part 3: Specification gaming
============================

> ### Note: Finish task 3 before reading further.

If all has gone to plan, you should be looking at an example of specification
gaming (also known as reward hacking), in which the reinforcement learning
algorithm found a better way to increase expected return than the class of
valid solutions the reward designer had in mind.

The two kinds of discrepancies we expect to see in this example are:

1. The agent prefers to repeatedly pick up and drop shards compared to picking
   them up once and taking them to the bin; and
2. The agent is willing to break urns to find new shards to pick up.

In this part, we will redesign the reward function to prevent these unintended
behaviours from being incentivised.

The inverse reward hypothesis
-----------------------------

Recall the reward hypothesis from above. While this hypothesis is usually
invoked in the context of designing an agent using reinforcement learning by
identifying an appropriate reward, it applies more broadly as stated.

In particular, the reward hypothesis applies to the behaviour of existing
agents. Suppose we have a system whose behaviour fulfils some purpose. Then (by
the reward hypothesis), that behaviour can be well thought of as maximising
expected return for *some* reward function.

One could call the above corollary the **inverse reward hypothesis,** following
  [Stuart Russell (1998)](https://dl.acm.org/doi/10.1145/279943.279964), also
  [Andrew Ng and Stuart Russell (2000)](https://dl.acm.org/doi/10.5555/645529.657801),
who introduced the problem of *inverse reinforcement learning* (that of
taking a policy and extracting from it a reward function for which that policy
maximises expected return).

Task 4: Quantifying reward hacking
----------------------------------

We won't trouble ourselves with inverse reinforcement learning algorithms
today. Instead, let's just take a step in this direction by quantifying the
misbehaviour of our agents using reward functions that incentivise each
problematic behaviour. We won't use these reward functions for training, but we
can use them to measure to what extent we are inadvertently training for these
behaviours.

Your next task is to write one reward function that measures the extent to
which an agent is engaging in each problematic behaviour:

1. First, write a reward function `reward_drop` that assigns `+1` every time
   the agent drops a shard (other than in the bin).
2. Second, write a reward function `reward_break` that assigns `+1` every time
   the agent breaks an urn.

```python
"TODO"
```

Having written these reward functions, we can get a quantitative signal about
whether our policy is engaging in these misbehaviours. Unlike a reward function
that correctly incentivises the *intended* behaviour, we ideally want these
reward functions *to be minimised.*

<!--Hey I should look into whether reward function orthogonality and projection
like this has been studied.-->

The following code collects a large number of rollouts using the trained agent,
computes the return from these rollouts under the new reward functions, and
plots the results as histograms.

```python
@functools.partial(
    jax.jit,
    static_argnames=["rewards", "num_steps", "num_rollouts"],
)
def evaluate_behaviour(
    env: Environment,
    net: ActorCriticNetwork,
    key: PRNGKeyArray,
    reward_fns: list[RewardFunction],
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
        net.forward,
        num_steps,
    )
    return_vecs = []
    for reward_fn in reward_fns:
        rewards = jax.vmap(jax.vmap(reward_fn))(
            rollouts.transitions.state,
            rollouts.transitions.action,
            rollouts.transitions.next_state,
        )
        returns = jax.vmap(compute_average_return, in_axes=(0, None))(
            rewards,
            discount_rate,
        )
        return_vecs.append(returns)
    return return_vecs
```

```python
return_vecs = evaluate_behaviour(
    key=jax.random.key(seed=seed),
    env=env,
    net=net,
    rewards=[reward1, reward_drop, reward_break],
)
 # TODO: make histograms.
```

> ### `jax.jit` and `jax.vmap`
>
> This is our first use of just-in-time compilation and vectorised mapping with
> JAX. It's not necessary to understand the details right now, but suffice it
> to say these transformations are making the code shorter and making it run
> faster. If you are interested, take a look at the JAX tutorials on these
> topics:
> 
> * [Just in time compilation](https://docs.jax.dev/en/latest/jit-compilation.html).
> * [Automatic vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html).
> 

Potential shaping: Avoiding cycles
----------------------------------

Regarding the first problematic behaviour (repeatedly picking up and dropping
shards), one solution would be to remove the reward for picking up shards
altogether. The intention is for the shards to end up in the bin, not in the
inventory for their own sake, anyway.

However, sometimes rewarding instrumental goals can assist the agent's learning
process. When we know that something is instrumentally useful for achieving a
task, it would be nice if we could incorporate that knowledge into the reward.

<!--Is there an interpretation of potential shaping in terms of priors? When I
do Q learning or V learning is there a sense in which I am potential shaping
with the latest value function or something like that?-->

The only problem is that incorporating these kinds of hints into the reward is
a delicate operation---as we have seen, it can lead to misspecification and
reward hacking if it becomes possible to satisfy the hint without completing
the original task!

Fortunately, there is a sure-fire scheme for adding a so-called **shaping**
term to a reward function without introducing such loops. This is a method
called **potential shaping,** and it works as follows:

1. Formulate the information as a function of states, $\Phi : S -> \mathbb{R}$,
   called a potential function.
2. When we transition from state $s$ to state $s'$, add reward $\gamma
   \Phi(s')$ to represent gaining the potential from being in state $s'$, but
   also *subtract* reward $\Phi(s)$ to represent *losing* the potential from
   *leaving* state $s$.
3. This helps the agent learn to steer towards states with 'high potential',
   without giving it a long-term incentive to stay there for the sake of this
   potential---the potential should eventually either be lost (if the agent
   leaves those states without achieving return) or actualised (if the policy
   leaves the states while achieving return).

TODO: What, if any, assumptions should we have placed on the potential
function? Check the paper. I think things should be bounded so that they
converge.

Bonus task: Cancelling potentials
---------------------------------

Those of you who are theoretically inclined may like to try this optional
exercise.

Let
  $\Phi : S \to \mathbb{R}$ be a bounded potential function,
  $r : S \times A \times S \to \mathbb{R}$ a bounded reward function, and
  $\gamma \in (0,1)$ a discount rate.
Define a shaped reward function
  $r' : S \times A \times S \to \mathbb{R}$
such that for all triples $s,a,s'$ we have
$$
  r'(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s).
$$

Given a trajectory $s_0, a_0, s_1, a_1, \ldots$, calculate the return under the
two reward functions $r$ and $r'$ and show that they differ by an additive
constant that depends only on $s_0$.

Conclude that ordering on policies induced by the expected return under $r$ is
that same as the ordering on policies induced by the expected return under
$r'$.

TODO: Check that this is accurate.

Task 5: Implementing potential shaping
--------------------------------------

Using potential shaping, write a reward function `reward_shaped` that still
incentivises the robot to pick up shards but not to put them back down again.
Hint: Consider a potential function that looks at the contents of the
inventory.

```python
def reward_shaped(...):
    # TODO
```

Task 6: Negative rewards
------------------------

TODO: Describe the approach of giving negative rewards to penalise hitting
urns. Ask them to implement this. Basically they have already implemented this
above, so they are just negating it. Make it a bit more nuanced by explaining
how the agent can get most of that reward back by binning the shards some
number of steps later, so we should probably make it -2.

```
TODO: Solution
```

TODO: Make the second part of the task the following question. If the reward
for breaking an urn is -2, is there ever any situation where breaking an urn
leads to higher return than leaving it intact?

Task 7: Fixing the specification
--------------------------------

Your next task is to bring together the previous three tasks to eliminate
specification gaming:

1. Combine the rewards from tasks 5 and 6 into a single new reward function,
   `reward2.`

2. Train a new network, using the same environment, agent architecture, and
   hyper-parameters as last time, but this time using the new reward function.

3. Inspect some rollouts, manually and/or by using your evaluation reward
   function probes, to confirm that the agent now behaves as intended.

```
TODO
```

Part 4: Generalisation
======================

Sketch (TODO: Flesh out)
------------------------

Introduce:

* What it means for a policy to generalise.

Warmup:

* Here is a procedural generator that generates pottery shop environments. We
  want to train a policy in one of these and then have it generalise to more of
  them.
* This code calls a function that trains a policy given an environment
  generator. Run it.

Task:

* Test the policy on different shop layouts using this function. Can it handle
  them?
* In each case, predict what you think will be the behaviour, and then evaluate
  it and see the behaviour matches your prediction.
* Can you qualitatively characterise the kinds of environments that the policy
  can handle and the kinds that it does not? What about the ways in which it
  fails?

Part 5: Goal misgeneralisation
==============================

Sketch (TODO: Flesh out)
------------------------

Introduce:

* Goal misgeneralisation, proxy goals.

Task:

* What is the proxy goal here? Can you write a reward function that describes
  the misgeneral behaviour?
* Train a new policy on the more general generator and see if it works to
  remove the misgeneralisation.

Conclusion
==========

TODO: Summarise the workshop's key takeaway.
