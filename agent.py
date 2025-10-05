"""
RL agent network.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import einops
from jaxtyping import Array, Float, Bool, PRNGKeyArray

import strux


@strux.struct
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> AffineTransform:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights=jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        biases=jnp.zeros(num_outputs)
        return AffineTransform(weights=weights, biases=biases)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights + self.biases


@functools.partial(strux.struct, static_fieldnames=("stride_size", "pad_same"))
class Convolution:
    kernel: Float[Array, "channels_out channels_in kernel_size kernel_size"]
    stride_size: int
    pad_same: bool

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "channels_in",
            "channels_out",
            "kernel_size",
            "stride_size",
            "pad_same",
        ),
    )
    def init(
        key: PRNGKeyArray,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride_size: int,
        pad_same: bool,
    ) -> Convolution:
        num_inputs = channels_in * kernel_size**2
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return Convolution(
            kernel=jax.random.uniform(
                key=key,
                shape=(channels_out, channels_in, kernel_size, kernel_size),
                minval=-bound,
                maxval=+bound,
            ),
            stride_size=stride_size,
            pad_same=pad_same,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "height_in width_in channels_in"],
    ) -> Float[Array, "height_out width_out channels_out"]:
        x_1hwc = einops.rearrange(x, 'h w c -> 1 h w c')
        y_1hwc = jax.lax.conv_general_dilated(
            lhs=x_1hwc,
            rhs=self.kernel,
            window_strides=(self.stride_size, self.stride_size),
            padding="SAME" if self.pad_same else "VALID",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        return y_1hwc[0]


@strux.struct
class ActorCriticNetwork:
    conv0: Convolution
    convs: Convolution["num_conv_layers-1"]
    dense0: AffineTransform
    denses: AffineTransform["num_dense_layers-1"]
    actor_head: AffineTransform
    critic_head: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "obs_height",
            "obs_width",
            "obs_channels",
            "obs_features",
            "net_channels",
            "net_width",
            "num_conv_layers",
            "num_dense_layers",
            "num_actions",
        ),
    )
    def init(
        key: PRNGKeyArray,
        obs_height: int,
        obs_width: int,
        obs_channels: int,
        obs_features: int,
        net_channels: int,
        net_width: int,
        num_conv_layers: int,
        num_dense_layers: int,
        num_actions: int,
    ):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        # initialise convolutional layers
        conv0 = Convolution.init(
            key=k1,
            channels_in=obs_channels,
            channels_out=net_channels,
            kernel_size=3,
            stride_size=1,
            pad_same=True,
        )
        convs = jax.vmap(
            Convolution.init,
            in_axes=(0, None, None, None, None, None),
        )(
            jax.random.split(k2, num_conv_layers-1),
            net_channels,
            net_channels,
            3,
            1,
            True,
        )
        # initialise dense layers
        dense0 = AffineTransform.init(
            key=k3,
            num_inputs=obs_height * obs_width * net_channels + obs_features,
            num_outputs=net_width,
        )
        denses = jax.vmap(
            AffineTransform.init,
            in_axes=(0, None, None),
        )(
            jax.random.split(k4, num_dense_layers-1),
            net_width,
            net_width,
        )
        # initialise critic / actor heads
        actor_head = AffineTransform.init(
            key=k5,
            num_inputs=net_width,
            num_outputs=num_actions,
        )
        critic_head = AffineTransform.init(
            key=k6,
            num_inputs=net_width,
            num_outputs=1,
        )
        return ActorCriticNetwork(
            conv0=conv0,
            convs=convs,
            dense0=dense0,
            denses=denses,
            actor_head=actor_head,
            critic_head=critic_head,
        )

    def forward(
        self: Self,
        obs_grid: Bool[Array, "obs_height obs_width obs_channels"],
        obs_vec: Bool[Array, "obs_features"],
    ) -> tuple[
        Float[Array, "num_actions"],
        Float[Array, ""],
    ]:
        # case
        obs_grid = obs_grid.astype(float)
        obs_vec = obs_vec.astype(float)
        # embed observation grid part with residual CNN
        x = self.conv0.forward(obs_grid)
        x = jax.nn.relu(x)
        x, _ = jax.lax.scan(
            lambda x, conv: (x + jax.nn.relu(conv.forward(x)), None),
            x,
            self.convs,
        )
        # further compute with residual dense network
        x = jnp.concatenate((jnp.ravel(x), obs_vec))
        x = self.dense0.forward(x)
        x = jax.nn.relu(x)
        x, _ = jax.lax.scan(
            lambda x, dense: (x + jax.nn.relu(dense.forward(x)), None),
            x,
            self.denses,
        )
        # apply action/value heads
        action_logits = self.actor_head.forward(x)
        value_pred = self.critic_head.forward(x)[0]
        return action_logits, value_pred


if __name__ == "__main__":
    key = jax.random.key(seed=42)
    
    # initialisation
    net = ActorCriticNetwork.init(
        key=key,
        obs_height=8,
        obs_width=8,
        obs_channels=4,
        obs_features=2,
        net_channels=16,
        net_width=16,
        num_conv_layers=8,
        num_dense_layers=4,
        num_actions=7,
    )
    print(net)
    print(strux.size(net))
    
    # forward pass
    obs_grid = jnp.ones((8,8,4))
    obs_vec = jnp.ones(2)
    action_logits, value_pred = net.forward(obs_grid, obs_vec)
    print(action_logits)
    print(value_pred)
