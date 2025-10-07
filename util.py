import io
import math

import numpy as np
import einops
from PIL import Image
import jax

import ipywidgets as widgets
from IPython.display import display
import plotly.subplots
import plotly.graph_objects

import environment


def display_rollout(
    env: environment.Environment,
    rollout: environment.Rollout,
    upscale: int = 6,
):
    frames = environment.animate_rollouts(
        env=env,
        rollouts=jax.tree.map(lambda x: x[None], rollout), # + batch dimension
        grid_width=1,
    )
    frames = einops.repeat(
        frames,
        't h w rgb -> t (h h2) (w w2) rgb',
        h2=upscale,
        w2=upscale,
    )
    display_gif(frames)


def display_rollouts(
    envs: environment.Environment["n"],
    rollouts: environment.Rollout["n"],
    grid_width: int,
    upscale: int = 3,
):
    frames = environment.animate_rollouts(
        env=jax.tree.map(lambda x: x[0], envs),
        rollouts=rollouts,
        grid_width=grid_width,
    )
    frames = einops.repeat(
        frames,
        't h w rgb -> t (h h2) (w w2) rgb',
        h2=upscale,
        w2=upscale,
    )
    display_gif(frames)


def display_gif(frames):
  frames = np.array(frames)
  with io.BytesIO() as buffer:
    Image.fromarray(frames[0]).save(
        buffer,
        format="gif",
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=100,
        loop=0,
    )
    animation_widget = widgets.Image(
        value=buffer.getvalue(),
        format='gif',
    )
    display(animation_widget)


class InteractivePlayer:
    def __init__(self, env: environment.Environment):
        # Initialise state
        self.env = env
        self.state = env.reset()

        # Image display widget
        self.image_widget = widgets.Image(value=b'', format='png')
        self._render()

        # Controls
        btn_up = widgets.Button(description="Up")
        btn_left = widgets.Button(description="Left")
        btn_down = widgets.Button(description="Down")
        btn_right = widgets.Button(description="Right")
        btn_pickup = widgets.Button(description="Pickup")
        btn_putdown = widgets.Button(description="Drop")
        btn_reset = widgets.Button(description="Reset", button_style='warning')

        btn_up.on_click(lambda b: self._action(environment.Action.UP))
        btn_left.on_click(lambda b: self._action(environment.Action.LEFT))
        btn_down.on_click(lambda b: self._action(environment.Action.DOWN))
        btn_right.on_click(lambda b: self._action(environment.Action.RIGHT))
        btn_pickup.on_click(lambda b: self._action(environment.Action.PICKUP))
        btn_putdown.on_click(lambda b: self._action(environment.Action.PUTDOWN))
        btn_reset.on_click(lambda b: self._reset())

        # Combine into UI
        self.ui = widgets.HBox([
            self.image_widget,
            widgets.VBox(
                [btn_up, widgets.HBox([btn_left, btn_right]), btn_down],
                layout=widgets.Layout(align_items='center'),
            ),
            widgets.VBox([btn_pickup, btn_putdown, btn_reset]),
        ], layout=widgets.Layout(align_items='center'))

    def _reset(self):
        self.state = self.env.reset()
        self._render()

    def _action(self, action: environment.Action):
        self.state = self.env.step(self.state, action)
        self._render()

    def _render(self):
        image_array = self.env.render(self.state)
        image_array = image_array.repeat(8, axis=0).repeat(8, axis=1)
        image = Image.fromarray(np.array(image_array))
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            self.image_widget.value = buffer.getvalue()
    
    def _ipython_display_(self):
        display(self.ui)


class LiveSubplots:
    def __init__(
        self,
        metric_names: list,
        total_steps: int,
        num_cols: int = 3,
    ):
        # Create plot
        num_rows = math.ceil(len(metric_names) / num_cols)
        fig = plotly.subplots.make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=metric_names,
            vertical_spacing=0.06,
            horizontal_spacing=0.03,
        )
        fig.update_layout(
            height=350 * num_rows,
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        fig.update_xaxes(range=[0, total_steps])
        for i, metric in enumerate(metric_names):
            fig.add_trace(
                plotly.graph_objects.Scatter(
                    name=metric,
                    x=[],
                    y=[],
                    line=dict(width=1, color='#636EFA'),
                  ),
                row=1 + (i // num_cols),
                col=1 + (i % num_cols),
            )
        self.fig = plotly.graph_objects.FigureWidget(fig)
        
        # State tracking
        self.data = {name: (i, [], []) for i, name in enumerate(metric_names)}

        # Display the widget
        display(self.fig)

    def log(self, t: int, logs: dict):
        for name, value in logs.items():
            self.data[name][1].append(t)
            self.data[name][2].append(value)

    def refresh(self):
        with self.fig.batch_update():
            for name, (i, xs, ys) in self.data.items():
                self.fig.data[i].x = xs
                self.fig.data[i].y = ys
