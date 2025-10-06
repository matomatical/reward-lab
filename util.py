import io
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import einops
from PIL import Image

import environment


def display_gif(frames):
  frames = np.array(frames)
  frames = einops.repeat(
    frames,
    't h w rgb -> t (h h2) (w w2) rgb',
    h2=2,
    w2=2,
  )
  with io.BytesIO() as buffer:
    Image.fromarray(frames[0]).save(
        buffer,
        format="gif",
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=5,
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

