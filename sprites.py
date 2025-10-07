import numpy as np
import einops

from PIL import Image


_IMAGE = Image.open("sprites.png")


# # # 
# Palette


_COLORS = {i: rgb for rgb, i in _IMAGE.palette.colors.items()}

PALETTE = np.array([
    _COLORS[i] for i in range(len(_COLORS))
])


# # # 
# Sprites


_SPRITESHEET = einops.rearrange(
    np.array(_IMAGE),
    '(H h) (W w) -> H W h w',
    h=16,
    w=8,
)


FLOOR = _SPRITESHEET[0,0]


BIN = _SPRITESHEET[0,1]


SHARDS = _SPRITESHEET[0,2]


URN = _SPRITESHEET[0,3]


ROBOT = _SPRITESHEET[0,4]


ROBOT_SHARDS = _SPRITESHEET[0,5]


ROBOT_URN = _SPRITESHEET[0,6]


