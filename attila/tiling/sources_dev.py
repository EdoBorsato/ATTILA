"""
Deprecated alias of `attila.tiling.sources`.

The development variant that lived here became the main module. This
alias keeps existing driver scripts working and will be removed in a
later version: import from `attila.tiling.sources` instead.
"""
from attila.tiling.sources import *  # noqa: F401,F403
from attila.tiling.sources import (  # noqa: F401
    _get_bbox_arr,
    _get_bbox_image,
)
