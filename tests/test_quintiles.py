import pytest
import numpy as np
import xpipe.likelihood.quintiles as quintiles


def test_quintile_init():
    qqq = quintiles.QuintileExplorer(None, None, None)

    quintile_limits = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
    assert quintile_limits == qqq._quintiles

