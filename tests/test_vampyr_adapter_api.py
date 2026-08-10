"""Interface-level checks that do not require VAMPyR to be installed."""
import inspect

from amdi.vampyr_adapter import make_mra


def test_make_mra_exposes_memory_safety_depth_cap():
    sig = inspect.signature(make_mra)
    assert "max_depth" in sig.parameters
    assert sig.parameters["max_depth"].default == 9
