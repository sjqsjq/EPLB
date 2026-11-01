# Auto-imported by `site` at every interpreter startup (incl. multiprocessing
# spawn children, since PYTHONPATH propagates). Shadows the editable-installed
# sglang with our patched tree at /workspace/sglang-oeplb/python WITHOUT touching
# the shared pristine tree. Only removes the *sglang* editable finder; leaves
# megatron_core / slime finders intact.
import sys
_SHADOW = "/workspace/sglang-oeplb/python"
def _fm(f):
    return (getattr(getattr(f, "__class__", type(f)), "__module__", "") or "").lower()
try:
    sys.meta_path = [f for f in sys.meta_path if "sglang" not in _fm(f)]
    if _SHADOW not in sys.path:
        sys.path.insert(0, _SHADOW)
except Exception:
    pass
