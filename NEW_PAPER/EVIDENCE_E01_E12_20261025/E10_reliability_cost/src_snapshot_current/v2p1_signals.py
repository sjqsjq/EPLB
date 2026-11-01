"""ADW v2.1 signal layer (audit fixes F1/F2) -- pure tensor arithmetic, no
distributed calls, no I/O, CPU/GPU agnostic so it is unit-testable offline.

F1 (detection identity): the detection distribution is unified in LOGICAL
expert identity. The detection accumulator (controller._det_load) is physical-
slot indexed and kept consistent with the CURRENT physical_to_logical_map by
mirroring the count-follows-expert remap at every mapping-commit site (the
same remap the planner history already receives, controller.py:1382-1386);
therefore one gather at window end (logical_load) is exact for the whole
window, including mid-window swap commits. The detection EWMA reference lives
in logical space, whose identity is stable across swaps -- it needs NO remap
at commit (that was the v2 layout-coupling artifact).

F2 (headroom scale): headroom enters the state machine as a work FRACTION
h_rate = h_w / n_fresh (recoverable excess per recorded assignment,
dimensionless, in [0,1] because per-layer excess <= per-layer total). Both
numerator and denominator accumulate over the SAME window, so the fraction is
invariant under proportional accumulation -- this removes the window-size
scale dependence that fixed thresholds on window-cumulative counts carried
(v2 thetas were calibrated on a W-fixed-at-16 logging run).

h_w itself remains a PHYSICAL-space quantity (rank attribution is
slot // num_local, defined only on physical slots); formula identical to the
frozen v2 (controller.py:1102-1111).

NOTE ON FRAMING: these are correctness/consistency revisions. They are NOT
presented as performance improvements, and prior v2 performance results are
NOT causally attributed to the audited findings.
"""
import torch


def logical_load(det_global, p2l):
    """[L, P] physical-slot counts -> [L, E] logical-expert counts.
    p2l[l, s] = logical id currently resident at physical slot s (per layer).
    Requires det_global contents to be consistent with p2l (commit-remap
    invariant; guarded at runtime by the mapping-version tripwire)."""
    out = torch.zeros_like(det_global)
    return out.scatter_add_(1, p2l.to(torch.int64), det_global)


def row_distribution(t):
    """Per-layer normalized distribution + validity mask (zero-sum rows --
    e.g. leading dense layers -- are excluded from the change score)."""
    tf = t.float()
    rowsum = tf.sum(-1, keepdim=True)
    valid = rowsum.squeeze(-1) > 0
    p = tf / rowsum.clamp(min=1.0)
    return p, valid


def change_score(p, ref, valid):
    """s_change = mean over valid layers of (1 - cos(p_l, ref_l)).
    Returns (s_change, n_valid_layers); (None, None) when nothing is valid."""
    if ref is None:
        return None, None
    dot = (p * ref).sum(-1)
    nc = p.norm(dim=-1)
    nr = ref.norm(dim=-1)
    ok = valid & (nc > 0) & (nr > 0)
    if not bool(ok.any().item()):
        return None, None
    cos = dot[ok] / (nc[ok] * nr[ok])
    return float((1.0 - cos).mean().item()), int(ok.sum().item())


def headroom(det_global, ep_size):
    """h_w = sum_l max(0, max_rank_load_l - max(total_l/EP, max_expert_l)),
    in assignment units. PHYSICAL space (rank = slot // num_local). Identical
    formula to frozen v2."""
    gl = det_global.float()
    L, P = gl.shape
    S = P // ep_size
    rank_loads = gl.view(L, ep_size, S).sum(-1)
    total = rank_loads.sum(-1)
    max_expert = gl.max(-1).values
    floor = torch.maximum(total / ep_size, max_expert)
    excess = (rank_loads.max(-1).values - floor).clamp(min=0)
    return float(excess.sum().item())


def headroom_rate(h_w, n_fresh):
    """F2 work-fraction scale: dimensionless fraction in [0, 1]."""
    return h_w / float(n_fresh) if n_fresh > 0 else 0.0


def z_from_signals(s_change, n_fresh, n_rows, kappa):
    """z = s_change / (kappa / n_layer), n_layer = max(n_fresh / n_rows, 1).
    Same noise normalization as frozen v2 (removes the shrink->noise->shrink
    feedback); kappa is RE-FIT for the logical-space signal under the
    pre-registered v2.1 calibration (same rule FORM as step-5, new values,
    W0-only)."""
    if s_change is None:
        return None
    n_layer = max(n_fresh / float(max(n_rows, 1)), 1.0)
    return s_change / (kappa / n_layer)
