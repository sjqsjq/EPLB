"""ADW v2.1 -- limited revision of the frozen v2 window controller.

PRESERVATION: window_v2.py is NOT modified (frozen sha256 810c54da...). This
module subclasses it for exactly the three audit-verified findings of
2026-10-16. Framing rule: these are correctness/consistency revisions; they
are NOT presented as performance improvements, and prior v2 performance
results are NOT causally attributed to these findings.

  F1 (detection identity)   handled in controller.py + v2p1_signals.py:
     detection distributions unified in logical-expert identity; the detection
     accumulator mirrors the count-follows-expert remap at every mapping-
     commit site; a mapping-version tripwire guards the invariant. The state
     machine itself is unchanged by F1.

  F2 (headroom scale)       the theta_h_low/theta_h_act knob FIELDS carry
     headroom-RATE thresholds (h_w / n_fresh, dimensionless). The inherited
     comparison logic is scale-invariant because the caller passes the rate
     where v2 passed the window-cumulative count. Startup refuses unfrozen
     placeholders (0); a rate > 1 is inert by construction (logging runs only).

  F3 (confirm-then-commit, revision r2 2026-10-17, pre-freeze)
     snapshot()/restore() + confirm_fail_streak: the controller proposes one
     step(); ALL ranks then reach a JOINT commit verdict from reduced MIN/MAX
     data -- commit iff (h_min == h_max) AND (v_min == 1), i.e. identical
     decision hashes AND every rank holds a valid proposal. The verdict is a
     function of bit-identical reduced values only, so every rank necessarily
     reaches the SAME verdict (r1's per-rank predicate sum == world*h_own was
     asymmetric in principle: a rank whose hash equalled the mean of divergent
     hashes could pass alone -- eliminated in r2). Any non-unanimity or local
     error rolls back identically on every rank (hold W); two consecutive
     failures freeze W at the deterministic w_init (inherited safe mode).
     Collective schedule (r2): ALL collectives of a window -- det SUM, confirm
     MIN, confirm MAX -- are UNCONDITIONAL; the local recoverable scope
     (mapping-version tripwire, signals, proposal) contains no collective and
     runs strictly between them, so a rank-local failure can never skip or
     reorder a collective (r1 had the tripwire before the det all_reduce: an
     asymmetric local failure could misalign the det collective -> NCCL
     size-mismatch hang; fixed in r2). Error windows present a deterministic
     sentinel hash with prop_valid=0 (coordinated hold, never commit).

Event codes: all v2 codes are preserved verbatim; new codes added below.
Threshold VALUES are not frozen in this file -- they come from the
pre-registered v2.1 calibration on W0 ONLY
(results/adw_v2p1/calibration_plan_v2p1.md). Nothing here reads domain
labels, switch times, or evaluation-derived quantities (same rule as v2).
"""
import hashlib
import json

try:  # runtime (deployed package)
    from sglang.srt.managers.pb_oeplb.window_v2 import WindowControllerV2
except ImportError:  # offline unit-test context
    from window_v2 import WindowControllerV2

EV_CONFIRM_HOLD = "confirm_hold"
EV_CONFIRM_FAIL_FREEZE = "confirm_fail_freeze"

_STATE_FIELDS = ("W", "alert_k", "stable_s", "dwell", "dwell_dir",
                 "uncertain_run", "carry_windows", "episode_shrinks",
                 "window_seq")

_HASH_MOD = 1 << 58


class WindowControllerV2P1(WindowControllerV2):
    """Frozen v2 state machine + rate-scale validation + rollback support."""

    def __init__(self, w_init, knobs=None, beta=0.5):
        super().__init__(w_init, knobs, beta=beta)
        lo, hi = self.k["theta_h_low"], self.k["theta_h_act"]
        if not (0.0 < lo < hi <= 2.0):
            raise ValueError(
                "ADW v2.1: theta_h_* carry headroom-RATE thresholds "
                "(h_w/n_fresh units); require 0 < theta_r_low < theta_r_act "
                "<= 2 (rate > 1 is inert by construction -- logging runs "
                "only; placeholder zeros are refused)")
        self.confirm_fail_streak = 0

    # --- F3: propose/commit/rollback ---------------------------------------
    def snapshot(self):
        """Full pre-step state (step() is the proposal; nothing is effective
        until the controller commits after cross-rank confirmation)."""
        return {f: getattr(self, f) for f in _STATE_FIELDS}

    def restore(self, snap):
        """Idempotent rollback to a snapshot taken before step()."""
        for f in _STATE_FIELDS:
            setattr(self, f, snap[f])


def decision_hash(window_id, rec):
    """Deterministic cross-rank hash of a PROPOSED decision (F3).
    rec['h_w'] carries the headroom RATE in v2.1 (field name inherited from
    the frozen record schema). json float repr round-trips exactly, so the
    digest is bit-identical on every rank for bit-identical inputs."""
    payload = json.dumps(
        ["adw21", window_id, rec["W_after"], rec["event"],
         rec["z"], rec["h_w"], rec["n_fresh"]], sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8],
                          "big") % _HASH_MOD


def error_sentinel(window_id):
    """Deterministic participation hash for a window whose LOCAL computation
    errored. Same window_id -> same sentinel on any rank hitting the same
    failure; the 'adw21-err' namespace makes collision with a decision hash
    cryptographically improbable. A unanimous sentinel still HOLDS (commit
    requires a non-None proposal) -- coordinated hold, then streak logic."""
    return int.from_bytes(hashlib.sha256(
        ("adw21-err:%d" % window_id).encode()).digest()[:8], "big") % _HASH_MOD


def confirm_commit(h, prop_valid, reduce_fn):
    """Joint commit verdict (F3, revision r2): symmetric by construction.

    Each rank presents h (its decision hash, or the deterministic error
    sentinel) and prop_valid (1 = this rank holds a step() proposal,
    0 = local recoverable error). reduce_fn(pair, op) performs an in-place
    elementwise all_reduce with op in {"min", "max"} on the 2-element list
    [h, prop_valid] -- two small int64 collectives per window in FIXED order
    (MIN then MAX), unconditional on every rank.

    The verdict is computed ONLY from the reduced pairs, which are
    bit-identical on every rank:
        unanimous  <=>  h_min == h_max   (identical hashes everywhere)
        all_valid  <=>  v_min == 1       (every rank holds a proposal)
        commit     <=>  unanimous and all_valid
    so all ranks necessarily derive the same verdict. MIN/MAX never enlarge
    magnitudes: no overflow (h < 2^58 < int64 max; prop_valid in {0,1}).
    Returns (commit, h_min, h_max, v_min) -- the last three for logging.
    """
    lo = [h, prop_valid]
    hi = [h, prop_valid]
    reduce_fn(lo, "min")
    reduce_fn(hi, "max")
    commit = (lo[0] == hi[0]) and (lo[1] == 1)
    return commit, lo[0], hi[0], lo[1]
